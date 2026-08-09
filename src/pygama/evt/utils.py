"""
This module provides utilities to build the `evt` tier.
"""

from __future__ import annotations

import copy
import re
from collections import namedtuple

import awkward as ak
import h5py
import lh5
import numpy as np
from numpy.typing import NDArray

H5DataLoc = namedtuple(
    "H5DataLoc", ("file", "group", "table_fmt"), defaults=3 * (None,)
)
DataInfo = namedtuple("DataInfo", ("raw", "tcm", "evt"), defaults=3 * (None,))

TCMData = namedtuple(
    "TCMData", ("table_key", "row_in_table", "cache"), defaults=(None,)
)


class EvtCache:
    """Memoises the work that :func:`build_evt_cols` would otherwise repeat.

    An `evt` configuration evaluates many operations over the same channels,
    and each operation is evaluated again for every TCM chunk. Without
    memoisation the same HDF5 table is listed, the same lower-tier column is
    read, and the same per-channel TCM index arrays are rebuilt once per
    (chunk, operation, channel) — which is why calibration files, holding
    ~200x more events than physics files and so ~100x more chunks, are so much
    slower to process.

    Three caches with two different lifetimes:

    `fields`
        which columns each ``(tier, channel)`` table holds. File-scoped: the
        input files do not change between chunks.
    `code`
        compiled expression objects, keyed by expression string. File-scoped.
    `cols`, `idx`
        lower-tier column data and per-channel TCM indices. Chunk-scoped, and
        must be dropped by :meth:`new_chunk` because both are tied to the rows
        of the chunk currently being processed.
    """

    def __init__(self):
        self.fields = {}
        self.code = {}
        self.cols = {}
        self.idx = {}
        self._n_hits = 0

    def new_chunk(self) -> None:
        """Drop everything tied to the previous chunk's rows."""
        self.cols.clear()
        self.idx.clear()
        self._n_hits = 0

    def table_fields(self, datainfo, tier_name: str, ch: str) -> frozenset:
        """Names of the columns held by ``ch``'s table in tier `tier_name`."""
        key = (tier_name, ch)
        if key not in self.fields:
            tier = datainfo._asdict()[tier_name]
            self.fields[key] = frozenset(
                k.split("/")[-1]
                for k in lh5.ls(tier.file, f"{ch.replace('/', '')}/{tier.group}/")
            )
        return self.fields[key]

    def table_names(self, datainfo, tier_name: str = "hit") -> frozenset:
        """Names of the tables (channels) present in tier `tier_name`."""
        key = (tier_name, None)
        if key not in self.fields:
            tier = datainfo._asdict()[tier_name]
            self.fields[key] = frozenset(lh5.ls(tier.file))
        return self.fields[key]

    def compile(self, expr: str):
        """Compile `expr` once and reuse the code object across channels."""
        if expr not in self.code:
            self.code[expr] = compile(expr, "<evt expression>", "eval")
        return self.code[expr]

    def channel_indices(self, tcm, table_id: int):
        """``(chan_tcm_indexs, idx_ch, evt_ids_ch)`` for one channel.

        Computed for every channel at once on first use, because the
        per-channel form is a full jagged reduction over the chunk's TCM and
        every aggregator needs it for every operation.
        """
        if not self.idx:
            self._build_indices(tcm)
        entry = self.idx.get(int(table_id))
        if entry is None:
            empty_mask = np.zeros(self._n_hits, dtype=bool)
            empty = np.empty(0, dtype=np.int64)
            return empty_mask, empty, empty
        return entry

    def _build_indices(self, tcm) -> None:
        flat_key = ak.flatten(tcm.table_key).to_numpy()
        flat_row = ak.flatten(tcm.row_in_table).to_numpy()
        counts = ak.num(tcm.table_key, axis=1).to_numpy()
        # event index of every hit, in flattened (event-major) order
        evt_of_hit = np.repeat(np.arange(len(counts)), counts)
        self._n_hits = len(flat_key)
        for table_id in np.unique(flat_key):
            mask = flat_key == table_id
            self.idx[int(table_id)] = (mask, flat_row[mask], evt_of_hit[mask])


def table_names(datainfo, tier_name="hit", cache=None) -> frozenset:
    """Names of the tables (channels) present in tier `tier_name`.

    Loop-invariant, so callers should hoist it out of per-channel loops; the
    `cache` also keeps it to one listing per file.
    """
    if cache is not None:
        return cache.table_names(datainfo, tier_name)
    return frozenset(lh5.ls(datainfo._asdict()[tier_name].file))


def channel_indices(tcm, table_id):
    """Locate one channel's hits within the current TCM chunk.

    Returns ``(chan_tcm_indexs, idx_ch, evt_ids_ch)``: the mask selecting this
    channel's entries among the chunk's flattened hits, the rows to read from
    the channel's lower-tier table, and the event each of those hits belongs
    to. Served from ``tcm.cache`` when there is one — every aggregator needs
    these for every operation, and recomputing them is a full jagged reduction
    over the chunk.

    Raises
    ------
    ValueError
        If *table_id* is ``None``, i.e. the channel name did not match the
        table format.  :func:`get_tcm_id_by_pattern` returns ``None`` for those,
        and callers that mean to skip them must filter on it (as
        :func:`~pygama.evt.build_evt.build_evt` does) rather than pass the
        ``None`` through.
    """
    if table_id is None:
        msg = (
            "cannot locate hits for a channel whose name does not match the "
            "table format (get_tcm_id_by_pattern returned None); filter such "
            "channels out or correct the channel name in the config"
        )
        raise ValueError(msg)

    cache = getattr(tcm, "cache", None)
    if cache is not None:
        return cache.channel_indices(tcm, table_id)

    chan_tcm_indexs = (ak.flatten(tcm.table_key) == table_id).to_numpy()
    idx_ch = ak.flatten(tcm.row_in_table)[chan_tcm_indexs].to_numpy()
    evt_ids_ch = np.repeat(
        np.arange(0, len(tcm.table_key)), ak.sum(tcm.table_key == table_id, axis=1)
    )
    return chan_tcm_indexs, idx_ch, evt_ids_ch


def make_files_config(data: dict):
    if not isinstance(data, tuple):
        if "raw" not in data:
            data["raw"] = (None,)
        if "tcm" not in data:
            data["tcm"] = (None,)
        if "evt" not in data:
            data["evt"] = (None,)
        DataInfo = namedtuple(
            "DataInfo", tuple(data.keys()), defaults=len(data.keys()) * (None,)
        )
        return DataInfo(
            *[
                H5DataLoc(*data[tier]) if tier in data else H5DataLoc()
                for tier in DataInfo._fields
            ]
        )

    return data


def make_numpy_full(size, fill_value, try_dtype):
    """Allocate an array of *size* filled with *fill_value*, typed to hold both.

    *try_dtype* is the dtype of the data that will later be written into the
    array, so the result must accommodate *fill_value* **and** *try_dtype*.
    Promoting is essential rather than cosmetic: an integer ``initial`` value
    against float32 data cannot be cast to float32, and falling back to the
    fill value's own dtype would hand back an *integer* accumulator for float
    data, so that ``out += res`` raises ``UFuncOutputCastingError``.

    *fill_value* is passed through as a scalar, so NEP 50 weak promotion keeps
    the data's dtype (``0`` with float32 data gives float32, not float64)
    rather than widening every accumulator.
    """
    return np.full(size, fill_value, dtype=np.result_type(fill_value, try_dtype))


def copy_lgdo_attrs(obj):
    attrs = copy.copy(obj.attrs)
    attrs.pop("datatype")
    return attrs


def get_lgdo_attrs(datainfo, channels, tier_name, field):
    """Read LGDO attrs for a field from the first available channel in the LH5 file.

    Uses a metadata-only h5py read — no data is loaded from disk.

    Parameters
    ----------
    datainfo
        input/output LH5 datainfo.
    channels
        flat list of channel strings (e.g. ``["ch1000000", "ch1000001"]``).
    tier_name
        name of the tier (e.g. ``"hit"``).
    field
        field name within the tier table.
    """
    tier = datainfo._asdict().get(tier_name)
    if tier is None or tier.file is None:
        return {}

    # tier.file is an open handle once build_evt_cols has taken ownership of
    # the input files; only close what we opened ourselves
    if isinstance(tier.file, h5py.Group):
        return _first_field_attrs(tier.file, channels, tier.group, field)

    with h5py.File(tier.file, "r") as f:
        return _first_field_attrs(f, channels, tier.group, field)


def _first_field_attrs(f, channels, group, field):
    for ch in channels:
        path = f"{ch.replace('/', '')}/{group}/{field}"
        if path not in f:
            continue
        attrs = dict(f[path].attrs)
        attrs.pop("datatype", None)
        return attrs
    return {}


def get_tcm_id_by_pattern(table_id_fmt: str, ch: str) -> int:
    pre = table_id_fmt.split("{", maxsplit=1)[0]
    post = table_id_fmt.split("}")[1]
    try:
        return int(ch.strip(pre).strip(post))
    except ValueError:
        return None


def get_table_name_by_pattern(table_id_fmt: str, ch_id: int) -> str:
    # check table_id_fmt validity
    pattern_check = re.findall(r"{([^}]*?)}", table_id_fmt)[0]
    if pattern_check == "" or pattern_check[0] == ":":
        return table_id_fmt.format(ch_id)
    msg = "only empty placeholders {} in format specifications are currently supported"
    raise NotImplementedError(msg)


def find_parameters(
    datainfo,
    ch,
    idx_ch,
    field_list,
    cache=None,
) -> dict:
    """Finds and returns parameters from non `tcm`, `evt` tiers.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    ch
       "rawid" in the tiers.
    idx_ch
       index array of entries to be read from datainfo.
    field_list
       list of tuples ``(tier, field)`` to be found in non `tcm`, `evt` tiers.
    cache
       optional :class:`EvtCache`. When given, table listings are looked up
       instead of read from the file, and each column is read from disk at
       most once per chunk however many operations reference it.
    """
    if not isinstance(datainfo, DataInfo):
        datainfo = make_files_config(datainfo)

    final_dict = {}

    for name, tier in datainfo._asdict().items():
        if name in ("tcm", "evt") or tier.file is None:  # skip other tables
            continue

        # nothing from this tier is referenced: never touch the file
        wanted = [e[1] for e in field_list if e[0] == name]
        if not wanted:
            continue

        if cache is not None:
            keys = cache.table_fields(datainfo, name, ch)
        else:
            keys = {
                k.split("/")[-1]
                for k in lh5.ls(tier.file, f"{ch.replace('/', '')}/{tier.group}/")
            }
        flds = [f for f in wanted if f in keys]

        if not flds:
            continue

        if cache is None:
            tier_ak = lh5.read_as(
                f"{ch.replace('/', '')}/{tier.group}/",
                tier.file,
                field_mask=flds,
                idx=idx_ch,
                library="ak",
            )
            final_dict |= dict(
                zip(
                    [f"{name}_" + e for e in ak.fields(tier_ak)],
                    ak.unzip(tier_ak),
                    strict=False,
                )
            )
            continue

        # read only the columns this chunk has not already loaded
        missing = list(
            dict.fromkeys(f for f in flds if (name, ch, f) not in cache.cols)
        )
        if missing:
            tier_ak = lh5.read_as(
                f"{ch.replace('/', '')}/{tier.group}/",
                tier.file,
                field_mask=missing,
                idx=idx_ch,
                library="ak",
            )
            for fld, col in zip(ak.fields(tier_ak), ak.unzip(tier_ak), strict=False):
                cache.cols[(name, ch, fld)] = col

        for fld in flds:
            # a requested column can still be absent if the read dropped it
            if (name, ch, fld) in cache.cols:
                final_dict[f"{name}_{fld}"] = cache.cols[(name, ch, fld)]

    return final_dict


def get_data_at_channel(
    datainfo,
    ch,
    tcm,
    expr,
    field_list,
    pars_dict,
) -> NDArray:
    """Evaluates an expression and returns the result.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    ch
       "rawid" of channel to be evaluated.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    expr
       expression to be evaluated.
    field_list
       list of parameter-tuples ``(root_group, field)`` found in the expression.
    pars_dict
       dict of additional parameters that are not channel dependent.
    is_evaluated
       if false, the expression does not get evaluated but an array of default
       values is returned.
    default_value
       default value.
    """
    if not isinstance(datainfo, DataInfo):
        datainfo = make_files_config(datainfo)
    table_id = get_tcm_id_by_pattern(datainfo.hit.table_fmt, ch)
    cache = getattr(tcm, "cache", None)

    # get index list for this channel to be loaded
    chan_tcm_indexs, idx_ch, _ = channel_indices(tcm, table_id)
    outsize = len(idx_ch)

    if expr == "tcm.table_key":
        res = np.full(outsize, table_id, dtype=int)
    elif expr == "tcm.row_in_table":
        res = idx_ch
    elif expr == "tcm.index":
        res = np.where(chan_tcm_indexs)[0]
    else:
        var = find_parameters(
            datainfo=datainfo,
            ch=ch,
            idx_ch=idx_ch,
            field_list=field_list,
            cache=cache,
        )

        if pars_dict is not None:
            var = var | pars_dict

        # evaluate expression
        # move tier+dots in expression to underscores (e.g. evt.foo -> evt_foo)

        new_expr = expr
        for name in datainfo._asdict():
            if name == "evt":
                new_expr = new_expr.replace(f"{name}.", "")
            elif name not in ["tcm", "raw"]:
                new_expr = new_expr.replace(f"{name}.", f"{name}_")

        # compiled once per expression, then reused for every channel
        code = cache.compile(new_expr) if cache is not None else new_expr
        res = eval(
            code,
            var,
        )

        # in case the expression evaluates to a single value blow it up
        if not hasattr(res, "__len__") or isinstance(res, str):
            return np.full(outsize, res)

        # the resulting arrays need to be 1D from the operation,
        # this can only change once we support larger than two dimensional LGDOs
        # ak.to_numpy() raises error if array not regular
        res = ak.to_numpy(res, allow_missing=False)

        # in this method only 1D values are allowed
        if res.ndim > 1:
            msg = (
                f"expression '{expr}' must return 1D array. If you are using "
                "VectorOfVectors or ArrayOfEqualSizedArrays, use awkward "
                "reduction functions to reduce the dimension"
            )
            raise ValueError(msg)

    return res


def get_mask_from_query(
    datainfo,
    query,
    length,
    ch,
    idx_ch,
    cache=None,
) -> NDArray:
    """Evaluates a query expression and returns a mask accordingly.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    query
       query expression.
    length
       length of the return mask.
    ch
       "rawid" of channel to be evaluated.
    idx_ch
       channel indices to be read.
    cache
       optional :class:`EvtCache`, forwarded to :func:`find_parameters`.
    """
    if not isinstance(datainfo, DataInfo):
        datainfo = make_files_config(datainfo)

    # get sub evt based query condition if needed
    if isinstance(query, str):
        query_lst = re.findall(
            rf"({'|'.join(datainfo._asdict().keys())}).([a-zA-Z_$][\w$]*)", query
        )
        query_var = find_parameters(
            datainfo=datainfo,
            ch=ch,
            idx_ch=idx_ch,
            field_list=query_lst,
            cache=cache,
        )

        new_query = query
        for name in datainfo._asdict():
            if name not in ["tcm", "evt"]:
                new_query = new_query.replace(f"{name}.", f"{name}_")

        code = cache.compile(new_query) if cache is not None else new_query
        limarr = eval(
            code,
            query_var,
        )

        # in case the expression evaluates to a single value blow it up
        if (not hasattr(limarr, "__len__")) or (isinstance(limarr, str)):
            return np.full(len(idx_ch), limarr)

        limarr = ak.to_numpy(limarr, allow_missing=False)
        if limarr.ndim > 1:
            msg = (
                f"query '{query}' must return 1D array. If you are using "
                "VectorOfVectors or ArrayOfEqualSizedArrays, use awkward "
                "reduction functions to reduce the dimension"
            )
            raise ValueError(msg)

    # or forward the array
    elif isinstance(query, np.ndarray):
        limarr = query

    # if no condition, it must be true
    else:
        limarr = np.ones(length).astype(bool)

    # explicit cast to bool
    if limarr.dtype != bool:
        limarr = limarr.astype(bool)

    return limarr
