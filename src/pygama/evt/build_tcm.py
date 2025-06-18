from __future__ import annotations

import logging
import re
from pathlib import Path

import awkward as ak
import lgdo
from lgdo import lh5
from lgdo.types import Struct, Table, VectorOfVectors

from . import tcm as ptcm

log = logging.getLogger(__name__)


def readd_attrs(final_table, input_table):
    if hasattr(input_table, "attrs"):
        final_table.attrs = input_table.attrs
    for key in input_table:
        if hasattr(input_table[key], "attrs"):
            final_table[key].attrs = input_table[key].attrs
    if isinstance(final_table[key], (Table, Struct)):
        readd_attrs(final_table[key], input_table[key])
    if isinstance(input_table, VectorOfVectors):
        final_table[key].flattened_data.attrs = input_table[key].flattened_data.attrs
        final_table[key].cumulative_length.attrs = input_table[
            key
        ].cumulative_length.attrs


def _concat_tables(tbls):
    out_tbl = tbls[0].view_as("ak")
    for tbl in tbls[1:]:
        out_tbl = ak.concatenate([out_tbl, tbl.view_as("ak")], axis=0)
    out_tbl = Table(col_dict=out_tbl)
    readd_attrs(out_tbl, tbls[0])
    return out_tbl


def build_tcm(
    input_tables: list[tuple[str, str | list[str]]],
    coin_cols: str | list[str],
    hash_func: str | None = r"\d+",
    coin_windows: float | list[float] = 0,
    window_refs: str | list[str] = "last",
    out_file: str | None = None,
    out_name: str = "tcm",
    wo_mode: str = "write_safe",
    buffer_len: int | None = None,
    out_fields: str | list[str] | None = None,
) -> lgdo.Table | None:
    r"""Build a Time Coincidence Map (TCM).

    Given a list of input tables, create an output table containing an entry
    list of coincidences among the inputs. Uses
    :func:`.evt.tcm.generate_tcm_cols`. For use with the
    :class:`~.flow.data_loader.DataLoader`.

    Parameters
    ----------
    input_tables
        Each entry is ``(filename, table_name_pattern)``. ``table_name_pattern``
        may be a string or list of strings. All tables matching each pattern in
        ``filename`` will be used as input tables.
    coin_cols
        Name of the column (or columns) in each table used to build
        coincidences. All input tables must contain these columns.
    hash_func
        mapping of table names to integers for use in the TCM.  `hash_func` is
        a regexp pattern that acts on each table name. The default `hash_func`
        ``r"\d+"`` pulls the first integer out of the table name. Setting to
        ``None`` will use a table's index in `input_tables`.
    coin_windows
        Width of the clustering window(s). If a single value is supplied it will
        be used for all ``coin_cols``.
    window_refs
        Window reference for the clustering window. Currently only ``"last"`` is
        implemented.
    out_file
        name (including path) for the output file. If ``None``, no file will be
        written; the TCM will just be returned in memory.
    out_name
        name for the TCM table in the output file.
    wo_mode
        mode to send to :meth:`~.lgdo.lh5.LH5Store.write`.

    out_fields
        Optional additional fields to propagate from the input tables into the
        output TCM.

    Returns
    -------
    lgdo.Table or None
        If ``out_file`` is ``None`` the resulting TCM is returned as a
        :class:`lgdo.Table`. Otherwise ``None`` is returned after writing the
        table to ``out_file``.

    See Also
    --------
    .tcm.generate_tcm_cols
    """
    # hash_func: later can add list or dict or a function(str) --> int.

    if not isinstance(coin_cols, list):
        coin_cols = [coin_cols]
    if not isinstance(coin_windows, list):
        coin_windows = [coin_windows]
    if not isinstance(window_refs, list):
        window_refs = [window_refs]
    if out_fields is not None and not isinstance(out_fields, list):
        out_fields = [out_fields]

    if len(coin_cols) != len(coin_windows):
        if len(coin_windows) == 1:
            coin_windows = coin_windows * len(coin_cols)
        else:
            msg = (
                "coin_cols and coin_windows must have the same length, "
                f"got {len(coin_cols)} and {len(coin_windows)}"
            )
            raise ValueError(msg)

    if len(coin_cols) != len(window_refs):
        if len(window_refs) == 1:
            window_refs = window_refs * len(coin_cols)
        else:
            msg = (
                "coin_cols and coin_windows must have the same length, "
                f"got {len(coin_cols)} and {len(window_refs)}"
            )
            raise ValueError(msg)

    _filenames = [tpl[0] for tpl in input_tables]
    if len(set(_filenames)) != len(_filenames):
        msg = "file names specified multiple times in input_tables"
        raise ValueError(msg)

    iterators = []
    table_keys = []
    all_tables = []

    # determine buffer length automatically
    if buffer_len is None:
        ntables = 0
        for filename, patterns in input_tables:
            if isinstance(patterns, str):
                patterns = [patterns]
            for pattern in patterns:
                ntables += len(lh5.ls(filename, lh5_group=pattern))

        n_fields = (
            2 + len(set(coin_cols + out_fields))
            if out_fields is not None
            else 2 + len(set(coin_cols))
        )
        buffer_len = int(10**7 / (ntables * n_fields))

    # loop over files
    for filename, patterns in input_tables:
        if isinstance(patterns, str):
            patterns = [patterns]

        # make a list of tables in the file
        for pattern in patterns:
            for table in lh5.ls(filename, lh5_group=pattern):
                all_tables.append(table)

        for table_idx, table in enumerate(all_tables):
            if hash_func is not None:
                if isinstance(hash_func, str):
                    table_key = int(re.search(hash_func, table).group())
                else:
                    raise NotImplementedError(
                        f"hash_func of type {type(hash_func).__name__}"
                    )
            else:
                table_key = table_idx

            h5py_open_mode = "a" if out_file == filename else "r"

            iterators.append(
                lh5.LH5Iterator(
                    filename,
                    table,
                    field_mask=coin_cols,
                    buffer_len=buffer_len,
                    h5py_open_mode=h5py_open_mode,
                )
            )
            table_keys.append(table_key)

    coin_windows = [
        ptcm.coin_groups(n, w, r)
        for n, w, r in zip(coin_cols, coin_windows, window_refs)
    ]

    tcm_gen = ptcm.generate_tcm_cols(
        iterators, coin_windows=coin_windows, table_keys=table_keys, fields=out_fields
    )
    tcm = []
    # clear existing output files
    if out_file is not None and wo_mode == "of":
        if Path(out_file).exists():
            Path(out_file).unlink()

    wrote_first = False
    while True:
        try:
            out_tbl = tcm_gen.__next__()
            out_tbl.attrs.update(
                {"tables": str(all_tables), "hash_func": str(hash_func)}
            )
            if out_file is not None:
                lh5.write(
                    out_tbl,
                    out_name,
                    out_file,
                    wo_mode=wo_mode if not wrote_first else "a",
                )
                wrote_first = True
            else:
                tcm.append(out_tbl)
        except StopIteration:
            break

    if out_file is None:
        out_tbl = _concat_tables(tcm)
        return out_tbl
