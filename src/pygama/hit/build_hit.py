"""
This module implements routines to evaluate expressions to columnar data.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Iterable, Mapping

import lgdo
import lgdo.lh5 as lh5
import numpy as np
from lgdo.lh5 import LH5Iterator, ls

from .. import utils

log = logging.getLogger(__name__)


def build_hit(
    infile: str,
    outfile: str = None,
    hit_config: str | Mapping = None,
    lh5_tables: Iterable[str] = None,
    lh5_tables_config: str | Mapping[str, Mapping] = None,
    n_max: int = np.inf,
    wo_mode: str = "write_safe",
    buffer_len: int = 3200,
) -> None:
    """
    Transform a :class:`~lgdo.types.table.Table` into a new
    :class:`~lgdo.types.table.Table` by evaluating strings describing column
    operations.

    Operates on columns only, not specific rows or elements. Relies on
    :meth:`~lgdo.types.table.Table.eval`.

    Parameters
    ----------
    infile
        input LH5 file name containing tables to be processed.
    outfile
        name of the output LH5 file. If ``None``, create a file in the same
        directory and append `_hit` to its name.
    hit_config
        dictionary or name of JSON file defining column transformations. Must
        contain an ``outputs`` and an ``operations`` keys. For example:

        .. code-block:: json

            {
              "outputs": ["calE", "AoE"],
              "operations": {
                "calE": {
                  "expression": "sqrt(a + b * trapEmax**2)",
                  "parameters": {"a": "1.23", "b": "42.69"},
                },
                "AoE": {"expression": "A_max/calE"},
              }
            }

        The ``outputs`` array lists columns that will be effectively written in
        the output LH5 file. Add here columns that will be simply forwarded as
        they are from the DSP tier.

    lh5_tables
        tables to consider in the input file. if ``None``, tables with name
        `dsp` will be searched for in the file, even nested by one level.
    lh5_tables_config
        dictionary or JSON file defining the mapping between LH5 tables in
        `infile` and hit configuration. Table names can be directly mapped to
        configuration blocks or to JSON files containing them. This option is
        mutually exclusive with `hit_config` and `lh5_tables`.
    n_max
        maximum number of rows to process
    wo_mode
        forwarded to :meth:`lgdo.lh5.write`.

    See Also
    --------
    lgdo.types.table.Table.eval
    """

    if lh5_tables_config is None and hit_config is None:
        raise ValueError("either lh5_tables_config or hit_config must be specified")

    if lh5_tables_config is not None and (
        hit_config is not None or lh5_tables is not None
    ):
        raise ValueError(
            "lh5_tables_config and hit_config/lh5_tables options are mutually exclusive"
        )

    if lh5_tables_config is not None:
        tbl_cfg = lh5_tables_config
        # sanitize config
        if isinstance(tbl_cfg, str):
            tbl_cfg = utils.load_dict(tbl_cfg)

        for k, v in tbl_cfg.items():
            if isinstance(v, str):
                tbl_cfg[k] = utils.load_dict(v)
        lh5_tables_config = tbl_cfg

    else:
        if isinstance(hit_config, str):
            # sanitize config
            hit_config = utils.load_dict(hit_config)

        lh5_tables_config = {}
        if lh5_tables is None:
            if "dsp" in ls(infile):
                log.debug("found candidate table /dsp")
                lh5_tables_config["dsp"] = hit_config
            for el in ls(infile):
                if f"{el}/dsp" in ls(infile, f"{el}/"):
                    log.debug(f"found candidate table /{el}/dsp")
                    lh5_tables_config[f"{el}/dsp"] = hit_config
        else:
            for tbl in lh5_tables:
                lh5_tables_config[tbl] = hit_config

    if outfile is None:
        outfile = os.path.splitext(os.path.basename(infile))[0]
        outfile = outfile.removesuffix("_dsp") + "_hit.lh5"

    # reorder blocks in "operations" based on dependency
    log.debug("reordering operations based on mutual dependency")
    for cfg in lh5_tables_config.values():
        cfg["operations"] = _reorder_table_operations(cfg["operations"])

    first_done = False
    for tbl, cfg in lh5_tables_config.items():
        lh5_it = LH5Iterator(infile, tbl, buffer_len=buffer_len)
        write_offset = 0

        log.info(f"Processing table '{tbl}' in file {infile}")

        for tbl_obj in lh5_it:
            start_row = lh5_it.current_i_entry

            # create a new table object that links all the columns in the
            # current table (i.e. no copy)
            outtbl_obj = lgdo.Table(col_dict=tbl_obj)

            for outname, info in cfg["operations"].items():
                outcol = outtbl_obj.eval(
                    info["expression"], info.get("parameters", None)
                )
                if "lgdo_attrs" in info:
                    outcol.attrs |= info["lgdo_attrs"]

                log.debug(f"made new column {outname!r}={outcol!r}")
                outtbl_obj.add_column(outname, outcol)

            # make high level flags
            if "aggregations" in cfg:
                for high_lvl_flag, flags in cfg["aggregations"].items():
                    flags_list = list(flags.values())
                    n_flags = len(flags_list)
                    if n_flags <= 8:
                        flag_dtype = np.uint8
                    elif n_flags <= 16:
                        flag_dtype = np.uint16
                    elif n_flags <= 32:
                        flag_dtype = np.uint32
                    else:
                        flag_dtype = np.uint64

                    df_flags = outtbl_obj.view_as("pd", cols=flags_list)
                    flag_values = df_flags.values.astype(flag_dtype)

                    multiplier = 2 ** np.arange(n_flags, dtype=flag_values.dtype)
                    flag_out = np.dot(flag_values, multiplier)

                    outtbl_obj.add_field(high_lvl_flag, lgdo.Array(flag_out))

            # remove or add columns according to "outputs" in the configuration
            # dictionary
            if "outputs" in cfg:
                if isinstance(cfg["outputs"], list):
                    # add missing columns (forwarding)
                    for out in cfg["outputs"]:
                        if out not in outtbl_obj:
                            outtbl_obj.add_column(out, tbl_obj[out])

                    # remove non-required columns
                    existing_cols = list(outtbl_obj.keys())
                    for col in existing_cols:
                        if col not in cfg["outputs"]:
                            outtbl_obj.remove_column(col, delete=True)

            lh5.write(
                obj=outtbl_obj,
                name=tbl.replace("/dsp", "/hit"),
                lh5_file=outfile,
                n_rows=len(tbl_obj),
                wo_mode=wo_mode if first_done is False else "append",
                write_start=write_offset + start_row,
            )

            first_done = True


def _reorder_table_operations(
    config: Mapping[str, Mapping],
) -> OrderedDict[str, Mapping]:
    """Reorder operations in `config` according to mutual dependency."""

    def _one_pass(config):
        """Loop once over `config` and do a first round of reordering"""
        # list to hold reordered config keys
        ordered_keys = []

        # start looping over config
        for outname in config:
            # initialization
            if not ordered_keys:
                ordered_keys.append(outname)
                continue

            if outname in ordered_keys:
                raise RuntimeError(f"duplicated operation '{outname}' detected")

            # loop over existing reordered keys and figure out where to place
            # the new key
            idx = 0
            for k in ordered_keys:
                # get valid names in the expression
                c = compile(
                    config[k]["expression"], "gcc -O3 -ffast-math build_hit.py", "eval"
                )

                # if we need "outname" for this expression, insert it before!
                if outname in c.co_names:
                    break
                else:
                    idx += 1

            ordered_keys.insert(idx, outname)

        # now replay the config dictionary based on sorted keys
        opdict = OrderedDict()
        for k in ordered_keys:
            opdict[k] = config[k]

        return opdict

    # okay, now we need to repeat this until we've sorted everything
    current = OrderedDict(config)

    while True:
        new = _one_pass(current)

        if new == current:
            return new
        else:
            current = new


def _get_dependencies(config, par, pars=None):
    """
    Recursive func to iterate back through tree of input blocks for a given output block
    """
    if pars is None:
        pars = []
    par_op = config[par]
    c = compile(par_op["expression"], "gcc -O3 -ffast-math build_hit.py", "eval")
    for p in c.co_names:
        if p in par_op["parameters"]:
            pass
        else:
            pars.append(p)
            if p in config:
                pars = _get_dependencies(config, p, pars)
    return pars


def _remove_uneeded_operations(config, outpars):
    """
    Function that removes any operations not needed to generate outpars from the config dictionary
    Returns the config without these blocks as well as a list of input keys from the dsp file
    needed to generate outpars
    """
    if not isinstance(outpars, list):
        outpars = [outpars]
    dependent_keys = [*outpars]
    inkeys = []
    for par in outpars:
        pars = _get_dependencies(config, par)
        for p in pars:
            if p in config and p not in dependent_keys:
                dependent_keys.append(p)
            elif p not in config and p not in inkeys:
                inkeys.append(p)

    for key in list(config):
        if key not in dependent_keys:
            config.pop(key)
    return config, inkeys
