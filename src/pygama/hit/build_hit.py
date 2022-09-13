"""
This module implements routines to evaluate expressions to columnar data.
"""
from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict

import numpy as np

from pygama.lgdo import LH5Iterator, LH5Store, ls

log = logging.getLogger(__name__)


def build_hit(
    infile: str,
    outfile: str = None,
    hit_config: str | dict = None,
    lh5_tables: list[str] = None,
    lh5_tables_config: str | dict[str] = None,
    n_max: int = np.inf,
    wo_mode: str = "write_safe",
    buffer_len: int = 3200,
) -> None:
    """
    Transform a :class:`~.lgdo.Table` into a new :class:`~.lgdo.Table` by
    evaluating strings describing column operations.

    Operates on columns only, not specific rows or elements.

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
                        "expression": "sqrt(@a + @b * trapEmax**2)",
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
        forwarded to :meth:`~.lgdo.lh5_store.write_object`.
    """
    store = LH5Store()

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
            with open(tbl_cfg) as f:
                tbl_cfg = json.load(f)

        for (k, v) in tbl_cfg.items():
            if isinstance(v, str):
                with open(v) as f:
                    # order in hit configs is important (dependencies)
                    tbl_cfg[k] = json.load(f, object_pairs_hook=OrderedDict)
        lh5_tables_config = tbl_cfg

    else:
        if isinstance(hit_config, str):
            # sanitize config
            with open(hit_config) as f:
                # order in hit configs is important (dependencies)
                hit_config = json.load(f, object_pairs_hook=OrderedDict)

        if lh5_tables is None:
            lh5_tables_config = {}
            if "dsp" in ls(infile):
                log.debug("found candidate table /dsp")
                lh5_tables_config["dsp"] = hit_config
            for el in ls(infile):
                if f"{el}/dsp" in ls(infile, f"{el}/"):
                    log.debug(f"found candidate table /{el}/dsp")
                    lh5_tables_config[f"{el}/dsp"] = hit_config

    if outfile is None:
        outfile = os.path.splitext(os.path.basename(infile))[0]
        outfile = outfile.removesuffix("_dsp") + "_hit.lh5"

    first_done = False
    for (tbl, cfg) in lh5_tables_config.items():
        lh5_it = LH5Iterator(infile, tbl, buffer_len=buffer_len)
        tot_n_rows = store.read_n_rows(tbl, infile)
        write_offset = 0

        log.info(f"Processing table '{tbl}' in file {infile}")

        for tbl_obj, start_row, n_rows in lh5_it:
            n_rows = min(tot_n_rows - start_row, n_rows)

            outtbl_obj = tbl_obj.eval(cfg["operations"])

            # remove or add columns according to "outputs" in the configuration
            # dictionary
            if "outputs" in cfg:
                if isinstance(cfg["outputs"], list):
                    # add missing columns (forwarding)
                    for out in cfg["outputs"]:
                        if out not in outtbl_obj.keys():
                            outtbl_obj.add_column(out, tbl_obj[out])

                    # remove non-required columns
                    existing_cols = list(outtbl_obj.keys())
                    for col in existing_cols:
                        if col not in cfg["outputs"]:
                            outtbl_obj.remove_column(col, delete=True)

            store.write_object(
                obj=outtbl_obj,
                name=tbl.replace("/dsp", "/hit"),
                lh5_file=outfile,
                n_rows=n_rows,
                wo_mode=wo_mode if first_done is False else "append",
                write_start=write_offset + start_row,
            )

            first_done = True
