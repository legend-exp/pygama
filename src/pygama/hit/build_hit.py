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
    hit_config: str | dict,
    outfile: str = None,
    lh5_tables: list[str] = None,
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
    hit_config
        dictionary or name of JSON file defining column transformations.
    outfile
        name of the output LH5 file. If ``None``, create a file in the same
        directory and append `_hit` to its name.
    lh5_tables
        tables to consider in the input file. if ``None``, tables with name
        `dsp` will be searched for in the file, even nested by one level.
    n_max
        maximum number of rows to process
    wo_mode
        forwarded to :meth:`~.lgdo.lh5_store.write_object`.
    """
    store = LH5Store()

    if lh5_tables is None:
        lh5_tables = []
        if "dsp" in ls(infile):
            lh5_tables.append("dsp")
        for el in ls(infile):
            if f"{el}/dsp" in ls(infile, f"{el}/"):
                lh5_tables.append(f"{el}/dsp")

    log.debug(f"found candidate tables: {lh5_tables}")

    if outfile is None:
        outfile = os.path.splitext(os.path.basename(infile))[0]
        outfile = outfile.removesuffix("_dsp") + "_hit.lh5"

    if isinstance(hit_config, str):
        with open(hit_config) as f:
            hit_config = json.load(f, object_pairs_hook=OrderedDict)

    for tbl in lh5_tables:
        lh5_it = LH5Iterator(infile, tbl, buffer_len=buffer_len)
        tot_n_rows = store.read_n_rows(tbl, infile)
        write_offset = 0

        log.info(f"Processing table '{tbl}' in file {infile}")

        for tbl_obj, start_row, n_rows in lh5_it:
            n_rows = min(tot_n_rows - start_row, n_rows)

            outtbl_obj = tbl_obj.eval(hit_config)

            store.write_object(
                obj=outtbl_obj,
                name=tbl.replace("/dsp", "/hit"),
                lh5_file=outfile,
                n_rows=n_rows,
                wo_mode=wo_mode,
                write_start=write_offset + start_row,
            )
