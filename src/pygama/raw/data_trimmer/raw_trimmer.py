from __future__ import annotations

import logging
import os

import h5py

import pygama.lgdo as lgdo
from pygama.lgdo import LH5Store
from pygama.raw.data_trimmer.data_trimmer import data_trimmer

log = logging.getLogger(__name__)


def raw_trimmer(
    lh5_raw_file_in: str,
    overwrite: bool = False,
    trim_config: str | dict = None,
    trim_file_name: str = None,
) -> None:
    """
    This function performs data trimming on an existing raw-level :func:`lh5` file according to the :func:`trim_config` passed in the arguments.


    Parameters
    ----------
    lh5_raw_file_in
        The path of a file created from :meth:`pygama.raw.build_raw`
    overwrite
        sets whether to overwrite the output file(s) if it (they) already exist.
    trim_config
        DSP config used for data trimming. If ``None``, no data trimming is performed.
    trim_file_name
        The path to the trimmed output file created

            - if None, uses ``{lh5_raw_file_in}_trim.lh5`` as the output filename.

    Notes
    -----
    This function assumes that all raw file data are stored in at most two :meth:`h5py.Group`s, as in the following
    :func:`ch000/raw` or :func:`raw/geds`
    """

    # Initialize the input raw file
    raw_store = LH5Store()
    lh5_file = raw_store.gimme_file(lh5_raw_file_in, "r")
    if lh5_file is None:
        raise ValueError(f"input file not found: {lh5_raw_file_in}")
        return

    # List the groups in the raw file
    lh5_groups = lgdo.ls(lh5_raw_file_in)
    lh5_tables = []

    # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
    for tb in lh5_groups:
        # Make sure that the upper level key isn't a dataset
        if isinstance(lh5_file[tb], h5py.Dataset):
            lh5_tables.append(f"{tb}")
        elif "raw" not in tb and lgdo.ls(lh5_file, f"{tb}/raw"):
            lh5_tables.append(f"{tb}/raw")
        # Look one layer deeper for a :meth:`lgdo.Table` if necessary
        elif lgdo.ls(lh5_file, f"{tb}"):
            # Check to make sure that this isn't a table itself
            maybe_table, _ = raw_store.read_object(f"{tb}", lh5_file)
            if isinstance(maybe_table, lgdo.Table):
                lh5_tables.append(f"{tb}")
                del maybe_table
            # otherwise, go deeper
            else:
                for sub_table in lgdo.ls(lh5_file, f"{tb}"):
                    maybe_table, _ = raw_store.read_object(
                        f"{tb}/{sub_table}", lh5_file
                    )
                    if isinstance(maybe_table, lgdo.Table):
                        lh5_tables.append(f"{tb}/{sub_table}")
                    del maybe_table

    if len(lh5_tables) == 0:
        raise RuntimeError(f"could not find any valid LH5 table in {lh5_raw_file_in}")

    # Initialize the trimmed file
    # Get the trim_file_name if the user doesn't pass one
    if trim_file_name is None:
        trim_file_name = lh5_raw_file_in
        i_ext = trim_file_name.rfind(".")
        if i_ext != -1:
            trim_file_name = trim_file_name[:i_ext]
        trim_file_name += "_trim.lh5"
    # clear existing output files
    if overwrite:
        if os.path.isfile(trim_file_name):
            os.remove(trim_file_name)
    raw_store.gimme_file(trim_file_name, "a")

    # Write everything in the raw file to the new file, trim appropriately
    for tb in lh5_tables:
        lgdo_obj, _ = raw_store.read_object(f"{tb}", lh5_file)
        # Find the out_name.
        # If the top level group has an lgdo table in it, then the out_name is group
        if len(tb.split("/")) == 1:
            out_name = tb.split("/")[0]
            group_name = "/"  # There is no group name
        # We are only considering nested groups of the form key1/key2/, so key2 is the out name
        else:
            out_name = tb.split("/")[-1]  # the out_name is the last key
            group_name = tb.split("/")[0]  # the group name is the first key

        # If we have a trim_config, then trim the data and write that to the file!
        if trim_config is not None:
            # If it's a table, and there's a waveform, trim it
            if isinstance(lgdo_obj, lgdo.Table) or isinstance(lgdo_obj, lgdo.Struct):
                if "waveform" in lgdo_obj.keys():
                    # trim the data in-place on lgdo_obj
                    # the waveform names are updated in data_trimmer
                    log.info(f"trimming on a waveform in {group_name}/{out_name}")
                    data_trimmer(lgdo_obj, trim_config, group_name)
        # Write the (possibly trimmed) lgdo_obj to a file
        raw_store.write_object(
            lgdo_obj, out_name, lh5_file=trim_file_name, group=group_name
        )
