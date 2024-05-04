"""
Module for cross talk correction of energies.
"""

from __future__ import annotations

import importlib

import awkward as ak
import numpy as np
from legendmeta.catalog import Props
from lgdo import lh5, types
from lgdo.lh5 import ls

from pygama.hit.build_hit import _remove_uneeded_operations, _reorder_table_operations

from .. import utils


def build_tcm_id_array(
    tcm: utils.TCMData, datainfo: utils.DataInfo, rawids: np.ndarray
) -> np.ndarray:
    """
    datainfo
        utils.DataInfo object
    tcm
        time-coincidence map object
    rawids
        list of channel rawids from the cross talk matrix.
    """

    # initialise the output object
    tcm_ids_out = np.full((len(rawids), np.max(tcm.idx) + 1), np.nan)

    # parse observables string. default to hit tier
    for idx_chan, channel in enumerate(rawids):

        # get the event indexes
        table_id = utils.get_tcm_id_by_pattern(
            datainfo._asdict()["dsp"].table_fmt, f"ch{channel}"
        )
        tcm_ids = np.where(tcm.id == table_id)[0]
        idx_events = ak.to_numpy(tcm.idx[tcm.id == table_id])
        tcm_ids_out[idx_chan][idx_events] = tcm_ids

    # transpose to return object where row is events and column rawid idx
    return tcm_ids_out.T


def build_energy_array(
    observable: str, tcm: utils.TCMData, datainfo: utils.DataInfo, rawids: np.ndarray
) -> np.ndarray:
    """
    Builds the array of energies for the cross talk correction
    Parameters
    ----------
    observable
        name of the pulse parameter to be gathered, optionally prefixed by tier
        name (e.g. ``hit.cuspEmax_ctc_cal``). If no tier is specified, it defaults
        to ``hit``.
    datainfo
        utils.DataInfo object
    tcm
        time-coincidence map object
    rawids
        list of channel rawids from the cross talk matrix.
    """

    # build the energy arrays
    p = observable.split(".")
    tier = p[0] if len(p) > 1 else "hit"
    column = p[1] if len(p) > 1 else p[0]

    table_fmt = datainfo._asdict()[tier].table_fmt
    group = datainfo._asdict()[tier].group
    file = datainfo._asdict()[tier].file

    # initialise the output object
    energies_out = np.full((len(rawids), np.max(tcm.idx) + 1), np.nan)

    # parse observables string. default to hit tier
    keys = ls(file)

    for idx_chan, channel in enumerate(rawids):

        # get the event indexes
        table_id = utils.get_tcm_id_by_pattern(table_fmt, f"ch{channel}")
        idx_events = ak.to_numpy(tcm.idx[tcm.id == table_id])

        # read the energy data
        if f"ch{channel}" in keys:
            data = lh5.read_as(
                f"ch{channel}/{group}/{column}", file, idx=idx_events, library="np"
            )
            energies_out[idx_chan][idx_events] = data

    # transpose to return object where row is events and column rawid idx
    return energies_out.T


def filter_hits(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    logic: str,
    corrected_energy: np.ndarray,
    rawids: np.ndarray,
) -> np.ndarray:
    """
    Function to which hits in an event are above threshold.
    Parameters:
    -----------
    datainfo, tcm
        utils.DataInfo and utils.TCMData objects
    logic
        string containing the logic used to define which events are above threshold.
        this string can also refer to the corrected energy as 'corrected_energy'
    corrected_energy
        2D numpy array of correct energy, the row corresponds to the event and the column the rawid
    rawids
        1D array of the rawids corresponding to each column
    Returns
        a numpy array of the mask of which

    """

    # find the fields in the string
    mask = np.full_like(corrected_energy, False, dtype=bool)

    # replace group. with group___
    for tier in datainfo._asdict():
        group = datainfo._asdict()[tier].group
        logic = logic.replace(f"{group}.", f"{group}___")

    # replace remaining . with __ as these are module calls
    logic = logic.replace(".", "__")

    c = compile(logic, "gcc -O3 -ffast-math build_hit.py", "eval")

    tier_params = []
    for name in c.co_names:
        if "___" in name:
            tier, column = name.split("___")
            group = datainfo._asdict()[tier].group
            file = datainfo._asdict()[tier].file
            if (file, group, column) not in tier_params:
                tier_params.append((file, group, column))
        elif "__" in name:
            # get module and function names
            package, func = name.rsplit("__", 1)
            # import function into current namespace
            importlib.import_module(package)

    for idx_chan, channel in enumerate(rawids):
        tbl = types.Table()
        idx_events = ak.to_numpy(tcm.idx[tcm.id == channel])

        for file, group, column in tier_params:
            keys = ls(file)
            try:
                # read the energy data
                if f"ch{channel}" in keys:
                    data = lh5.read(
                        f"ch{channel}/{group}/{column}", file, idx=idx_events
                    )
                tbl.add_column(name, data)
            except KeyError:
                tbl.add_column(name, np.full_like(idx_events, np.nan))

        # add the corrected energy to the table
        tbl.add_column(
            "corrected_energy", types.Array(corrected_energy[idx_events, idx_chan])
        )
        res = tbl.eval(logic)
        mask[idx_events, idx_chan] = res.nda

    return mask


def xtalk_corrected_energy(
    uncalibrated_energies: np.ndarray,
    calibrated_energies: np.ndarray,
    matrix: np.ndarray,
    threshold: float = None,
):
    """
    Function to perform the cross talk correction on awkward arrays of energy and rawid.
    1. The energies are converted to a sparse format where each row corresponds to a rawid
    2. All energy less than the threshold are set to 0
    3. The correction is computed as:
    .. math::
        E_{\text{cor},i}=-\times M_{i,j}E_{j}

    where $M_{i,j}$ is the cross talk matrix element where i is response and j trigger channel.
    Parameters
    ----------
    uncalibrated_energies
        2D numpy array of the uncalibrated energies in each event, the row corresponds to an event and the column the rawid
    calibrated_energies
        2D numpy array of the calibrated energies in each event, the row corresponds to an event and the column the rawid
    matrix
        2D numpy array of the cross talk correction matrix, the indices should correspond to rawids (with same mapping as energies)
    threshold
        energy threshold below which a hit is not used in xtalk correction.

    """
    # check input shapes and sizes

    uncalibrated_energies_no_nan = np.nan_to_num(uncalibrated_energies, 0)
    calibrated_energies_no_nan = np.nan_to_num(calibrated_energies, 0)

    if threshold is not None:
        energies_threshold = np.where(
            calibrated_energies_no_nan < threshold, 0, uncalibrated_energies_no_nan
        )
    else:
        energies_threshold = uncalibrated_energies_no_nan
    energies_correction = -np.matmul(matrix, energies_threshold.T).T
    return uncalibrated_energies + energies_correction


def get_xtalk_correction(
    tcm: utils.DataInfo,
    datainfo: utils.DataInfo,
    uncalibrated_energy_name: str,
    calibrated_energy_name: str,
    threshold: float = None,
    xtalk_matrix_filename: str = "",
    xtalk_rawid_name: str = "xtc/rawid_index",
    xtalk_matrix_name: str = "xtc/xtalk_matrix_negative",
    positive_xtalk_matrix_name: str = "xtc/xtalk_matrix_positive",
):

    # read lh5 files to numpy
    xtalk_matrix_numpy = lh5.read_as(xtalk_matrix_name, xtalk_matrix_filename, "np")
    xtalk_matrix_rawids = lh5.read_as(xtalk_rawid_name, xtalk_matrix_filename, "np")

    positive_xtalk_matrix_numpy = lh5.read_as(
        positive_xtalk_matrix_name, xtalk_matrix_filename, "np"
    )

    # Combine positive and negative matrixs
    # Now the matrix should have negative values corresponding to negative cross talk
    # and positive values corresponding to positive cross talk .
    # we also set nan to 0 and we transpose so that the row corresponds to response and column trigger
    xtalk_matrix = np.nan_to_num(
        np.where(
            abs(xtalk_matrix_numpy) > abs(positive_xtalk_matrix_numpy),
            xtalk_matrix_numpy,
            positive_xtalk_matrix_numpy,
        ),
        0,
    ).T

    uncalibrated_energy_array = build_energy_array(
        uncalibrated_energy_name, tcm, datainfo, xtalk_matrix_rawids
    )
    calibrated_energy_array = build_energy_array(
        calibrated_energy_name, tcm, datainfo, xtalk_matrix_rawids
    )

    energies_corr = xtalk_corrected_energy(
        uncalibrated_energy_array, calibrated_energy_array, xtalk_matrix, threshold
    )
    return energies_corr


def calibrate_energy(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    energies_corr: np.ndarray,
    xtalk_matrix_rawids: np.ndarray,
    par_files: str | list[str],
    uncalibrated_energy_name: str,
    out_param: str = None,
):

    out_arr = np.full_like(energies_corr, np.nan)
    par_dicts = Props.read_from(par_files)
    pars = {
        chan: chan_dict["pars"]["operations"] for chan, chan_dict in par_dicts.items()
    }

    p = uncalibrated_energy_name.split(".")
    tier = p[0] if len(p) > 1 else "hit"

    table_fmt = datainfo._asdict()[tier].table_fmt
    file = datainfo._asdict()[tier].file

    keys = ls(file)

    for i, chan in enumerate(xtalk_matrix_rawids):
        try:
            cfg = pars[f"ch{chan}"]
            cfg, chan_inputs = _remove_uneeded_operations(
                _reorder_table_operations(cfg), out_param
            )
            chan_inputs.remove(uncalibrated_energy_name.split(".")[-1])

            # get the event indices
            table_id = utils.get_tcm_id_by_pattern(table_fmt, f"ch{chan}")
            idx_events = ak.to_numpy(tcm.idx[tcm.id == table_id])

            # read the energy data
            if f"ch{chan}" in keys:
                outtbl_obj = lh5.read(
                    f"ch{chan}/dsp/", file, idx=idx_events, field_mask=chan_inputs
                )
            outtbl_obj.add_column(
                uncalibrated_energy_name.split(".")[-1],
                types.Array(energies_corr[:, i]),
            )

            for outname, info in cfg.items():
                outcol = outtbl_obj.eval(
                    info["expression"], info.get("parameters", None)
                )
                outtbl_obj.add_column(outname, outcol)
            out_arr[:, i] = outtbl_obj[out_param].nda
        except KeyError:
            out_arr[:, i] = np.nan

    return out_arr
