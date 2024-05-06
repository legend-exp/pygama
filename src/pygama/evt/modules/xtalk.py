"""
Module for cross talk correction of energies.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from legendmeta.catalog import Props
from lgdo import lh5, types
from lgdo.lh5 import ls

from pygama.hit.build_hit import _remove_uneeded_operations, _reorder_table_operations

from .. import utils


def build_tcm_index_array(
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
    tcm_indexs_out = np.full((len(rawids), np.max(tcm.idx) + 1), np.nan)

    # parse observables string. default to hit tier
    for idx_chan, channel in enumerate(rawids):

        # get the event indexes
        table_id = utils.get_tcm_id_by_pattern(
            datainfo._asdict()["dsp"].table_fmt, f"ch{channel}"
        )
        tcm_indexs = np.where(tcm.id == table_id)[0]
        idx_events = ak.to_numpy(tcm.idx[tcm.id == table_id])
        tcm_indexs_out[idx_chan][idx_events] = tcm_indexs

    # transpose to return object where row is events and column rawid idx
    return tcm_indexs_out.T


def gather_energy(
    observable: str, tcm: utils.TCMData, datainfo: utils.DataInfo, rawids: np.ndarray
) -> np.ndarray:
    """
    Builds the array of energies for the cross talk correction
    Parameters
    ----------
    observable
        expression for the pulse parameter to be gathered, can be a combination of different fields.
    datainfo
        utils.DataInfo object
    tcm
        time-coincidence map object
    rawids
        list of channel rawids from the cross talk matrix.
    """

    # replace group. with group___
    for tier in datainfo._asdict():
        group = datainfo._asdict()[tier].group
        observable = observable.replace(f"{group}.", f"{group}___")

    observable = observable.replace(".", "__")

    c = compile(observable, "gcc -O3 -ffast-math build_hit.py", "eval")

    tier_params = []
    for name in c.co_names:
        if "___" in name:
            tier, column = name.split("___")
            group = datainfo._asdict()[tier].group
            file = datainfo._asdict()[tier].file
            if (name, file, group, column) not in tier_params:
                tier_params.append((name, file, group, column))

    # initialise the output object
    energy_out = np.full((np.max(tcm.idx) + 1, len(rawids)), np.nan)

    for idx_chan, channel in enumerate(rawids):
        tbl = types.Table()
        idx_events = ak.to_numpy(tcm.idx[tcm.id == channel])

        for name, file, group, column in tier_params:
            try:
                # read the energy data
                data = lh5.read(f"ch{channel}/{group}/{column}", file, idx=idx_events)
                tbl.add_column(name, data)
            except (lh5.exceptions.LH5DecodeError, KeyError):
                tbl.add_column(name, types.Array(np.full_like(idx_events, np.nan)))

        res = tbl.eval(observable)
        energy_out[idx_events, idx_chan] = res.nda

    return energy_out


def filter_hits(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    filter_expr: str,
    xtalk_corr_energy: np.ndarray,
    rawids: np.ndarray,
) -> np.ndarray:
    """
    Function to which hits in an event are above threshold.
    Parameters:
    -----------
    datainfo, tcm
        utils.DataInfo and utils.TCMData objects
    filter_expr
        string containing the logic used to define which events are above threshold.
        this string can also refer to the corrected energy as 'xtalk_corr_energy'
    xtalk_corr_energy
        2D numpy array of correct energy, the row corresponds to the event and the column the rawid
    rawids
        1D array of the rawids corresponding to each column
    Returns
        a numpy array of the mask of which

    """

    # find the fields in the string
    mask = np.full_like(xtalk_corr_energy, False, dtype=bool)

    # replace group. with group___
    for tier in datainfo._asdict():
        group = datainfo._asdict()[tier].group
        filter_expr = filter_expr.replace(f"{group}.", f"{group}___")

    c = compile(filter_expr, "gcc -O3 -ffast-math build_hit.py", "eval")

    tier_params = []
    for name in c.co_names:
        if "___" in name:
            tier, column = name.split("___")
            group = datainfo._asdict()[tier].group
            file = datainfo._asdict()[tier].file
            if (name, file, group, column) not in tier_params:
                tier_params.append((name, file, group, column))

    for idx_chan, channel in enumerate(rawids):
        tbl = types.Table()
        idx_events = ak.to_numpy(tcm.idx[tcm.id == channel])

        for name, file, group, column in tier_params:
            try:
                # read the energy data
                data = lh5.read(f"ch{channel}/{group}/{column}", file, idx=idx_events)

                tbl.add_column(name, data)
            except (lh5.exceptions.LH5DecodeError, KeyError):
                tbl.add_column(name, types.Array(np.full_like(idx_events, np.nan)))

        # add the corrected energy to the table
        tbl.add_column(
            "xtalk_corr_energy", types.Array(xtalk_corr_energy[idx_events, idx_chan])
        )
        res = tbl.eval(filter_expr)
        mask[idx_events, idx_chan] = res.nda

    return mask


def xtalk_correct_energy_impl(
    uncal_energy: np.ndarray,
    cal_energy: np.ndarray,
    xtalk_matrix: np.ndarray,
    xtalk_threshold: float = None,
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
    uncal_energy
        2D numpy array of the uncalibrated energies in each event, the row corresponds to an event and the column the rawid
    cal_energy
        2D numpy array of the calibrated energies in each event, the row corresponds to an event and the column the rawid
    xtalk_matrix
        2D numpy array of the cross talk correction matrix, the indices should correspond to rawids (with same mapping as energies)
    xtalk_threshold
        threshold below which a hit is not used in xtalk correction.

    """
    # check input shapes and sizes

    uncal_energy_no_nan = np.nan_to_num(uncal_energy, 0)
    cal_energy_no_nan = np.nan_to_num(cal_energy, 0)

    if xtalk_threshold is not None:
        uncal_energy_with_threshold = np.where(
            cal_energy_no_nan < xtalk_threshold, 0, uncal_energy_no_nan
        )
    else:
        uncal_energy_with_threshold = uncal_energy_no_nan
    energy_correction = -np.matmul(xtalk_matrix, uncal_energy_with_threshold.T).T
    return uncal_energy_no_nan + energy_correction


def get_xtalk_correction(
    tcm: utils.DataInfo,
    datainfo: utils.DataInfo,
    uncal_energy_expr: str,
    cal_energy_expr: str,
    xtalk_threshold: float = None,
    xtalk_matrix_filename: str = "",
    xtalk_rawid_obj: str = "xtc/rawid_index",
    xtalk_matrix_obj: str = "xtc/xtalk_matrix_negative",
    positive_xtalk_matrix_obj: str = "xtc/xtalk_matrix_positive",
):

    # read lh5 files to numpy
    xtalk_matrix_numpy = lh5.read_as(xtalk_matrix_obj, xtalk_matrix_filename, "np")
    xtalk_matrix_rawids = lh5.read_as(xtalk_rawid_obj, xtalk_matrix_filename, "np")

    positive_xtalk_matrix_numpy = lh5.read_as(
        positive_xtalk_matrix_obj, xtalk_matrix_filename, "np"
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

    uncal_energy_array = gather_energy(
        uncal_energy_expr, tcm, datainfo, xtalk_matrix_rawids
    )
    cal_energy_array = gather_energy(
        cal_energy_expr, tcm, datainfo, xtalk_matrix_rawids
    )

    energy_corr = xtalk_correct_energy_impl(
        uncal_energy_array, cal_energy_array, xtalk_matrix, xtalk_threshold
    )
    return energy_corr


def calibrate_energy(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    energy_corr: np.ndarray,
    xtalk_matrix_rawids: np.ndarray,
    par_files: str | list[str],
    uncal_energy_expr: str,
    recal_var: str = None,
):

    out_arr = np.full_like(energy_corr, np.nan)
    par_dicts = Props.read_from(par_files)
    pars = {
        chan: chan_dict["pars"]["operations"] for chan, chan_dict in par_dicts.items()
    }

    p = recal_var.split(".")
    tier = p[0] if len(p) > 1 else "hit"

    table_fmt = datainfo._asdict()[tier].table_fmt
    file = datainfo._asdict()[tier].file

    keys = ls(file)

    for i, chan in enumerate(xtalk_matrix_rawids):
        try:
            cfg = pars[f"ch{chan}"]
            cfg, chan_inputs = _remove_uneeded_operations(
                _reorder_table_operations(cfg), recal_var.split(".")[-1]
            )
            chan_inputs.remove(recal_var.split(".")[-1])

            # get the event indices
            table_id = utils.get_tcm_id_by_pattern(table_fmt, f"ch{chan}")
            idx_events = ak.to_numpy(tcm.idx[tcm.id == table_id])

            # read the energy data
            if f"ch{chan}" in keys:
                outtbl_obj = lh5.read(
                    f"ch{chan}/dsp/", file, idx=idx_events, field_mask=chan_inputs
                )
            outtbl_obj.add_column(
                recal_var.split(".")[-1],
                types.Array(energy_corr[idx_events, i]),
            )

            for outname, info in cfg.items():
                outcol = outtbl_obj.eval(
                    info["expression"], info.get("parameters", None)
                )
                outtbl_obj.add_column(outname, outcol)
            out_arr[idx_events, i] = outtbl_obj[recal_var.split("")[-1]].nda
        except KeyError:
            out_arr[idx_events, i] = np.nan

    return out_arr