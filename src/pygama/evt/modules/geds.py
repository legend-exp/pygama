"""Event processors for HPGe data."""

from __future__ import annotations

from collections.abc import Sequence

import awkward as ak
import numpy as np
from legendmeta.catalog import Props
from lgdo import lh5, types
from lgdo.lh5 import ls

from pygama.hit.build_hit import _reorder_table_operations

from .. import utils
from . import xtalk

sto = lh5.LH5Store()


def apply_recovery_cut(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    *,
    timestamps: types.Array,
    flag: types.Array,
    time_window: float,
) -> types.Array:

    discharge_timestamps = timestamps.nda[flag.nda == 1]
    is_recovering = np.full(len(timestamps.nda), False)
    for tstamp in discharge_timestamps:
        is_recovering = is_recovering | np.where(
            (
                ((timestamps.nda - tstamp) < time_window)
                & ((timestamps.nda - tstamp) >= 0)
            ),
            True,
            False,
        )

    # return the result as LGDO
    return types.Array(is_recovering)


def apply_xtalk_correction(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    *,
    uncalibrated_energy_name: str,
    calibrated_energy_name: str,
    threshold: float = None,
    xtalk_matrix_filename: str,
    xtalk_rawid_name: str = "xtc/rawid_index",
    xtalk_matrix_name: str = "xtc/xtalk_matrix_negative",
    positive_xtalk_matrix_name: str = "xtc/xtalk_matrix_positive",
) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.
    The format of `xtalk_matrix_filename` should be currently be a path to a lh5 file.

    The correction is appplied using matrix algebra for all triggers above the threshold.

    Parameters
    ----------
    datainfo, tcm, table_names
        positional arguments automatically supplied by :func:`.build_evt`.
    uncalibrated_energy_name
        name of the pulse parameter for uncalibrated energy to be gathered, optionally prefixed by tier
        name (e.g. ``hit.cusp_Emax``). If no tier is specified, it defaults
        to ``hit``.
    calibrated_energy_name
        name of the pulse parameter for calibrated energy to be gathered, optionally prefixed by tier
        name (e.g. ``hit.cusp_Emax``). If no tier is specified, it defaults
        to ``hit``.
    threshold
        threshold used for xtalk correction, hits below this energy will not
        be used to correct the other hits.
    xtalk_matrix_filename
        name of the file containing the xtalk matrices.
    xtalk_matrix_name
        name of the lh5 object containing the xtalk matrix
    positive_xtalk_matrix_name
        name of the lh5 object containing the positive polarity xtalk matrix
    xtalk_matrix_rawids
        name of the lh5 object containing the name of the rawids
    """

    # read lh5 files to numpy
    xtalk_matrix_numpy = lh5.read_as(xtalk_matrix_name, xtalk_matrix_filename, "np")
    print(f"Read {xtalk_rawid_name}, {xtalk_matrix_filename}")
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

    uncalibrated_energy_array = xtalk.build_energy_array(
        uncalibrated_energy_name, tcm, datainfo, xtalk_matrix_rawids
    )
    calibrated_energy_array = xtalk.build_energy_array(
        calibrated_energy_name, tcm, datainfo, xtalk_matrix_rawids
    )

    energies_corr = xtalk.xtalk_corrected_energy(
        uncalibrated_energy_array, calibrated_energy_array, xtalk_matrix, threshold
    )

    # return the result as LGDO
    return types.VectorOfVectors(energies_corr)


def apply_xtalk_correction_and_calibrate(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    *,
    uncalibrated_energy_name: str,
    calibrated_energy_name: str,
    par_files: str | list[str],
    threshold: float = None,
    xtalk_matrix_filename: str,
    xtalk_rawid_name: str = "xtc/rawid_index",
    xtalk_matrix_name: str = "xtc/xtalk_matrix_negative",
    positive_xtalk_matrix_name: str = "xtc/xtalk_matrix_positive",
    out_param: str = None,
) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.
    The format of `xtalk_matrix_filename` should be currently be a path to a lh5 file.

    The correction is appplied using matrix algebra for all triggers above the threshold.

    Parameters
    ----------
    datainfo, tcm, table_names
        positional arguments automatically supplied by :func:`.build_evt`.
    uncalibrated_energy_name
        name of the pulse parameter for uncalibrated energy to be gathered, optionally prefixed by tier
        name (e.g. ``hit.cusp_Emax``). If no tier is specified, it defaults
        to ``hit``.
    calibrated_energy_name
        name of the pulse parameter for calibrated energy to be gathered, optionally prefixed by tier
        name (e.g. ``hit.cusp_Emax``). If no tier is specified, it defaults
        to ``hit``.
    threshold
        threshold used for xtalk correction, hits below this energy will not
        be used to correct the other hits.
    xtalk_matrix_filename
        name of the file containing the xtalk matrices.
    xtalk_matrix_name
        name of the lh5 object containing the xtalk matrix
    positive_xtalk_matrix_name
        name of the lh5 object containing the positive polarity xtalk matrix
    xtalk_matrix_rawids
        name of the lh5 object containing the name of the rawids
    """

    # read lh5 files to numpy
    xtalk_matrix_numpy = lh5.read_as(xtalk_matrix_name, xtalk_matrix_filename, "np")
    print(f"Read {xtalk_rawid_name}, {xtalk_matrix_filename}")
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

    uncalibrated_energy_array = xtalk.build_energy_array(
        uncalibrated_energy_name, tcm, datainfo, xtalk_matrix_rawids
    )
    calibrated_energy_array = xtalk.build_energy_array(
        calibrated_energy_name, tcm, datainfo, xtalk_matrix_rawids
    )
    dt_array = xtalk.build_energy_array(
        "dsp.dt_eff", tcm, datainfo, xtalk_matrix_rawids
    )

    energies_corr = xtalk.xtalk_corrected_energy(
        uncalibrated_energy_array, calibrated_energy_array, xtalk_matrix, threshold
    )
    out_arr = np.full_like(energies_corr, np.nan)
    par_dicts = Props.read_from(par_files)
    pars = {
        chan: chan_dict["pars"]["operations"] for chan, chan_dict in par_dicts.items()
    }

    p = uncalibrated_energy_name.split(".")
    tier = p[0] if len(p) > 1 else "hit"
    column = p[1] if len(p) > 1 else p[0]

    table_fmt = datainfo._asdict()[tier].table_fmt
    group = datainfo._asdict()[tier].group
    file = datainfo._asdict()[tier].file

    keys = ls(file)

    if out_param is None:
        out_param = calibrated_energy_name.split(".")[-1]

    for i, chan in enumerate(xtalk_matrix_rawids):
        try:
            cfg = pars[f"ch{chan}"]
            cfg, chan_inputs = xtalk.remove_uneeded_operations(
                _reorder_table_operations(cfg), out_param
            )
            chan_inputs.remove(uncalibrated_energy_name.split(".")[-1])

            # get the event indexs
            table_id = utils.get_tcm_id_by_pattern(table_fmt, f"ch{chan}")
            idx_events = ak.to_numpy(tcm.idx[tcm.id == table_id])

            # read the energy data
            if f"ch{chan}" in keys:
                outtbl_obj = sto.read(
                    f"ch{chan}/dsp/", file, idx=idx_events, field_mask=chan_inputs
                )[0]
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

    # return the result as LGDO
    return types.VectorOfVectors(out_arr)
