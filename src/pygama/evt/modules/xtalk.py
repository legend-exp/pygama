"""
Module for cross talk correction of energies.
"""

import awkward as ak
import numpy as np
from lgdo import lh5
from lgdo.lh5 import ls

from .. import utils


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

        # get the event indexs
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
        2D numpy array of the cross talk correction matrix, the indexs should correspond to rawids (with same mapping as energies)
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
