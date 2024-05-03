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


def filter_hits(datainfo:utils.DataInfo,tcm:utils.TCMData,logic:str,corrected_energy:np.ndarray,rawids:np.ndarray)->np.ndarray:
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
    mask = np.full((len(rawids), np.max(tcm.idx) + 1), False)

    # replace . with ____
    logic = logic.replace(".", "___")

    c = compile(logic, "gcc -O3 -ffast-math build_hit.py", "eval")
    for idx_chan,channel in enumerate(rawids):
        tbl = lgdo.Table()
    
        for name in c.co_names:
            if ("___" not in name):
                continue
            tier, column = name.split("___")

            try:
                table_fmt = datainfo._asdict()[tier].table_fmt
                group = datainfo._asdict()[tier].group
                file = datainfo._asdict()[tier].file
                keys=ls(file)
                table_id = utils.get_tcm_id_by_pattern(table_fmt, f"ch{channel}")
                idx_events = ak.to_numpy(tcm.idx[tcm.id == table_id])

                # read the energy data
                if f"ch{channel}" in keys:
                    data = lh5.read(
                        f"ch{channel}/{group}/{column}", file, idx=idx_events
                    )
                tbl.add_column(name,data)
            except KeyError:
                pass
        
        # add the corrected energy to the table
        tbl.add_column("corrected_energy",lgdo.Array(corrected_energy[:][idx_chan]))
        res = tbl.eval(logic)
        mask[idx_chan][idx_events]=res
    
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


def get_dependencies(config, par, pars=[]):
    par_op = config[par]
    c = compile(par_op["expression"], "gcc -O3 -ffast-math build_hit.py", "eval")
    for p in c.co_names:
        if p in par_op["parameters"]:
            pass
        else:
            pars.append(p)
            if p in config:
                pars = get_dependencies(config, p, pars)
    return pars


def remove_uneeded_operations(config, outpars):
    if not isinstance(outpars, list):
        outpars = [outpars]
    dependent_keys = [*outpars]
    inkeys = []
    for par in outpars:
        pars = get_dependencies(config, par)
        for p in pars:
            if p in config and p not in dependent_keys:
                dependent_keys.append(p)
            elif p not in config and p not in inkeys:
                inkeys.append(p)

    for key in list(config):
        if key not in dependent_keys:
            config.pop(key)
    return config, inkeys
