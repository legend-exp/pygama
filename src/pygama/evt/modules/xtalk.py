"""
Module for cross talk correction of energies.
"""

import awkward as ak
import numpy as np
from legendmeta import LegendMetadata


def numpy_to_dict(array: np.ndarray, rawids: np.ndarray) -> dict:
    """Converts a 2D numpy array with a 1D array of rawids (channel names) to a dictonary
    Parameters
    ----------
    array
        2D numpy array
    rawids
        numpy array of associated channels / rawids
    """

    if array.ndim != 2:
        raise ValueError("array must be 2D ")
    if rawids.ndim != 1:
        raise ValueError("rawids must be 1D")
    if len(rawids) != array.shape[0] or len(rawids) != array.shape[1]:
        raise ValueError(
            "The number of elements of rawids should match the shape of 'array'"
        )

    out_dict = {}
    for i, row in enumerate(array):
        out_dict[f"ch{rawids[i]}"] = {}
        for j, value in enumerate(row):
            out_dict[f"ch{rawids[i]}"][f"ch{rawids[j]}"] = value

    return out_dict


def manipulate_xtalk_matrix(
    xtalk_matrix: dict, positive_xtalk_matrix: dict = None, det_names: bool = False
):
    """
    Function to read in and manipulate the cross talk matrix.
    Parameters
    ----------
    xtalk_matrix_filename (str)
         Path to the xtalk matrix
    positive_xtalk_matrix_filename (str)
        Path to the positive polarity cross talk matrix
    det_names
        bool to convert det names to rawids

    """

    if det_names is True:
        xtalk_matrix = convert_matrix_det_names_to_rawid(xtalk_matrix)

    # read the positive matrix
    if positive_xtalk_matrix is not None:

        if det_names is True:
            positive_xtalk_matrix = convert_matrix_det_names_to_rawid(
                positive_xtalk_matrix
            )

        # merge +ive and -ive matrix
        matrix_merge = {}
        for key_row, row in xtalk_matrix.items():
            matrix_merge[key_row] = {}
            for key_col, data in row.items():

                positive_xtalk = positive_xtalk_matrix[key_row][key_col]
                negative_xtalk = data

                if positive_xtalk > negative_xtalk:
                    matrix_merge[key_row][key_col] = -positive_xtalk
                else:
                    matrix_merge[key_row][key_col] = negative_xtalk

        xtalk_matrix = matrix_merge

    return xtalk_matrix


def convert_matrix_det_names_to_rawid(matrix: dict) -> dict:
    """
    Converts a cross talk matrix with keys of detector names to one with keys rawids
    Parameters
    ----------
    matrix
        dictonary of the cross talk matrix
    """
    metadb = LegendMetadata()
    chmap = metadb.channelmap("20230323T000000Z")

    geds_mapping = {
        _name: f"ch{_dict['daq']['rawid']}"
        for _name, _dict in chmap.items()
        if chmap[_name]["system"] == "geds"
    }
    matrix_conv = {}
    for key_row, row in matrix.items():

        if key_row not in geds_mapping.keys():
            raise ValueError(f"channel {key_row} doesnt have a valid rawid")

        matrix_conv[geds_mapping[key_row]] = {}
        for key_col, data in row.items():

            if key_col not in geds_mapping.keys():
                raise ValueError(f"channel {key_col} doesnt have a valid rawid")

            matrix_conv[geds_mapping[key_row]][geds_mapping[key_col]] = data

    return matrix_conv


def xtalk_corrected_energy(
    energies: np.ndarray,
    matrix: np.ndarray,
    threshold: float = None,
):
    """
    Function to perform the cross talk correction on awkward arrays of energy and rawid.
    1. The energies are converted to a sparse format where each row corresponds to a rawid
    2. All energy less than the threshold are set to 0
    3. The correction is computed as:
    .. math::
        E_{\text{cor},i}=E_{j}\times M_{ij}

    where $M_{ij}$ is the cross talk matrix.
    Parameters
    ----------
    energies
        2D numpy array of the energies in each event, the row corresponds to an event and the column the rawid
    matrix
        2D numpy array of the cross talk correction matrix, the indexs should correspond to rawids (with same mapping as energies)
    threshold
        energy threshold below which a hit is not used in xtalk correction.

    """
    # check input shapes and sizes

    if energies.ndim != 2:
        raise ValueError("energies must be a 2D array")

    if matrix.ndim != 2:
        raise ValueError("xtalk matrix must be a 2D array")

    if matrix.shape[0] != energies.shape[1]:
        raise ValueError(
            "the energy vector must have the smae size as the cross talk matrix"
        )

    energies_threshold = energies * (energies > threshold)

    energies_correction = np.matmul(matrix, energies_threshold.T).T
    return energies + energies_correction


def xtalk_corrected_energy_awkward_slow(
    energies: ak.Array,
    rawids: ak.Array,
    matrix: dict,
    allow_non_existing: bool = True,
    threshold: float = None,
):
    """
    Function to perform the cross talk correction on awkward arrays of energy and rawid.
    The energies are first sorted from largest to smallest, a term is then added to the
        other energies of the cross talk matrix element multipled by the largest energy.
    .. math::
        E_{i,\text{cor}} = E_{1}\times M[c_i,c_1]+E_{i},
        where $c_i$ is the raw-id of the $i$th energy.

    This process is repeated recursively for the 2nd largest energy etc.
    This implementation is called 'slow' since it uses loops over events and energies

    Parameters
        - energies (ak.Array): array of energies
        - rawids   (ak.Array): array of rawids
        - matrix (dict)      : python dictonary of the cross talk correction matrix
        - allow_non_existing (bool): A boolean to control what happens if a rawid is not present
            in the matrix, if True, this matrix element is set to 0, if False an exception is raised.
        - threshold (float)  : energy threshold below which hits are not used to correct the other hits
    Returns:
        ak.Array of corrected energies

    """

    # some exceptions
    # check types
    if not isinstance(energies, ak.Array):
        raise TypeError("energies must be an awkward array")

    if not isinstance(rawids, ak.Array):
        raise TypeError("rawids must be an awkward array")

    if not isinstance(matrix, dict):
        raise TypeError("matrix must be a python dictonary")

    if not isinstance(allow_non_existing, bool):
        raise TypeError("allow_non_existing must be a Boolean")

    # first check that energies and rawids have the same dimensions
    if ak.any(ak.num(energies, axis=-1) != ak.num(rawids, axis=-1)):
        raise ValueError(
            "Error: the length of each subarray of energies and rawids must be equal"
        )

    if ak.num(energies, axis=-2) != ak.num(rawids, axis=-2):
        raise ValueError("Error: the number of energies is not equal to rawids")

    # check that the matrix elements exist
    for c1 in np.unique(ak.flatten(rawids).to_numpy()):
        if c1 not in matrix.keys():

            if allow_non_existing is True:
                matrix[c1] = {}
            else:
                raise ValueError(
                    f"Error allow_non_existing is set to False and {c1} isnt present in the matrix"
                )

        for c2 in np.unique(ak.flatten(rawids).to_numpy()):
            if c1 == c2:
                continue
            else:
                if c2 not in matrix[c1].keys():
                    if allow_non_existing is True:
                        matrix[c1][c2] = 0
                    else:
                        raise ValueError(
                            f"Error allow_non_existing is set to False and {c2} isnt present in the matrix[{c1}]"
                        )

    # add a check that the matrix is symmetric

    for c1 in matrix.keys():
        for c2 in matrix[c1].keys():
            if abs(matrix[c1][c2] - matrix[c2][c1]) > 1e-6:
                raise ValueError(
                    f"Error input cross talk matrix is not symmetric for {c1},{c2}"
                )

    # sort the energies and rawids
    args = ak.argsort(energies, ascending=False)
    energies = energies[args]
    rawids = rawids[args]

    # run the correction
    # --------------------

    energies_corrected = []

    # we should try to speed this up
    for energy_vec_tmp, rawid_vec_tmp in zip(energies, rawids):

        energies_corrected_tmp = list(energy_vec_tmp)
        for id_main, (energy_main, rawid_main) in enumerate(
            zip(energy_vec_tmp, rawid_vec_tmp)
        ):
            if threshold is not None and energy_main < threshold:
                break
            for id_other, (_energy_other, rawid_other) in enumerate(
                zip(energy_vec_tmp, rawid_vec_tmp)
            ):
                if id_main != id_other:
                    energies_corrected_tmp[id_other] += (
                        matrix[rawid_main][rawid_other] * energy_main
                    )

        energies_corrected.append(energies_corrected_tmp)

    # convert to awkward array and unsort
    return ak.Array(energies_corrected)[args]
