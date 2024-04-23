import awkward as ak
import numpy as np
import pytest

from pygama.evt.modules import xtalk


def test_xtalk_corrected_energy():

    energy = np.array([[1, 2, 3], [4, 5, 6], [2, 0, 1], [0, 1, 0]])
    matrix = np.array([[0, 0, 1], [1, 0, 2], [0, 2, 0]])
    energy_corrected_zero_threshold = xtalk.xtalk_corrected_energy(energy, matrix, 0)

    assert np.all(
        energy_corrected_zero_threshold
        == (energy + np.array([[3, 7, 4], [6, 16, 10], [1, 4, 0], [0, 0, 2]]))
    )


def test_xtalk_corrected_energy_awkward_slow():

    energies_test = ak.Array([[1000, 200], [100], [500, 3000, 100]])
    rawid_test = ak.Array([[1, 2], [3], [1, 2, 3]])

    # first check exceptions
    matrix = {
        1: {1: 1.00, 2: 0.01, 3: 0.02, 4: 0.00},
        2: {1: 0.01, 2: 1.00, 3: 0, 4: 0.01},
        3: {1: 0.02, 2: 0.00, 3: 1.00, 4: 0.01},
        4: {1: 0.00, 2: 0.01, 3: 0.01, 4: 1.00},
    }

    # if rawid and energies have different shapes (juts the first entry)
    with pytest.raises(ValueError):
        xtalk.xtalk_corrected_energy_awkward_slow(
            ak.Array([[1000, 200]]), rawid_test, matrix, True
        )

    # filter some values from energy first so each event has a different size
    with pytest.raises(ValueError):
        xtalk.xtalk_corrected_energy_awkward_slow(
            energies_test[energies_test < 600], rawid_test, matrix, True
        )

    # checks on the matrix
    # first check if the matrix has empty elements an exception is raised if allow_non_existing is false
    matrix_not_full = {
        1: {1: 1.00, 3: 0.02, 4: 0.00},
        2: {2: 1.00, 3: 0},
        3: {1: 0.02, 2: 0.00, 3: 1.00, 4: 0.01},
        4: {1: 0.00, 3: 0.01, 4: 1.00},
    }

    with pytest.raises(ValueError):
        xtalk.xtalk_corrected_energy_awkward_slow(
            energies_test, rawid_test, matrix_not_full, False
        )

    matrix_not_sym = {
        1: {1: 1.00, 2: 0.0, 3: 0.02, 4: 0.00},
        2: {1: 0.01, 2: 1.00, 3: 0, 4: 0.01},
        3: {1: 0.02, 2: 0.00, 3: 1.00, 4: 0.01},
        4: {1: 0.00, 2: 0.01, 3: 0.01, 4: 1.00},
    }
    with pytest.raises(ValueError):
        xtalk.xtalk_corrected_energy_awkward_slow(
            energies_test, rawid_test, matrix_not_sym, True
        )

    # now check the result returned (given no exceptions)
    # 1st event is two channels 1% cross talk and [1000,200] energy so we expect Ecorr  = [1002,210]
    # 2nd event one channel so the energy shouldnt change
    # 3rd event  3 channels [500,3000,100] energy ch2 has 1% ct to ch1 and 0% to ch3 so we will have
    # E= [530,3000,100]
    # next correct E2--> 3005, E3-->110]
    # finally correct from E3, 530-> 532 , 3005->3005
    energy_corr = xtalk.xtalk_corrected_energy_awkward_slow(
        energies_test, rawid_test, matrix, True
    )

    assert np.all(energy_corr[0].to_numpy() == np.array([1002, 210]))
    assert np.all(energy_corr[1].to_numpy() == np.array([100]))
    assert np.allclose(energy_corr[2].to_numpy(), np.array([532.0, 3005.0, 110.0]))

    # test with a threshold of 1 MeV

    energy_corr_threshold = xtalk.xtalk_corrected_energy_awkward_slow(
        energies_test, rawid_test, matrix, True, threshold=1000
    )

    assert np.allclose(energy_corr_threshold[2].to_numpy(), np.array([530, 3000, 100]))


def test_manipulate_xtalk_matrix():

    test_matrix = {"V02160A": {"V02160B": 0.1}, "V02160B": {"V02160A": 0.1}}
    test_positive = {"V02160A": {"V02160B": 0.3}, "V02160B": {"V02160A": 0.3}}
    test_small = {"V02160A": {"V02160B": 0.01}, "V02160B": {"V02160A": 0.01}}

    test_bad_channel = {"AAAAAAA": {"BBBBBBB": 0.1}, "BBBBBBB": {"AAAAAAA": 0.1}}
    matrix_rawids = xtalk.manipulate_xtalk_matrix(test_matrix, None, True)

    # check names are converted to rawid ok

    assert matrix_rawids["ch1104000"]["ch1104001"] == test_matrix["V02160A"]["V02160B"]

    matrix_comb = xtalk.manipulate_xtalk_matrix(test_matrix, test_positive, True)
    assert matrix_comb["ch1104000"]["ch1104001"] == -0.3

    matrix_comb_small = xtalk.manipulate_xtalk_matrix(test_matrix, test_small, True)
    assert matrix_comb_small["ch1104000"]["ch1104001"] == 0.1

    # check if the matrix contains some non-existing channel the right exception is raised
    with pytest.raises(ValueError):
        xtalk.manipulate_xtalk_matrix(test_bad_channel, None, True)


def test_numpy_to_dict():

    array = np.array([[1, 0], [0, 1]])
    rawids = np.array([1, 2])
    matrix_dict = xtalk.numpy_to_dict(array, rawids)
    assert matrix_dict == {"ch1": {"ch1": 1, "ch2": 0}, "ch2": {"ch1": 0, "ch2": 1}}

    with pytest.raises(ValueError):
        xtalk.numpy_to_dict(array, np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        xtalk.numpy_to_dict(array, np.array([[1, 2, 3], [4, 5, 6]]))
