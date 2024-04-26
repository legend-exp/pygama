import numpy as np

from pygama.evt.modules import xtalk


def test_xtalk_corrected_energy():

    energy = np.array([[1, 2, 3], [4, 5, 6], [2, 0, 1], [0, 1, 0]])
    matrix = np.array([[0, 0, 1], [1, 0, 2], [0, 2, 0]])
    energy_corrected_zero_threshold = xtalk.xtalk_corrected_energy(energy, matrix, 0)

    assert np.all(
        energy_corrected_zero_threshold
        == (energy - np.array([[3, 7, 4], [6, 16, 10], [1, 4, 0], [0, 0, 2]]))
    )
