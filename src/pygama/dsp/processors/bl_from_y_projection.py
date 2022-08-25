import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:],float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:],float64[:])",
    ],
    "(n),(n),(),(),()",
    nopython=True,
    cache=True,
)
def bl_from_y_projection(proj_in, border_in, mean_out, std_out, idx_out):

    """
    Provided a projection of a waveform onto the y axis, the baseline is reconstructed by taking the maximum of the projection and the stddev from it.

    Parameters
    ----------
    proj_in : array-like
        The array which depicts the projection (binned!)
    border_in : array-like
        The bin borders of the projection
    mean_out : scalar
        Returns the maximum of the projection (this should be the baseline)

    std_out : scalar
        Returns the standard deviation of the maximum of the projection (this should be the baseline standard deviation)

    idx_out : scalar
        Returns the index of the max. position found in the projection (proj_in[idx_out] = mean_out)
    """

    # 5) Initialize output parameters

    mean_out[0] = np.nan
    std_out[0] = np.nan
    idx_out[0] = np.nan

    # 6) Check inputs

    if np.isnan(proj_in).any():
        return

    # 7) Algorithm

    # get global maximum = our baseline mean
    max_index = 0
    for i in range(0, len(proj_in), 1):
        if proj_in[i] > proj_in[max_index]:
            max_index = i
    idx_out[0] = max_index
    mean_out[0] = border_in[max_index]

    # and the approx standarddev
    for i in range(max_index, len(proj_in) - 1, 1):
        if proj_in[i] <= 0.5 * proj_in[max_index] and proj_in[i] != 0:
            std_out[0] = abs(mean_out[0] - border_in[i]) * 2 / 2.355
            break
    # look also into the other direction
    for i in range(1, max_index, 1):
        if proj_in[i] >= 0.5 * proj_in[max_index] and proj_in[i] != 0:
            if std_out[0] < abs(mean_out[0] - border_in[i]) * 2 / 2.355:
                std_out[0] = abs(mean_out[0] - border_in[i]) * 2 / 2.355
            break
