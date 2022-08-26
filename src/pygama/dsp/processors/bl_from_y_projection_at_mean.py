import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:], float32, float32[:],float32[:])",
        "void(float64[:], float64[:], float64, float64[:],float64[:])",
    ],
    "(n),(n),(),(),()",
    nopython=True,
    cache=True,
)
def bl_from_y_projection_at_mean(proj_in, border_in, mean_in, std_out, mean_idx_out):

    """
    Provided a projection of a waveform onto the y axis, the baseline is reconstructed by assuming mean of 0 of the projection and the stddev from it.

    Parameters
    ----------
    proj_in : array-like
        The array which depicts the projection (binned!)
    border_in : array-like
        The bin borders of the projection
    mean_in : scalar
        mean value which should be used

    std_out : scalar
        Returns the standard deviation of the given mean of the projection (this should be the baseline standard deviation)

    mean_idx_out : scalar
        Returns the index of the closest value to the given mean in the projection

    """

    # 5) Initialize output parameters

    std_out[0] = np.nan
    mean_idx_out[0] = np.nan

    # 6) Check inputs

    if np.isnan(proj_in).any():
        return

    # 7) Algorithm

    # find mean index
    mean_idx = 0
    for i in range(0, len(proj_in), 1):
        if  abs(mean_in-border_in[i]) < abs(mean_in-border_in[mean_idx]):
            mean_idx = i
    mean_idx_out[0] = mean_idx
    # and the approx standarddev
    for i in range(mean_idx, len(proj_in) - 1, 1):
        if proj_in[i] <= 0.5 * proj_in[mean_idx] and proj_in[i] != 0:
            std_out[0] = abs(mean_in - border_in[i]) * 2 / 2.355
            break
    # look also into the other direction
    for i in range(1, mean_idx, 1):
        if proj_in[i] >= 0.5 * proj_in[mean_idx] and proj_in[i] != 0:
            if std_out[0] < abs(mean_in - border_in[i]) * 2 / 2.355:
                std_out[0] = abs(mean_in - border_in[i]) * 2 / 2.355
            break
