# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# All this code belongs to the team that coded Scipy, found at this link:
#     https://github.com/scipy/scipy/blob/v1.6.0/scipy/ndimage/filters.py#L210-L260
# The only thing changed was the calculation of the convulution, which
# originally called a function from a C library.  In this code, the convolution is
# performed with NumPy's built in convolution function.
from __future__ import annotations

import numpy
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


def gaussian_filter1d(sigma: int, truncate: float = 4.0) -> numpy.ndarray:
    """1-D Gaussian filter.

    Note
    ----
    This processor is composed of a factory function that is called using the
    `init_args` argument. The input and output waveforms are passed using
    `args`.

    Parameters
    ----------
    sigma
        standard deviation for Gaussian kernel
    truncate
        truncate the filter at this many standard deviations.
    """

    def _gaussian_kernel1d(sigma, radius):
        """
        Computes a 1-D Gaussian convolution kernel.
        """
        sigma2 = sigma * sigma
        x = numpy.arange(-radius, radius + 1)
        phi_x = numpy.exp(-0.5 / sigma2 * x**2)
        phi_x = phi_x / phi_x.sum()
        return phi_x

    sd = float(sigma)

    # Make the radius of the filter equal to truncate standard deviations

    lw = int(truncate * sd + 0.5)

    # Since we are calling correlate, not convolve, revert the kernel

    weights = _gaussian_kernel1d(sigma, lw)[::-1]
    weights = numpy.asarray(weights, dtype=numpy.float64)

    # Find the length of the kernel so we can reflect the signal an appropriate amount

    extension_length = int(len(weights) / 2) + 1

    @guvectorize(
        [
            "void(float32[:], float32[:])",
            "void(float64[:], float64[:])",
            "void(int32[:], int32[:])",
            "void(int64[:], int64[:])",
        ],
        "(n),(m)",
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    def gaussian_filter1d_out(wf_in, wf_out):

        # Have to create an array to enable the reflect mode
        # Extend the signal on the left and right by at least half of the length of the kernel

        wf_in = numpy.asarray(wf_in)

        # Short warning message if kernel is larger than signal, in which case signal can't be convolved

        if len(wf_in) < extension_length:
            raise ValueError(
                "Kernel calculated was larger than signal, try again with smaller parameters"
            )

        # This mode extends as a reflection
        # ‘reflect’ (d c b a | a b c d | d c b a)
        # The input is extended by reflecting about the edge of the last pixel. This mode is also
        # sometimes referred to as half-sample symmetric.

        reflected_front = numpy.flip(wf_in[0:extension_length])
        reflected_end = numpy.flip(wf_in[-extension_length:])

        # Extend the signal

        extended_signal = wf_in
        extended_signal = numpy.concatenate((extended_signal, reflected_end), axis=None)
        extended_signal = numpy.concatenate(
            (reflected_front, extended_signal), axis=None
        )
        output = numpy.correlate(extended_signal, weights, mode="same")

        # Now extract the original signal length

        wf_out[:] = output[extension_length:-extension_length]

    return gaussian_filter1d_out
