from __future__ import annotations

from typing import Callable

import numpy as np
from numba import guvectorize
from pyfftw import FFTW

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


def dft(buf_in: np.ndarray, buf_out: np.ndarray) -> Callable:
    """Perform discrete Fourier transforms using the FFTW library.

    Note
    ----
    This processor is composed of a factory function that is called using the
    `init_args` argument. The input and output waveforms are passed using
    `args`.

    Note
    ----
    FFTW optimizes the FFT algorithm based on the size of the arrays, with SIMD
    parallelized commands.  This optimization requires initialization, so this
    is a factory function that returns a Numba gufunc that performs the FFT.
    FFTW works on fixed memory buffers, so you must tell it what memory to use
    ahead of time.  When using this with
    :class:`~.dsp.processing_chain.ProcessingChain`, to ensure the correct
    buffers are used, call
    :meth:`~.dsp.processing_chain.ProcessingChain.get_variable` to give it the
    internal memory buffer directly. With :func:`~.dsp.build_dsp.build_dsp`,
    you can just give it the name, and it will automatically happen. The
    possible `dtypes` for the input/outputs are:

    =============================== ========= =============================== =============
    :class:`numpy.dtype`            Size      :class:`numpy.dtype`            Size
    ------------------------------- --------- ------------------------------- -------------
    ``float32``/``float``           :math:`n` ``complex64``                   :math:`n/2+1`
    ``float64``/``double``          :math:`n` ``complex128``                  :math:`n/2+1`
    ``float128``/``longdouble``     :math:`n` ``complex256``/``clongdouble``  :math:`n/2+1`
    ``complex64``                   :math:`n` ``complex64``                   :math:`n`
    ``complex128``                  :math:`n` ``complex128``                  :math:`n`
    ``complex256``/``clongdouble``  :math:`n` ``complex256``/``clongdouble``  :math:`n`
    =============================== ========= =============================== =============
    """
    try:
        dft_fun = FFTW(buf_in, buf_out, axes=(-1,), direction="FFTW_FORWARD")
    except ValueError:
        raise ValueError(
            "incompatible array types/shapes. See function documentation for allowed values"
        )

    typesig = "void(" + str(buf_in.dtype) + "[:, :], " + str(buf_out.dtype) + "[:, :])"
    sizesig = "(m, n)->(m, n)" if buf_in.shape == buf_out.shape else "(m, n),(m, l)"

    @guvectorize(
        [typesig],
        sizesig,
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    def dft(wf_in: np.ndarray, dft_out: np.ndarray) -> None:
        dft_fun(wf_in, dft_out)

    return dft


def inv_dft(buf_in: np.ndarray, buf_out: np.ndarray) -> Callable:
    """Perform inverse discrete Fourier transforms using the FFTW library.

    Note
    ----
    This processor is composed of a factory function that is called using the
    `init_args` argument. The input and output waveforms are passed using
    `args`.

    Note
    ----
    FFTW optimizes the FFT algorithm based on the size of the arrays, with SIMD
    parallelized commands.  This optimization requires initialization, so this
    is a factory function that returns a Numba gufunc that performs the FFT.
    FFTW works on fixed memory buffers, so you must tell it what memory to use
    ahead of time.  When using this with
    :class:`~.dsp.processing_chain.ProcessingChain`, to ensure the correct
    buffers are used, call
    :meth:`~.dsp.processing_chain.ProcessingChain.get_variable` to give it the
    internal memory buffer directly. With :func:`~.dsp.build_dsp.build_dsp`,
    you can just give it the name, and it will automatically happen. The
    possible `dtypes` for the input/outputs are:

    =============================== ============= =============================== =========
    :class:`numpy.dtype`            Size          :class:`numpy.dtype`            Size
    ------------------------------- ------------- ------------------------------- ---------
    ``complex64``                   :math:`n/2+1` ``float32``/``float``           :math:`n`
    ``complex128``                  :math:`n/2+1` ``float64``/``double``          :math:`n`
    ``complex256``/``clongdouble``  :math:`n/2+1` ``float128``/``longdouble``     :math:`n`
    ``complex64``                   :math:`n`     ``complex64``                   :math:`n`
    ``complex128``                  :math:`n`     ``complex128``                  :math:`n`
    ``complex256``/``clongdouble``  :math:`n`     ``complex256``/``clongdouble``  :math:`n`
    =============================== ============= =============================== =========
    """
    try:
        idft_fun = FFTW(buf_in, buf_out, axes=(-1,), direction="FFTW_BACKWARD")
    except ValueError:
        raise ValueError(
            "incompatible array types/shapes. See function documentation for allowed values"
        )

    typesig = "void(" + str(buf_in.dtype) + "[:, :], " + str(buf_out.dtype) + "[:, :])"
    sizesig = "(m, n)->(m, n)" if buf_in.shape == buf_out.shape else "(m, n),(m, l)"

    @guvectorize(
        [typesig],
        sizesig,
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    def inv_dft(wf_in: np.ndarray, dft_out: np.ndarray) -> None:
        idft_fun(wf_in, dft_out)

    return inv_dft


def psd(buf_in: np.ndarray, buf_out: np.ndarray) -> Callable:
    """Perform discrete Fourier transforms using the FFTW library, and use it to get
    the power spectral density.

    Note
    ----
    This processor is composed of a factory function that is called using the
    `init_args` argument. The input and output waveforms are passed using
    `args`.

    Note
    ----
    FFTW optimizes the FFT algorithm based on the size of the arrays, with SIMD
    parallelized commands.  This optimization requires initialization, so this
    is a factory function that returns a Numba gufunc that performs the FFT.
    FFTW works on fixed memory buffers, so you must tell it what memory to use
    ahead of time.  When using this with
    :class:`~.dsp.processing_chain.ProcessingChain`, to ensure the correct
    buffers are used, call
    :meth:`~.dsp.processing_chain.ProcessingChain.get_variable` to give it the
    internal memory buffer directly. With :func:`~.dsp.build_dsp.build_dsp`,
    you can just give it the name, and it will automatically happen. The
    possible `dtypes` for the input/outputs are:

    =============================== ========= ============================ =============
    :class:`numpy.dtype`            Size      :class:`numpy.dtype`         Size
    ------------------------------- --------- ---------------------------- -------------
    ``complex64``                   :math:`n` ``float32``/``float``        :math:`n`
    ``complex128``                  :math:`n` ``float64``/``double``       :math:`n`
    ``complex256``/``clongdouble``  :math:`n` ``float128``/``longdouble``  :math:`n`
    ``float32``/``float``           :math:`n` ``float32``/``float``        :math:`n/2+1`
    ``float64``/``double``          :math:`n` ``float64``/``double``       :math:`n/2+1`
    ``float128``/``longdouble``     :math:`n` ``float128``/``longdouble``  :math:`n/2+1`
    =============================== ========= ============================ =============
    """

    # build intermediate array for the dft, which will be abs'd to get the PSD
    buf_dft = np.ndarray(
        buf_out.shape, np.dtype("complex" + str(buf_out.dtype.itemsize * 16))
    )
    try:
        dft_fun = FFTW(buf_in, buf_dft, axes=(-1,), direction="FFTW_FORWARD")
    except ValueError:
        raise ValueError(
            "incompatible array types/shapes. See function documentation for allowed values"
        )

    typesig = "void(" + str(buf_in.dtype) + "[:, :], " + str(buf_out.dtype) + "[:, :])"
    sizesig = "(m, n)->(m, n)" if buf_in.shape == buf_out.shape else "(m, n),(m, l)"

    @guvectorize(
        [typesig],
        sizesig,
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    def psd(wf_in: np.ndarray, psd_out: np.ndarray) -> None:
        dft_fun(wf_in, buf_dft)
        np.abs(buf_dft, psd_out)

    return psd
