from numba import guvectorize
from pyfftw import FFTW


def dft(buf_in, buf_out):
    """
    Perform discrete fourier transforms using the FFTW library. FFTW optimizes
    the fft algorithm based on the size of the arrays, with SIMD parallelized
    commands. This optimization requires initialization, so this is a factory
    function that returns a numba gufunc that performs the FFT. FFTW works on
    fixed memory buffers, so you must tell it what memory to use ahead of time.
    When using this with ProcessingChain, to ensure the correct buffers are used
    call ProcessingChain.get_variable('var_name') to give it the internal memory
    buffer directly (with raw_to_dsp, you can just give it the name and it will
    automatically happen!). The possible dtypes for the input/outputs are:
    - float32/float (size n) -> complex64 (size n/2+1)
    - float64/double (size n) -> complex128 (size n/2+1)
    - float128/longdouble (size n) -> complex256/clongdouble (size n/2+1)
    - complex64 (size n) -> complex64 (size n)
    - complex128 (size n) -> complex128 (size n)
    - complex256/clongdouble (size n) -> complex256/clongdouble (size n)
    """

    try:
        dft_fun = FFTW(buf_in, buf_out, axes=(-1,), direction='FFTW_FORWARD')
    except ValueError:
        raise ValueError("""Incompatible array types/shapes. Allowed:
    - float32/float (size n) -> complex64 (size n/2+1)
    - float64/double (size n) -> complex128 (size n/2+1)
    - float128/longdouble (size n) -> complex256/clongdouble (size n/2+1)
    - complex64 (size n) -> complex64 (size n)
    - complex128 (size n) -> complex128 (size n)
    - complex256/clongdouble (size n) -> complex256/clongdouble (size n)""")
    
    typesig = 'void(' + str(buf_in.dtype) + '[:, :], ' + str(buf_out.dtype) + '[:, :])'
    sizesig = '(m, n)->(m, n)' if buf_in.shape == buf_out.shape else '(m, n),(m, l)'
    print(typesig, sizesig)
    @guvectorize([typesig], sizesig, forceobj=True)
    def dft(wf_in, dft_out):
        dft_fun(wf_in, dft_out)

    return dft

def inv_dft(buf_in, buf_out):
    """
    Perform inverse discrete fourier transforms using FFTW. FFTW optimizes
    the fft algorithm based on the size of the arrays, with SIMD parallelized
    commands. This optimization requires initialization, so this is a factory
    function that returns a numba gufunc that performs the FFT. FFTW works on
    fixed memory buffers, so you must tell it what memory to use ahead of time.
    When using this with ProcessingChain, to ensure the correct buffers are used
    call ProcessingChain.get_variable('var_name') to give it the internal memory
    buffer directly (with raw_to_dsp, you can just give it the name and it will
    automatically happen!). The possible dtypes for the input/outputs are:
    - complex64 (size n/2+1) -> float32/float (size n) 
    - complex128 (size n/2+1) -> float64/double (size n)
    - complex256/clongdouble (size n/2+1) -> float128/longdouble (size n)
    - complex64 (size n) -> complex64 (size n)
    - complex128 (size n) -> complex128 (size n)
    - complex256/clongdouble (size n) -> complex256/clongdouble (size n)
    """

    try:
        idft_fun = FFTW(buf_in, buf_out, axes=(-1,), direction='FFTW_BACKWARD')
    except ValueError:
        raise ValueError("""Incompatible array types/shapes. Allowed:
    - complex64 (size n/2+1) -> float32/float (size n) 
    - complex128 (size n/2+1) -> float64/double (size n)
    - complex256/clongdouble (size n/2+1) -> float128/longdouble (size n)
    - complex64 (size n) -> complex64 (size n)
    - complex128 (size n) -> complex128 (size n)
    - complex256/clongdouble (size n) -> complex256/clongdouble (size n)""")
    
    typesig = 'void(' + str(buf_in.dtype) + '[:, :], ' + str(buf_out.dtype) + '[:, :])'
    sizesig = '(m, n)->(m, n)' if buf_in.shape == buf_out.shape else '(m, n),(m, l)'
    print(typesig, sizesig)
    @guvectorize([typesig], sizesig, forceobj=True)
    def inv_dft(wf_in, dft_out):
        idft_fun(wf_in, dft_out)

    return inv_dft

