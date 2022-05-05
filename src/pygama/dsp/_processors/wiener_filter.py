import numpy as np
from numba import guvectorize

import pygama.lgdo.lh5_store as lh5
from pygama.dsp.errors import DSPFatal


def wiener_filter(file_name_array):
    """Apply a wiener filter to the waveform.

    Note that this convolution is performed in the frequency domain

    Parameters
    ----------
    file_name_array : string
        Array with path to an lh5 file containing the time domain version
        of the superpulse in one column and noise waveform in another,
        the superpulse group must be titled 'spms/processed/superpulse' and
        the noise waveform must be called 'spms/processed/noise_wf'

    Examples
    --------
    .. code-block :: json

        "wf_wiener": {
            "function": "wiener_filter",
            "module": "pygama.dsp.processors",
            "args": ["wf_bl_fft", "wf_wiener(2000,f)"],
            "unit": "dB",
            "prereqs": ["wf_bl_fft"],
            "init_args": ["/path/to/file/wiener.lh5"]
        }
    """

    sto = lh5.LH5Store()

    # Check that the file is valid and the data is in the correct format

    try: file_name_array[0]
    except: raise DSPFatal('init_args must be an array with the filename')

    file_name = file_name_array[0]

    try: f = sto.gimme_file(file_name, 'r')
    except: raise DSPFatal('File must be a valid lh5 file')

    if 'spms/processed/superpulse' not in f:
        raise DSPFatal('lh5 file must have \'spms/processed/superpulse\' as a group')

    if 'spms/processed/noise_wf' not in f:
        raise DSPFatal('lh5 file must have \'spms/processed/noise_wf\' as a group')

    # Read in the data

    superpulse, _ = sto.read_object('spms/processed/superpulse', file_name)
    superpulse = superpulse.nda

    noise_wf, _ = sto.read_object('spms/processed/noise_wf', file_name)
    noise_wf = noise_wf.nda

    # Now check that the data are valid

    if len(superpulse) <= 0:
        raise DSPFatal('The length of the filter must be positive')

    if len(superpulse) != len(noise_wf):
        raise DSPFatal('The length of the superpulse must be equal to the length of the noise waveform')

    if np.argmax(superpulse) <= 0 or np.argmax(superpulse) > len(superpulse):
        raise DSPFatal('The index of the maximum of the superpulse must occur within the waveform')

    # Transform these to the frequency domain to eventually create the wiener filter

    fft_superpulse = np.fft.fft(superpulse)
    fft_noise_wf = np.fft.fft(noise_wf)

    # Create the point spread function for the detector's response

    def PSF(superpulse, fft_superpulse):

        delta = np.zeros_like(superpulse)
        arg_max = np.argmax(superpulse)
        delta[arg_max] = np.amax(superpulse)

        return fft_superpulse/np.fft.fft(delta)

    # Now create the wiener filter in the frequency domain

    fft_PSF = PSF(superpulse, fft_superpulse)
    PSD_noise_wf = fft_noise_wf*np.conj(fft_noise_wf)
    PSD_superpulse = fft_superpulse*np.conj(fft_superpulse)

    w_filter = (np.conj(fft_PSF))/((fft_PSF*np.conj(fft_PSF))+(PSD_noise_wf/PSD_superpulse))

    # Create a factory function that performs the convolution with the wiener filter, the output is still in the frequency domain

    @guvectorize(["void(complex64[:], complex64[:])",
                  "void(complex128[:], complex128[:])"],
                 "(n)->(n)", forceobj=True)
    def wiener_out(fft_w_in, fft_w_out):
        """
        Parameters
        ----------
        fft_w_in : array-like
            The fourier transform input waveform
        fft_w_out : array-like
            The filtered waveform, in the frequency domain
        """
        fft_w_out[:] = np.nan

        if np.isnan(fft_w_in).any():
            return

        if len(w_filter) != len(fft_w_in):
            raise DSPFatal('The filter is not the same length of the input waveform')

        fft_w_out[:] = fft_w_in * w_filter

    return wiener_out
