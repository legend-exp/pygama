import numpy as np
from pygama import lh5
from numba import guvectorize
from pygama.dsp.errors import DSPFatal


def inl_correction(file_name_array):
    """
    Apply ADC INL correction to the waveform.
    
    Initialization Parameters
    -------------------------
    file_name_array : string
            Array with path to an lh5 file containing a dictionary/struct with the ADC values as keys and the nonlinearity correction as values

    Parameters
    ----------
    w_in : array-like
           The input waveform
    w_out: array-like
           The INL corrected waveform     
           
    Processing Chain Example
    ------------------------
    "wf_inl": {
        "function": "inl_correction",
        "module": "pygama.dsp.processors",
        "args": ["waveform", "wf_inl"],
        "unit": "ADC",
        "prereqs": ["waveform"],
        "init_args": ["/path/to/file/inl_table.lh5"]
    }
    """
    
    sto = lh5.Store()

    # Check that the file is valid and the data is in the correct format

    try: file_name_array[0]
    except: raise DSPFatal('init_args must be an array with the filename')
    
    file_name = file_name_array[0]
     
    try: f = sto.gimme_file(file_name, 'r')
    except: raise DSPFatal('File must be a valid lh5 file') 

    if 'inl_table' not in f:
        raise DSPFatal('lh5 file must have \'inl_table\' as a group')
        
    # Read in the data 
    
    inl_table, _ = sto.read_object('inl_table', file_name)
     
    # Create a factory function that performs the convolution with the Wiener filter, the output is still in the frequency domain 
    @guvectorize(["void(float32[:], float32[:])",
                  "void(float64[:], float64[:])"],
                 "(n)->(n)", forceobj=True)
    def correct_wf(w_in, w_out):
        w_out[:] = np.nan
        inls = [inl_table[str(int(w))].value for w in w_in]
        w_out[:] = w_in + inls

    return correct_wf