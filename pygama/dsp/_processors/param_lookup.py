import numpy as np
from numba import guvectorize, types, from_dtype

def param_lookup(param_dict, default_val, dtype):
    """
    Generate the ufunc lookup(channel, val), which returns a NumPy array of
    values corresponding to various channels that are looked up in the provided
    param_dict.  If there is no key, use default_val instead.
    """
    out_type = from_dtype(np.dtype(dtype))

    # convert types to avoid any necessity of casting
    param_dict  = {types.uint32(k): out_type(v) for k, v in param_dict.items()}
    default_val = out_type(default_val)
    
    @guvectorize(["void(uint32, " + out_type.name + "[:])"],
                 "()->()", forceobj=True)
    def lookup(channel, val):
        """
        Look up a value for the provided channel from a dictionary provided at
        compile time.
        """
        val[0] = param_dict.get(channel, default_val)

    return lookup
