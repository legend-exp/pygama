import numba as nb

kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def nb_poly(x, pars):
    """
    A polynomial function with pars following the polyfit convention.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like 
        Input data 
    pars : array-like 
        Coefficients of the polynomial, in polyfit convention
    TODO: add a CDF as well
    """
    result = x*0 # do x*0 to keep shape of x (scalar or array)
    if len(pars) == 0: return result
    result += pars[-1]
    for i in range(1, len(pars)):
        result += pars[-i-1]*x
        x = x*x
    return result
