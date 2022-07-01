import numba as nb

kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def nb_poly(x, pars):
    """
    A polynomial function with pars following the polyfit convention
    TO DO: add a CDF as well
    """
    result = x*0 # do x*0 to keep shape of x (scalar or array)
    if len(pars) == 0: return result
    result += pars[-1]
    for i in range(1, len(pars)):
        result += pars[-i-1]*x
        x = x*x
    return result
