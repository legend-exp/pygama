import math

import numba as nb
import numpy as np


@nb.vectorize([nb.float64(nb.float64,nb.float64,nb.float64,nb.float64,nb.float64,nb.float64),
nb.float32(nb.float32,nb.float32,nb.float32,nb.float32,nb.float32,nb.float32)])
def nb_xtalball_pdf(x: float, beta: float, m: float, mu: float, sigma: float, A: float) -> float:
    """
    PDF of a power-law tail plus gaussian.
    As a Numba vectorized function, it runs slightly faster than
    'out of the box' functions.
    It computes

    .. math::
        y = \\frac{x-mu}{sigma}


        PDF = A\\frac{pdf(y, beta, m)}{sigma}

    Parameters
    ----------
    x : array-like
        The input data
    beta : float
        The point where the pdf changes from power-law to Gaussian
    m : float
        The power of the power-law tail
    mu : float
        The amount to shift the distribution
    sigma : float
        The amount to scale the distribution
    A : float
        An overall scale factor

    Returns
    -------
    y : float
        The value at x given the parameters

    TODO: Potentially replace by a guvectorized factory function so that the normalization isn't computed every time
    """
    if (beta <= 0) or (m <= 1):
        raise ValueError("beta must be greater than 0, and m must be greater than 1")
    # Define some constants to calculate the function
    const_A = np.power(m/np.abs(beta),m) * np.exp(-1*beta**2/2)
    const_B = m/np.abs(beta) - np.abs(beta)

    # Calculate the normalization constant
    normalization_denom = (np.sqrt(np.pi/2)*(math.erf(beta/np.sqrt(2)) + 1))\
        + ((const_A*np.power(const_B+beta,1-m))/(m-1))
    N = 1/normalization_denom

    # Shift the distribution
    y = (x-mu)/sigma

    # Check if it is powerlaw
    if y <= -1*beta:
        return N*const_A*np.power((const_B-y),-1*m)*A/sigma

    # If it isn't power law, then it Gaussian

    else:
        return N*np.exp(-1*y**2/2)*A/sigma


@nb.vectorize([nb.float64(nb.float64,nb.float64,nb.float64,nb.float64,nb.float64,nb.float64),
nb.float32(nb.float32,nb.float32,nb.float32,nb.float32,nb.float32,nb.float32)])
def nb_xtalball_cdf(x: float, beta: float, m: float, mu: float, sigma: float, A: float) -> float:
    """
    CDF for power-law tail plus gaussian.
    As a Numba vectorized function, it runs slightly faster than
    'out of the box' functions.
    It computes

    .. math::
        y = \\frac{x-mu}{sigma}


        PDF = A\\times cdf(y, beta, m)

    Parameters
    ----------
    x : array-like
        The input data
    beta : float
        The point where the Cdf changes from power-law to Gaussian
    m : float
        The power of the power-law tail
    mu : float
        The amount to shift the distribution
    sigma : float
        The amount to scale the distribution
    A : float
        An overall scale factor
    TODO: Potentially replace by a guvectorized factory function so that the normalization isn't computed every time
    """
    if (beta <= 0) or (m <= 1):
        raise ValueError("beta must be greater than 0, and m must be greater than 1")
    # Define some constants to calculate the function
    const_A = np.power(m/np.abs(beta),m) * np.exp(-1*beta**2/2)
    const_B = m/np.abs(beta) - np.abs(beta)

    # Calculate the normalization constant
    normalization_denom = (np.sqrt(np.pi/2)*(math.erf(beta/np.sqrt(2)) + 1))\
        + ((const_A*np.power(const_B+beta,1-m))/(m-1))
    N = 1/normalization_denom

    # Shift the distribution
    y = (x-mu)/sigma

    # Check if it is in the power law part
    if y <= -1*beta:
        return N*const_A*np.power(const_B-y,1-m)/(m-1)*A

    # If it isn't in the power law, then it is Gaussian
    else:
        return (((const_A*N*np.power(const_B+beta,1-m))/(m-1))\
            + N*np.sqrt(np.pi/2)*(math.erf(beta/np.sqrt(2))+math.erf(y/np.sqrt(2))))*A
