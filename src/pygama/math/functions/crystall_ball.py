from scipy.stats import crystalball
import numba as nb
import numpy as np
import math


kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def nb_xtalball_pdf(x, beta, m, mu, sigma, A): 
    """
    power-law tail plus gaussian. 
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions. 
    It computes A*pdf(y, beta, m)/sigma, with y=(x-mu)/sigma

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
    """
    if (beta <= 0) or (m <= 1):
        raise ValueError
    # Define some constants to calculate the function 
    const_A = np.power(m/np.abs(beta),m) * np.exp(-1*beta**2/2)
    const_B = m/np.abs(beta) - np.abs(beta)

    # Calculate the normalization constant
    normalization_denom = (np.sqrt(np.pi/2)*(math.erf(beta/np.sqrt(2)) + 1)) + ((const_A*np.power(const_B+beta,1-m))/(m-1))
    N = 1/normalization_denom

    # Shift the distribution 
    y = (x-mu)/sigma

    # Compute the mask 
    if np.amin(y) <= -1*beta: 
        idx_beta = np.where(y<=-1*beta)[0][-1]
    else: 
        idx_beta = 0
    power_law = y[:idx_beta]
    gauss = y[idx_beta:]

    pdf = np.empty_like(x)

    # Compute the power law part of the pdf 
    for i in range(len(power_law)): 
        pdf[i] = N*const_A*np.power((const_B-power_law[i]),-1*m)
    
    # Compute the gaussian part of the pdf 
    for i in range(len(gauss)): 
        pdf[i+idx_beta] = N*np.exp(-1*gauss[i]**2/2)
    
    return A*pdf/sigma


@nb.njit(**kwd)
def nb_xtalball_cdf(x, beta, m, mu, sigma, A): 
    if (beta <= 0) or (m <= 1):
        raise ValueError
    # Define some constants to calculate the function 
    const_A = np.power(m/np.abs(beta),m) * np.exp(-1*beta**2/2)
    const_B = m/np.abs(beta) - np.abs(beta)

    # Calculate the normalization constant
    normalization_denom = (np.sqrt(np.pi/2)*(math.erf(beta/np.sqrt(2)) + 1))\
        + ((const_A*np.power(const_B+beta,1-m))/(m-1))
    N = 1/normalization_denom

    # Shift the distribution 
    y = (x-mu)/sigma

    # Compute the mask 
    if np.amin(y) <= -1*beta: 
        idx_beta = np.where(y<=-1*beta)[0][-1]
    else: 
        idx_beta = 0
    power_law = y[:idx_beta]
    gauss = y[idx_beta:]

    cdf = np.empty_like(x)

    # Compute the power law part of the cdf 
    for i in range(len(power_law)): 
        cdf[i] = N*const_A*np.power(const_B-power_law[i],1-m)/(m-1)
    
    # Compute the gaussian part of the cdf 
    for i in range(len(gauss)): 
        cdf[i+idx_beta] = ((const_A*N*np.power(const_B+beta,1-m))/(m-1))\
            + N*np.sqrt(np.pi/2)*(math.erf(beta/np.sqrt(2))+math.erf(gauss[i]/np.sqrt(2)))
    
    return A*cdf
