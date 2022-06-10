from scipy.stats import crystalball


def xtalball(x, mu, sigma, A, beta, m):
    """
    power-law tail plus gaussian https://en.wikipedia.org/wiki/Crystal_Ball_function
    To Do: Make an Numba JIT version, as well as a CDF
    """
    return A * crystalball.pdf(x, beta, m, loc=mu, scale=sigma)
