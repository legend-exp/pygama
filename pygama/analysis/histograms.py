"""
pygama convenience functions for histograms.
we don't want to create a monolithic class like TH1 in ROOT,
it encourages users to understand what their code is actually doing.
"""
import numpy as np
import matplotlib.pyplot as plt


def get_hist(np_arr, nbins=100, range=None, dx=None, wts=None):
    """
    wrapper for numpy.histogram, with optional weights for each element.
    allows a user to set a number of bins and auto-detect the x-range,
    or you can set a range (x_lo, x_hi) and an increment (dx), which
    overrides the nbins argument.
    """
    if dx is not None:
        nbins = int((range[1] - range[0]) / dx)

    hist, bins = np.histogram(np_arr, bins=nbins, range=range, weights=wts)

    if wts is None:
        return hist, bins, hist
    else:
        var, bins = np.histogram(np_arr, bins=nbins, weights=wts*wts)
        return hist, bins, var


def get_fwhm(hist, bin_centers):
    """
    find a FWHM from a hist
    """
    idxs_over_50 = hist > 0.5 * np.amax(hist)
    first_energy = bin_centers[np.argmax(idxs_over_50)]
    last_energy = bin_centers[len(idxs_over_50) - np.argmax(idxs_over_50[::-1])]
    return (last_energy - first_energy)


def get_bin_centers(bins):
    """
    convenience func for plot_hist
    """
    return (bins[:-1] + bins[1:]) / 2.


def get_bin_widths(bins):
    """
    convenience func for plot_hist
    """
    return (bins[1:] - bins[:-1])


def plot_hist(hist, bins, var=None, **kwargs):
    """
    plot a step histogram, with optional error bars
    """
    if var is None:
        plt.step(bins, np.concatenate((hist, [0])), where="post")
    else:
        plt.errorbar(get_bin_centers(bins), hist,
                     xerr=get_bin_widths(bins) / 2, yerr=np.sqrt(var),
                     fmt='none', **kwargs)


def plot_func(func, pars, range=None, npx=None, **kwargs):
    """
    plot a function.  take care of the x-axis points automatically
    """
    if npx is None:
        npx = 100
    if range is None:
        range = plt.xlim()
    xvals = np.linspace(range[0], range[1], npx)
    plt.plot(xvals, func(xvals, *pars), **kwargs)


def print_fit_results(pars, cov, par_names=None):
    """
    convenience function for scipy.optimize.curve_fit results
    """
    if par_names is None:
        par_names = []
        for i in range(len(pars)):
            par_names.append("p" + str(i))
    for i in range(len(pars)):
        print(par_names[i], "=", pars[i], "+/-", np.sqrt(cov[i][i]))


def get_gaussian_guess(hist, bin_centers):
    """
    given a hist, gives guesses for mu, sigma, and amplitude
    """
    max_idx = np.argmax(hist)
    guess_e = bin_centers[max_idx]
    guess_amp = hist[max_idx]

    # find 50% amp bounds on both sides for a FWHM guess
    guess_sigma = get_fwhm(hist, bin_centers) / 2.355  # FWHM to sigma
    guess_area = guess_amp * guess_sigma * np.sqrt(2 * np.pi)

    return (guess_e, guess_sigma, guess_area)
