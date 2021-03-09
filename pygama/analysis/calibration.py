"""
routines for automatic calibration.
- peakdet (useful to find maxima in an array without taking derivative)
- get_most_prominent_peaks (find by looking for spikes in spectrum derivative)
- match_peaks (identify peaks based on ratios between known gamma energies)
- calibrate_tl208 (main routine -- fits multiple peaks w/ Radford peak shape)
- get_calibration_energies (a good place to put pk energies)
"""
import sys
import numpy as np
from pygama.analysis.peak_fitting import *
from pygama.analysis.histograms import get_bin_centers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.signal import argrelextrema, medfilt, find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import norm
import scipy.optimize as op

# match_peaks replacement
def poly_match(xx, yy, deg=-1, rtol=1e-5, atol=1e-8)
    """
    Find the polynomial function best matching pol(xx) = yy

    Finds the poly fit of xx to yy that obtains the most matches between pol(xx)
    and yy in the np.isclose() sense. If multiple fits give the same number of
    matches, the fit with the best gof is used, where gof is computed only among
    the matches.

    Assumes that the relationship between xx and yy is monotonic

    Parameters
    ----------
    xx : array-like
        domain data array. Must be sorted from least to largest. Must satisfy
        len(xx) >= len(yy)
    yy : array-like
        range data array: the values to which pol(xx) will be compared. Must be
        sorted from least to largest. Must satisfy len(yy) > max(2, deg+2)
    deg : int
        degree of the polynomial to be used. If deg = -1, will fit for a simple
        scaling: scale * xx = yy. If deg = 0, fits to a simple shift in the
        data: xx + shift = yy. Otherwise, deg is equivalent to the deg argument
        of np.polyfit()
    rtol : float
        the relative tolerance to be sent to np.isclose()
    atol : float
        the absolute tolerance to be sent to np.isclose(). Has the same units
        as yy.

    Returns
    -------
    pars, n_matches : None or float or array of floats, int
        pars: the parameters of the best poly fit. If deg = -1 (0) returns the best
        scaling (shifting) parameter. Otherwise, pars follows the convention
        used for the return value "p" of polyfit. Returns None when the inputs
        are bad.
        n_matches : the number of matches
    """

    # input handling
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    if len(xx) <= len(yy):
        print(f"poly_match: len(xx)={len(xx)} <= len(yy)={len(yy)}")
        return None, 0
    deg = int(deg)
    if deg < -1:
        print(f"poly_match: got bad deg = {deg}")
        return None, 0
    req_ylen = max(2, deg+2)
    if len(yy) < req_ylen:
        print(f"poly_match: len(yy) must be at least {req_ylen} for deg={deg}, got {len(yy)}")
        return None, 0

    # build itup: the indices in xx to compare with the values in yy
    itup = list(range(len(yy)))
    n_close = 0
    gof = np.inf # lower is better gof
    while True:
        xx_i = xx[itup]
        gof_i = np.inf

        # simple scaling
        if deg == -1:
            pars_i = np.sum(yy) / np.sum(xx_i)
            polxx = pars_i * xx_i

        # simple shift
        elif deg == 0:
            pars_i = (np.sum(yy) - np.sum(xx_i)) / len(yy)
            polxx = xx_i + shift

        # generic poly of degree >= 1
        else:
            pars_i = np.polyfit(xx_i, yy, deg)
            polxx = np.zeros(len(yy))
            xxn = np.ones(len(yy))
            for j in len(pars_i)
                polxx += xxn*pars_i[-j-1]
                xxn *= xx_i

        # by here we have the best polxx. Search for matches and store pars_i if
        # its the best so far
        matches = np.isclose(polxx, yy, rtol=rtol, atol=atol)
        n_close_i = np.sum(matches)
        if n_close_i >= n_close_i:
            gof_i = np.sum(np.power(polxx[matches] - yy[matches], 2))
            if n_close_i > n_close or (n_close_i == n_close and gof_i < gof):
                n_close = n_close_i
                gof = gof_i
                pars = pars_i

        # increment itup
        # first find the index of itup that needs to be incremented
        ii = 0
        while ii < len(yy)-1:
            if itup[ii] < itup[ii+1]-1: break
            ii += 1
        # quit if ii is the last index of itup and it's already maxed out
        if ii == len(yy) and itup[ii] == len(xx)-1: break
        # otherwise increment ii and reset indices < ii
        itup[ii]
        itup[0:ii] = list(range(ii))

    return pars


def get_i_local_extrema(data, delta):
    """
    Get lists of indices of the local maxima and minima of data

    The "local" extrema are those maxima / minima that have heights / depths of
    at least delta.

    Converted from MATLAB script at: http://billauer.co.il/peakdet.html

    Parameters
    ----------
    data : array-like
        the array of data within which extrema will be found
    delta : scalar
        the absolute level by which data must vary (in one direction) about an
        extremum in order for it to be tagged

    Returns
    -------
    maxes, mins : 2-tuple ( array, array )
        A 2-tuple containing arrays of variable length that hold the indices of
        the identified local maxima (first tuple element) and minima (second
        tuple element)
    """

    # prepare output
    maxes, mins = [], []

    # sanity checks
    data = np.asarray(data)
    if not np.isscalar(delta):
        print("get_i_local_extrema: Input argument delta must be a scalar")
        return np.array(maxes), np.array(mins)
    if delta <= 0:
        print(f"get_i_local_extrema: delta ({delta}) must be positive")
        return np.array(maxes), np.array(mins)

    # now loop over data
    imax, imin = 0, 0
    find_max = True
    for i in range(len(data)):

        if data[i] > data[imax]: imax = i
        if data[i] < data[imin]: imin = i

        if find_max:
            # if the sample is less than the current max by more than delta,
            # declare the previous one a maximum, then set this as the new "min"
            if data[i] < data[imax] - delta:
                maxes.append(imax)
                imin = i
                find_max = False
        else:
            # if the sample is more than the current min by more than delta,
            # declare the previous one a minimum, then set this as the new "max"
            if data[i] > data[imin] + delta:
                mins.append(imin)
                imax = i
                find_max = True

    return np.array(maxes), np.array(mins)

def get_i_local_maxima(data, delta): return get_i_local_extrema(data, delta)[0]

def get_i_local_minima(data, delta): return get_i_local_extrema(data, delta)[1]


def get_most_prominent_peaks(energySeries, xlo, xhi, xpb,
                             max_num_peaks=np.inf, test=False):
    """
    find the most prominent peaks in a spectrum by looking for spikes in derivative of spectrum
    energySeries: array of measured energies
    max_num_peaks = maximum number of most prominent peaks to find
    return a histogram around the most prominent peak in a spectrum of a given percentage of width
    """
    nb = int((xhi-xlo)/xpb)
    hist, bin_edges = np.histogram(energySeries, range=(xlo, xhi), bins=nb)
    bin_centers = get_bin_centers(bin_edges)

    # median filter along the spectrum, do this as a "baseline subtraction"
    hist_med = medfilt(hist, 21)
    hist = hist - hist_med

    # identify peaks with a scipy function (could be improved ...)
    peak_idxs = find_peaks_cwt(hist, np.arange(1, 6, 0.1), min_snr=5)
    peak_energies = bin_centers[peak_idxs]

    # pick the num_peaks most prominent peaks
    if max_num_peaks < len(peak_energies):
        peak_vals = hist[peak_idxs]
        sort_idxs = np.argsort(peak_vals)
        peak_idxs_max = peak_idxs[sort_idxs[-max_num_peaks:]]
        peak_energies = np.sort(bin_centers[peak_idxs_max])

    if test:
        plt.plot(bin_centers, hist, ls='steps', lw=1, c='b')
        for e in peak_energies:
            plt.axvline(e, color="r", lw=1, alpha=0.6)
        plt.xlabel("Energy [uncal]", ha='right', x=1)
        plt.ylabel("Filtered Spectrum", ha='right', y=1)
        plt.tight_layout()
        plt.show()
        exit()

    return peak_energies


def match_peaks(data_pks, cal_pks):
    """
    Match uncalibrated peaks with literature energy values.
    """
    from itertools import combinations
    from scipy.stats import linregress

    n_pks = len(cal_pks) if len(cal_pks) < len(data_pks) else len(data_pks)

    cal_sets = combinations(range(len(cal_pks)), n_pks)
    data_sets = combinations(range(len(data_pks)), n_pks)

    best_err, best_m, best_b = np.inf, None, None
    for i,cal_set in enumerate(cal_sets):

        cal = cal_pks[list(cal_set)] # lit energies for this set

        for data_set in data_sets:

            data = data_pks[list(data_set)] # uncal energies for this set

            m, b, _, _, _ = linregress(data, y=cal)
            err = np.sum((cal - (m * data + b))**2)

            if err < best_err:
                best_err, best_m, best_b = err, m, b

    print(i, best_err)
    print("cal:",cal)
    print("data:",data)
    plt.scatter(data, cal, label='min.err:{:.2e}'.format(err))
    xs = np.linspace(data[0], data[-1], 10)
    plt.plot(xs, best_m * xs + best_b , c="r",
             label="y = {:.2f} x + {:.2f}".format(best_m,best_b) )
    plt.xlabel("Energy (uncal)", ha='right', x=1)
    plt.ylabel("Energy (keV)", ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    exit()

    return best_m, best_b


def calibrate_tl208(energy_series, cal_peaks=None, plotFigure=None):
    """
    energy_series: array of energies we want to calibrate
    cal_peaks: array of peaks to fit

    1.) we find the 2614 peak by looking for the tallest peak at >0.1 the max adc value
    2.) fit that peak to get a rough guess at a calibration to find other peaks with
    3.) fit each peak in peak_energies
    4.) do a linear fit to the peak centroids to find a calibration
    """

    if cal_peaks is None:
        cal_peaks = np.array(
            [238.632, 510.770, 583.191, 727.330, 860.564,
             2614.553])  #get_calibration_energies(peak_energies)
    else:
        cal_peaks = np.array(cal_peaks)

    if len(energy_series) < 100:
        return 1, 0

    #get 10 most prominent ~high e peaks
    max_adc = np.amax(energy_series)
    energy_hi = energy_series  #[ (energy_series > np.percentile(energy_series, 20)) & (energy_series < np.percentile(energy_series, 99.9))]

    peak_energies, peak_e_err = get_most_prominent_peaks(energy_hi,)
    rough_kev_per_adc, rough_kev_offset = match_peaks(peak_energies, cal_peaks)
    e_cal_rough = rough_kev_per_adc * energy_series + rough_kev_offset

    # return rough_kev_per_adc, rough_kev_offset
    # print(energy_series)
    # plt.ion()
    # plt.figure()
    # # for peak in cal_peaks:
    # #     plt.axvline(peak, c="r", ls=":")
    # # energy_series.hist()
    # # for peak in peak_energies:
    # #      plt.axvline(peak, c="r", ls=":")
    # #
    # plt.hist(energy_series)
    # # plt.hist(e_cal_rough[e_cal_rough>100], bins=2700)
    # val = input("do i exist?")
    # exit()

    ###############################################
    #Do a real fit to every peak in peak_energies
    ###############################################
    max_adc = np.amax(energy_series)

    peak_num = len(cal_peaks)
    centers = np.zeros(peak_num)
    fit_result_map = {}
    bin_size = 0.2  #keV

    if plotFigure is not None:
        plot_map = {}

    for i, energy in enumerate(cal_peaks):
        window_width = 10  #keV
        window_width_in_adc = (window_width) / rough_kev_per_adc
        energy_in_adc = (energy - rough_kev_offset) / rough_kev_per_adc
        bin_size_adc = (bin_size) / rough_kev_per_adc

        peak_vals = energy_series[
            (energy_series > energy_in_adc - window_width_in_adc) &
            (energy_series < energy_in_adc + window_width_in_adc)]

        peak_hist, bins = np.histogram(
            peak_vals,
            bins=np.arange(energy_in_adc - window_width_in_adc,
                           energy_in_adc + window_width_in_adc + bin_size_adc,
                           bin_size_adc))
        bin_centers = get_bin_centers(bins)
        # plt.ion()
        # plt.figure()
        # plt.plot(bin_centers,peak_hist,  color="k", ls="steps")

        # inpu = input("q to quit...")
        # if inpu == "q": exit()

        try:
            guess_e, guess_sigma, guess_area = get_gaussian_guess(
                peak_hist, bin_centers)
        except IndexError:
            print("\n\nIt looks like there may not be a peak at {} keV".format(
                energy))
            print("Here is a plot of the area I'm searching for a peak...")
            plt.ion()
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.plot(bin_centers, peak_hist, color="k", ls="steps")
            plt.subplot(122)
            plt.hist(e_cal_rough, bins=2700, histtype="step")
            input("-->press any key to continue...")
            sys.exit()

        plt.plot(
            bin_centers,
            gauss(bin_centers, guess_e, guess_sigma, guess_area),
            color="b")

        # inpu = input("q to quit...")
        # if inpu == "q": exit()

        bounds = ([0.9 * guess_e, 0.5 * guess_sigma, 0, 0, 0, 0, 0], [
            1.1 * guess_e, 2 * guess_sigma, 0.1, 0.75, window_width_in_adc, 10,
            5 * guess_area
        ])
        params = fit_binned(
            radford_peak,
            peak_hist,
            bin_centers,
            [guess_e, guess_sigma, 1E-3, 0.7, 5, 0, guess_area],
        )  #bounds=bounds)

        plt.plot(bin_centers, radford_peak(bin_centers, *params), color="r")

        # inpu = input("q to quit...")
        # if inpu == "q": exit()

        fit_result_map[energy] = params
        centers[i] = params[0]

        if plotFigure is not None:
            plot_map[energy] = (bin_centers, peak_hist)

    #Do a linear fit to find the calibration
    linear_cal = np.polyfit(centers, cal_peaks, deg=1)

    if plotFigure is not None:

        plt.figure(plotFigure.number)
        plt.clf()

        grid = gs.GridSpec(peak_num, 3)
        ax_line = plt.subplot(grid[:, 1])
        ax_spec = plt.subplot(grid[:, 2])

        for i, energy in enumerate(cal_peaks):
            ax_peak = plt.subplot(grid[i, 0])
            bin_centers, peak_hist = plot_map[energy]
            params = fit_result_map[energy]
            ax_peak.plot(
                bin_centers * rough_kev_per_adc + rough_kev_offset,
                peak_hist,
                ls="steps-mid",
                color="k")
            fit = radford_peak(bin_centers, *params)
            ax_peak.plot(
                bin_centers * rough_kev_per_adc + rough_kev_offset,
                fit,
                color="b")

        ax_peak.set_xlabel("Energy [keV]")

        ax_line.scatter(
            centers,
            cal_peaks,
        )

        x = np.arange(0, max_adc, 1)
        ax_line.plot(x, linear_cal[0] * x + linear_cal[1])
        ax_line.set_xlabel("ADC")
        ax_line.set_ylabel("Energy [keV]")

        energies_cal = energy_series * linear_cal[0] + linear_cal[1]
        peak_hist, bins = np.histogram(energies_cal, bins=np.arange(0, 2700))
        ax_spec.semilogy(get_bin_centers(bins), peak_hist, ls="steps-mid")
        ax_spec.set_xlabel("Energy [keV]")

    return linear_cal


def get_calibration_energies(cal_type):
    if cal_type == "th228":
        return np.array([238, 277, 300, 452, 510.77, 583.191,
                         727, 763, 785, 860.564, 1620, 2614.533],
                        dtype="double")

    elif cal_type == "uwmjlab":
        # return np.array([239, 295, 351, 510, 583, 609, 911, 969, 1120,
        #                  1258, 1378, 1401, 1460, 1588, 1764, 2204, 2615],
        #                 dtype="double")
        return np.array([239, 911, 1460, 1764, 2615],
                        dtype="double")
    else:
        raise ValueError
