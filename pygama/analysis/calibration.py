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
from pygama.analysis.histograms import get_bin_centers, get_gaussian_guess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.signal import argrelextrema, medfilt, find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import norm
import scipy.optimize as op


def peakdet(v, delta, x):
    """
    Converted from MATLAB script at: http://billauer.co.il/peakdet.html
    Returns two arrays: [maxtab, mintab] = peakdet(v, delta, x)
    An updated (vectorized) version is in pygama.dsp.transforms.peakdet
    """
    maxtab, mintab = [], []

    # sanity checks
    x, v = np.asarray(x), np.asarray(v)
    if len(v) != len(x): exit("Input vectors v and x must have same length")
    if not np.isscalar(delta): exit("Input argument delta must be a scalar")
    if delta <= 0: exit("Input argument delta must be positive")

    maxes, mins = [], []
    min, max = np.inf, -np.inf
    find_max = True
    for i in range(len(x)):

        # for i=0, all 4 of these get set
        if v[i] > max:
            max, imax = v[i], x[i]
        if v[i] < min:
            min, imin = v[i], x[i]

        if find_max:
            # if the sample is less than the current max,
            # declare the previous one a maximum, then set this as the new "min"
            if v[i] < max - delta:
                maxes.append((imax, max))
                min, imin = v[i], x[i]
                find_max = False
        else:
            # if the sample is more than the current min,
            # declare the previous one a minimum, then set this as the new "max"
            if v[i] > min + delta:
                mins.append((imin, min))
                max, imax = v[i], x[i]
                find_max = True

    return np.array(maxes), np.array(mins)


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
    peak_idxs = find_peaks_cwt(hist, np.arange(5, 10, 0.1), min_snr=5) #changed range from (0,6,0.1)
    peak_energies = bin_centers[peak_idxs]

    # pick the num_peaks most prominent peaks
    if max_num_peaks < len(peak_energies):
        peak_vals = hist[peak_idxs]
        sort_idxs = np.argsort(peak_vals)
        peak_idxs_max = peak_idxs[sort_idxs[-max_num_peaks:]]
        peak_energies = np.sort(bin_centers[peak_idxs_max])

    if test:
        plt.plot(bin_centers, hist, ds='steps', lw=1, c='b')
        for e in peak_energies:
            plt.axvline(e, color="r", lw=1, alpha=0.6)
        plt.xlabel("Energy [ADC]", ha='right', x=1)
        plt.ylabel("Filtered Spectrum", ha='right', y=1)
        plt.tight_layout()
        plt.show()
        #exit()

    return peak_energies


def match_peaks(data_pks, cal_pks, plotFigure=None):
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
    
    if plotFigure is not None:
        
        plt.scatter(data, cal, label='min.err:{:.2e}'.format(best_err))
        xs = np.linspace(data[0], data[-1], 10)
        plt.plot(xs, best_m * xs + best_b , c="r",
             label="y = {:.2f} x + {:.2f}".format(best_m,best_b) )
        plt.xlabel("Energy [ADC]", ha='right', x=1)
        plt.ylabel("Energy (keV)", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
    #exit()

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
    energy_hi = energy_series[(energy_series > np.percentile(energy_series, 20)) & \
                              (energy_series < np.percentile(energy_series, 99.9))] #uncommented range
    peak_energies = get_most_prominent_peaks(energy_hi,
                                            xlo=np.percentile(energy_series, 20),
                                            xhi=np.percentile(energy_series, 99.9),
                                            xpb=4,
                                            max_num_peaks = len(cal_peaks)) #modified limits and binwidth
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
    bin_size = 0.3  #keV

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

        bounds = ([0.9 * guess_e, 0.5 * guess_sigma, 0, 0, 0, 0, -0.1, 0], [
            1.1 * guess_e, 2 * guess_sigma, 0.1, 0.75, window_width_in_adc, 10,
            0.1, 5 * guess_area
        ])
        params, _ = fit_hist(
            radford_peak,
            peak_hist,
            bins,
            guess = [guess_e, guess_sigma, 1E-3, 0.7, 5, 0, 0, guess_area],
            bounds=bounds)
        print(params[1], params[5], params[6])

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

        fig, axs = plt.subplots(peak_num, 1, figsize=(12,16)) 
        
        #grid = gs.GridSpec(peak_num, 1, hspace=0.5)
        #ax_line = plt.subplot(grid[:, 1])  #changed plotting arrangement, added 2 figure objects
        #ax_spec = plt.subplot(grid[:, 2])

        for i, energy in enumerate(cal_peaks):
            ax_peak = plt.subplot(axs[i])
            bin_centers, peak_hist = plot_map[energy]
            bin_centers_keV = bin_centers * rough_kev_per_adc + rough_kev_offset
            params = fit_result_map[energy]
            ax_peak.plot(
                bin_centers_keV,
                peak_hist,
                ds="steps-mid",
                color="b",
                label="data")
            fit = radford_peak(bin_centers, *params)
            _, gaussian, bg, st, le = radford_peak(bin_centers, *params, components=True)
            ax_peak.plot(bin_centers_keV, fit, color="r", label="total fit")
            ax_peak.plot(bin_centers_keV, gaussian, color="c", label="gauss")
            ax_peak.plot(bin_centers_keV, le, color="g", label="le_tail")
            ax_peak.plot(bin_centers_keV, st, color="y", label="step")
            ax_peak.plot(bin_centers_keV, bg, color="k", label="linear bg")
            ax_peak.legend(loc="upper right")
            ax_peak.set_xlabel("Energy [keV]")
        
        plt.figure()
        ax_line = plt.subplot()
        ax_line.scatter(
            centers,
            cal_peaks)
        x = np.arange(0, max_adc, 1)
        ax_line.plot(x, linear_cal[0] * x + linear_cal[1])
        ax_line.set_xlabel("Energy [ADC]")
        ax_line.set_ylabel("Energy [keV]")
        
        plt.figure()
        ax_spec = plt.subplot()
        energies_cal = energy_series * linear_cal[0] + linear_cal[1]
        peak_hist, bins = np.histogram(energies_cal, bins=np.arange(0, 2700))
        ax_spec.semilogy(get_bin_centers(bins), peak_hist, ds="steps-mid")
        for pk in cal_peaks:
            ax_spec.axvline(pk, 0, 1e5, color='r')
        ax_spec.set_xlabel("Energy [keV]")
        ax_spec.set_ylabel("Counts")

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
