"""
mainly pulser tagging
- gaussian_cut (fits data to a gaussian, returns mean +/- cut_sigma values)
- xtalball_cut (fits data to a crystalball, returns mean +/- cut_sigma values)
- find_pulser_properties (find pulser by looking for which peak has a constant time between events)
- tag_pulsers
"""
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pygama.math.peak_fitting import *


def gaussian_cut(data, cut_sigma=3, plotAxis=None):
    """
    fits data to a gaussian, returns mean +/- cut_sigma values for a cut
    """

    nbins = 100

    median = np.median(data)
    width = np.percentile(data, 80) - np.percentile(data, 20)

    good_data = data[(data > (median - 4 * width)) & (data < (median + 4 * width))]

    hist, bins = np.histogram(good_data, bins=101)  # np.linspace(1,5,101)
    bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2

    # fit gaussians to that
    # result = fit_unbinned(gauss, hist, [median, width/2] )
    # print("unbinned: {}".format(result))

    result = fit_binned(
        gauss,
        hist,
        bin_centers,
        [median, width / 2, np.amax(hist) * (width / 2) * np.sqrt(2 * np.pi)],
    )
    # print("binned: {}".format(result))
    cut_lo = result[0] - cut_sigma * result[1]
    cut_hi = result[0] + cut_sigma * result[1]

    if plotAxis is not None:
        plotAxis.plot(bin_centers, hist, ls="steps-mid", color="k", label="data")
        fit = gauss(bin_centers, *result)
        plotAxis.plot(bin_centers, fit, label="gaussian fit")
        plotAxis.axvline(result[0], color="g", label="fit mean")
        plotAxis.axvline(cut_lo, color="r", label=f"+/- {cut_sigma} sigma")
        plotAxis.axvline(cut_hi, color="r")
        plotAxis.legend()
        # plt.xlabel(params[i])

    return cut_lo, cut_hi, result[0], cut_sigma


def xtalball_cut(data, cut_sigma=3, plotFigure=None):
    """
    fits data to a crystalball, returns mean +/- cut_sigma values for a cut
    """

    nbins = 100

    median = np.median(data)
    width = np.percentile(data, 80) - np.percentile(data, 20)

    good_data = data[(data > (median - 4 * width)) & (data < (median + 4 * width))]

    hist, bins = np.histogram(good_data, bins=101)  # np.linspace(1,5,101)
    bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2

    # fit gaussians to that
    # result = fit_unbinned(gauss, hist, [median, width/2] )
    # print("unbinned: {}".format(result))
    p0 = get_gaussian_guess(hist, bin_centers)
    bounds = [
        (p0[0] * 0.5, p0[1] * 0.5, p0[2] * 0.2, 0, 1),
        (p0[0] * 1.5, p0[1] * 1.5, p0[2] * 5, np.inf, np.inf),
    ]
    result = fit_binned(
        xtalball, hist, bin_centers, [p0[0], p0[1], p0[2], 10, 1], bounds=bounds
    )
    # print("binned: {}".format(result))
    cut_lo = result[0] - cut_sigma * result[1]
    cut_hi = result[0] + cut_sigma * result[1]

    if plotFigure is not None:
        plt.figure(plotFigure.number)
        plt.plot(bin_centers, hist, ls="steps-mid", color="k", label="data")
        fit = xtalball(bin_centers, *result)
        plt.plot(bin_centers, fit, label="xtalball fit")
        plt.axvline(result[0], color="g", label="fit mean")
        plt.axvline(cut_lo, color="r", label=f"+/- {cut_sigma} sigma")
        plt.axvline(cut_hi, color="r")
        plt.legend()
        # plt.xlabel(params[i])

    return cut_lo, cut_hi


def find_pulser_properties(df, energy="trap_max"):
    from .calibration import get_most_prominent_peaks

    # print (df[energy])
    # exit()
    # find pulser by looking for which peak has a constant time between events
    # df should already be grouped by channel

    peak_energies, peak_e_err = get_most_prominent_peaks(df[energy], max_num_peaks=10)
    peak_e_err *= 3

    for e in peak_energies:
        e_cut = (df[energy] > e - peak_e_err) & (df[energy] < e + peak_e_err)
        df_peak = df[e_cut]
        # df_after_0 = df_peak.iloc[1:]
        time_since_last = df_peak.timestamp.values[1:] - df_peak.timestamp.values[:-1]

        tsl = time_since_last[
            (time_since_last >= 0)
            & (time_since_last < np.percentile(time_since_last, 99.9))
        ]
        last_ten = np.percentile(tsl, 97) - np.percentile(tsl, 90)
        first_ten = np.percentile(tsl, 10) - np.percentile(tsl, 3)
        # print("{:e}, {:e}".format(last_ten,first_ten))

        if last_ten > first_ten:
            # print("...no pulser?")
            continue
        else:
            # df["pulser_energy"] = e
            pulser_e = e
            period = stats.mode(tsl).mode[0]

            return pulser_e, peak_e_err, period, energy
    return None


def tag_pulsers(df, chan_info, window=250):
    chan = df.channel.unique()[0]
    df["isPulser"] = 0

    try:
        pi = chan_info.loc[chan]
    except KeyError:
        return df

    energy_name = pi.energy_name
    pulser_energy = pi.pulser_energy
    period = pi.pulser_period
    peak_e_err = pi.peak_e_err

    # pulser_energy, peak_e_err, period, energy_name = chan_info

    e_cut = (df[energy_name] < pulser_energy + peak_e_err) & (
        df[energy_name] > pulser_energy - peak_e_err
    )
    df_pulser = df[e_cut]

    time_since_last = np.zeros(len(df_pulser))
    time_since_last[1:] = (
        df_pulser.timestamp.values[1:] - df_pulser.timestamp.values[:-1]
    )

    # plt.figure()
    # plt.hist(time_since_last, bins=1000)
    # plt.show()

    mode_idxs = (time_since_last > period - window) & (
        time_since_last < period + window
    )

    pulser_events = np.count_nonzero(mode_idxs)
    # print("pulser events: {}".format(pulser_events))
    if pulser_events < 3:
        return df

    df_pulser = df_pulser[mode_idxs]

    ts = df_pulser.timestamp.values
    diff_zero = np.zeros(len(ts))
    diff_zero[1:] = np.around((ts[1:] - ts[:-1]) / period)
    diff_cum = np.cumsum(diff_zero)
    z = np.polyfit(diff_cum, ts, 1)
    p = np.poly1d(z)

    # plt.figure()
    # xp = np.linspace(0, diff_cum[-1])
    # plt.plot(xp,p(xp))
    # plt.scatter(diff_cum,ts)
    # plt.show()

    period = z[0]
    phase = z[1]

    mod = np.abs(df.timestamp - phase) % period

    # pulser_mod  =np.abs(df_pulser.timestamp - phase) %period
    # pulser_mod[ pulser_mod > 10*window] = period - pulser_mod[ pulser_mod > 10*window]
    # plt.hist(pulser_mod , bins="auto")
    # plt.show()
    period_cut = (mod < window) | ((period - mod) < window)

    # print("pulser events: {}".format(np.count_nonzero(e_cut & period_cut)))
    df.loc[e_cut & period_cut, "isPulser"] = 1

    return df
