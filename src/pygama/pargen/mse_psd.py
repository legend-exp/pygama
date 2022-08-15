"""
- get_avse_cut (does AvsE)
- get_ae_cut (does A/E)
"""

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from pygama.math.histogram import get_bin_centers
from pygama.math.peak_fitting import *


def get_avse_cut(e_cal, current, plotFigure=None):
    # DEP range and nearby BG for BG subtraction
    dep_idxs = (e_cal > 1585) & (
        e_cal < 1595
    )  # (asymmetric around 1592 to account for CT)
    bg_idxs = (e_cal > 1560) & (e_cal < 1570)

    # SEP range and nearby BG for BG subtraction
    sep_idxs = (e_cal > 2090) & (e_cal < 2115)
    bg_sep_idxs = (e_cal > 2060) & (e_cal < 2080)

    # Bin the data in 2D, 25 keV bins
    # xedges = np.arange(200,2300,25)
    # e_cent = get_bin_centers(xedges)
    # y_max = np.zeros_like(e_cent)
    # plt.ion()
    # plt.figure()

    compton_shoulder = 2614 * (1 - 1.0 / (1 + 2 * 2614 / 511))

    # peaks = [238,510,583,727,860,1078,1512,1592,1806,compton_shoulder, 2614]
    # peaks = np.array(peaks)
    # e_cent = peaks
    # y_max = np.zeros_like(e_cent)

    xedges = np.arange(200, 2300, 25)
    e_cent = get_bin_centers(xedges)
    y_max = np.zeros_like(e_cent)

    # plt.ion()
    # plt.figure()
    for i, peak in enumerate(e_cent):
        plt.clf()
        # e_bin_idxs = (e_cal > peak-10) & (e_cal < peak+10 )
        e_bin_idxs = (e_cal > xedges[i]) & (e_cal < xedges[i + 1])
        a_ebin = current[e_bin_idxs]
        a_5 = np.percentile(a_ebin, 25)
        a_95 = np.percentile(a_ebin, 99)

        h, a_bins = np.histogram(a_ebin, bins=np.linspace(a_5, a_95, 500))

        a_bins_cent = get_bin_centers(a_bins)
        a_mode = a_bins_cent[np.argmax(h)]
        y_max[i] = a_mode

        p0 = get_gaussian_guess(h, a_bins_cent)
        fit_idxs = a_bins_cent > p0[0] - 5 * p0[1]
        p = fit_binned(gauss, h[fit_idxs], a_bins_cent[fit_idxs], p0)
        y_max[i] = p[0]

        # plt.plot(a_bins_cent,h,ls="steps")
        # plt.axvline(a_mode, c="r")
        # plt.title("Energy: {} keV".format(e_cent[i]))
        #
        # fit = gauss(a_bins_cent[fit_idxs], *p)
        # plt.plot(a_bins_cent[fit_idxs], fit, c="g")

        # guess = gauss(a_bins_cent[fit_idxs], *p0)
        # plt.plot(a_bins_cent[fit_idxs], guess, c="r")

        # inp = input("q to quit")
        # if inp == "q": exit()

    # quadratic fit
    # first fit a line:
    p_lin = np.polyfit(e_cent, y_max, 1)
    # find residuals:
    resid = y_max - np.poly1d(p_lin)(e_cent)
    resid_std = np.std(resid)
    # a really big residual is a pulser peak:
    puls_idx = resid > resid_std

    e_cent_cut = e_cent[~puls_idx]
    y_max_cut = y_max[~puls_idx]

    avse_quad = np.polyfit(e_cent_cut, y_max_cut, 2)
    avse_lin = np.polyfit(e_cent_cut, y_max_cut, 1)

    a_adjusted = current - np.poly1d(avse_quad)(e_cal)

    # Look at DEP, bg subtract the AvsE spectrum
    h_dep, bins = np.histogram(a_adjusted[dep_idxs], bins=5000)
    h_bg, bins = np.histogram(a_adjusted[bg_idxs], bins=bins)
    bin_centers = get_bin_centers(bins)
    h_bgs = h_dep - h_bg
    # fit AvsE peak to gaussian to get the 90% cut
    p0 = get_gaussian_guess(h_bgs, bin_centers)
    p = fit_binned(gauss, h_bgs, bin_centers, p0)
    fit = gauss(bin_centers, *p)

    ae_mean, ae_std = p[0], p[1]
    ae_cut = p[0] - 1.28 * p[1]  # cuts at 10% of CDF

    avse2, avse1, avse0 = avse_quad[:]
    avse_cut = ae_cut
    avse_mean = ae_mean
    avse_std = ae_std

    # plt.figure()
    # x = np.linspace(0,2700,5000)
    # plt.scatter(e_cent_cut, y_max_cut, color="k", s=10)
    # plt.scatter(e_cent[puls_idx], y_max[puls_idx], color="r", s=10)
    # plt.plot(x, np.poly1d(avse_quad)(x))
    # plt.plot(x, np.poly1d(avse_lin)(x))
    #
    # plt.figure()
    # xedges = np.arange(1000,2700,25)
    # aa_5 = np.percentile(a_adjusted,5)
    # aa_95 = np.percentile(a_adjusted,99)
    # yedges = np.linspace(aa_5, aa_95,1000)
    # H, xedges, yedges = np.histogram2d(e_cal, a_adjusted, bins=( xedges, yedges))
    # plt.imshow(H.T, interpolation='nearest', origin='low', aspect="auto",
    #             extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="OrRd", norm=LogNorm())
    # plt.scatter(e_cent_cut, y_max_cut - np.poly1d(avse_quad)(e_cent_cut) , color="k", s=10)
    # plt.axhline(ae_cut, color="k", ls="--")
    #
    # inp = input("q to quit")
    # if inp == "q": exit()

    if plotFigure is not None:
        ####
        # Plot A/E distributions
        ###
        plt.figure(plotFigure.number)
        plt.clf()
        grid = gs.GridSpec(2, 2)

        ax_dep = plt.subplot(grid[0, 0])
        ax_sep = plt.subplot(grid[1, 0])
        ax_ae = plt.subplot(grid[:, 1])

        # adjust a over e for mean
        a_over_e = a_adjusted
        ae_cut_mod = ae_cut

        h_dep, bins = np.histogram(
            a_over_e[dep_idxs], bins=np.linspace(-12 * ae_std, 8 * ae_std, 100)
        )
        h_bg, bins = np.histogram(a_over_e[bg_idxs], bins=bins)
        bin_centers = bins[:-1] + 0.5 * (bins[1] - bins[0])
        h_bgs = h_dep - h_bg

        h_sep, bins = np.histogram(a_over_e[sep_idxs], bins=bins)
        h_sepbg, bins = np.histogram(a_over_e[bg_sep_idxs], bins=bins)
        h_bgs_sep = h_sep - h_sepbg

        # ax_ae.plot(bin_centers,h_bgs / np.sum(h_bgs),  ls="steps-mid", color = "b", label = "DEP (BG subtracted)")
        # ax_ae.plot(bin_centers, h_bgs_sep/ np.sum(h_bgs), ls="steps-mid", color = "g", label = "SEP (BG subtracted)")

        ax_ae.plot(
            bin_centers, h_bgs, ls="steps-mid", color="b", label="DEP (BG subtracted)"
        )
        ax_ae.plot(
            bin_centers,
            h_bgs_sep,
            ls="steps-mid",
            color="g",
            label="SEP (BG subtracted)",
        )
        ax_ae.axvline(ae_cut_mod, color="r", ls=":")
        ax_ae.set_xlim(-12 * ae_std, 8 * ae_std)
        ax_ae.legend(loc=2)

        ax_ae.set_xlabel("A/E value [arb]")

        ###
        # Plot SEP/DEP before/after cut
        ##
        ae_cut_idxs = a_over_e > ae_cut_mod
        e_cal_aepass = e_cal[ae_cut_idxs]

        pad = 50
        bin_size = 0.2  # keV
        bins = np.arange(1592 - pad, 1592 + pad + bin_size, bin_size)

        ax_dep.hist(
            e_cal[(e_cal > 1592 - pad) & (e_cal < 1592 + pad)],
            histtype="step",
            color="k",
            label="DEP",
            bins=bins,
        )
        ax_dep.hist(
            e_cal_aepass[(e_cal_aepass > 1592 - pad) & (e_cal_aepass < 1592 + pad)],
            histtype="step",
            color="b",
            label="After Cut",
            bins=bins,
        )
        ax_dep.legend(loc=2)
        ax_dep.set_xlabel("Energy [keV]")

        bins = np.arange(2103 - pad, 2103 + pad + bin_size, bin_size)
        ax_sep.hist(
            e_cal[(e_cal > 2103 - pad) & (e_cal < 2103 + pad)],
            histtype="step",
            color="k",
            label="SEP",
            bins=bins,
        )
        ax_sep.hist(
            e_cal_aepass[(e_cal_aepass > 2103 - pad) & (e_cal_aepass < 2103 + pad)],
            histtype="step",
            color="b",
            label="After Cut",
            bins=bins,
        )
        ax_sep.legend(loc=2)
        ax_sep.set_xlabel("Energy [keV]")

    return avse2, avse1, avse0, avse_cut, avse_mean, avse_std


def get_ae_cut(e_cal, current, plotFigure=None):
    # try to get a rough A/E cut

    # DEP range and nearby BG for BG subtraction
    dep_idxs = (e_cal > 1585) & (
        e_cal < 1595
    )  # (asymmetric around 1592 to account for CT)
    bg_idxs = (e_cal > 1560) & (e_cal < 1570)

    # SEP range and nearby BG for BG subtraction
    sep_idxs = (e_cal > 2090) & (e_cal < 2115)
    bg_sep_idxs = (e_cal > 2060) & (e_cal < 2080)

    a_over_e = current / e_cal

    # # peaks = [2381]
    # peaks = [1512, 1592,1620,1806,2381]
    # ae_cents = np.zeros((len(peaks)))

    h_dep, bins = np.histogram(a_over_e[dep_idxs], bins=500)
    h_bg, bins = np.histogram(a_over_e[bg_idxs], bins=bins)
    bin_centers = get_bin_centers(bins)
    h_bgs = h_dep - h_bg

    p0 = get_gaussian_guess(h_bgs, bin_centers)
    p = fit_binned(gauss, h_bgs, bin_centers, p0)
    fit = gauss(bin_centers, *p)

    ae_mean, ae_std = p[0], p[1]
    ae_cut = p[0] - 1.28 * p[1]  # cuts at 10% of CDF

    if plotFigure is not None:
        ####
        # Plot A/E distributions
        ###
        plt.figure(plotFigure.number)
        plt.clf()
        grid = gs.GridSpec(2, 2)

        ax_dep = plt.subplot(grid[0, 0])
        ax_sep = plt.subplot(grid[1, 0])
        ax_ae = plt.subplot(grid[:, 1])

        # adjust a over e for mean

        a_over_e = (a_over_e - ae_mean) / ae_std
        ae_cut_mod = (ae_cut - ae_mean) / ae_std

        h_dep, bins = np.histogram(a_over_e[dep_idxs], bins=np.linspace(-8, 6, 50))
        h_bg, bins = np.histogram(a_over_e[bg_idxs], bins=bins)
        bin_centers = bins[:-1] + 0.5 * (bins[1] - bins[0])
        h_bgs = h_dep - h_bg

        h_sep, bins = np.histogram(a_over_e[sep_idxs], bins=bins)
        h_sepbg, bins = np.histogram(a_over_e[bg_sep_idxs], bins=bins)
        h_bgs_sep = h_sep - h_sepbg

        ax_ae.plot(
            bin_centers,
            h_bgs / np.sum(h_bgs),
            ls="steps-mid",
            color="b",
            label="DEP (BG subtracted)",
        )
        ax_ae.plot(
            bin_centers,
            h_bgs_sep / np.sum(h_bgs),
            ls="steps-mid",
            color="g",
            label="SEP (BG subtracted)",
        )
        # plt.plot(bin_centers, fit, color="g")
        ax_ae.axvline(ae_cut_mod, color="r", ls=":")
        ax_ae.set_xlim(-8, 5)
        ax_ae.legend(loc=2)

        ax_ae.set_xlabel("A/E value [arb]")

        ###
        # Plot SEP/DEP before/after cut
        ##
        ae_cut_idxs = a_over_e > ae_cut_mod
        e_cal_aepass = e_cal[ae_cut_idxs]

        pad = 50
        bins = np.linspace(1592 - pad, 1592 + pad, 2 * pad + 1)

        ax_dep.hist(
            e_cal[(e_cal > 1592 - pad) & (e_cal < 1592 + pad)],
            histtype="step",
            color="k",
            label="DEP",
            bins=bins,
        )
        ax_dep.hist(
            e_cal_aepass[(e_cal_aepass > 1592 - pad) & (e_cal_aepass < 1592 + pad)],
            histtype="step",
            color="b",
            label="After Cut",
            bins=bins,
        )
        ax_dep.legend(loc=2)
        ax_dep.set_xlabel("Energy [keV]")

        bins = np.linspace(2103 - pad, 2103 + pad, 2 * pad + 1)
        ax_sep.hist(
            e_cal[(e_cal > 2103 - pad) & (e_cal < 2103 + pad)],
            histtype="step",
            color="k",
            label="SEP",
            bins=bins,
        )
        ax_sep.hist(
            e_cal_aepass[(e_cal_aepass > 2103 - pad) & (e_cal_aepass < 2103 + pad)],
            histtype="step",
            color="b",
            label="After Cut",
            bins=bins,
        )
        ax_sep.legend(loc=2)
        ax_sep.set_xlabel("Energy [keV]")

    return ae_cut, ae_mean, ae_std
