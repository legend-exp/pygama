"""
This module provides functions for correcting the a/e energy dependence, determining the cut level and calculating survival fractions.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib

import matplotlib as mpl

mpl.use("agg")
import matplotlib.cm as cmx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit, cost, util
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy.stats import chi2

import pygama.lgdo.lh5_store as lh5
import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.ecal_th as thc
import pygama.pargen.energy_cal as pgc

log = logging.getLogger(__name__)


def load_aoe(
    files: list,
    lh5_path: str,
    cal_dict: dict,
    energy_param: str,
    cal_energy_param: str,
    cut_field: str = "Cal_cuts",
) -> tuple(np.array, np.array, np.array, np.array):

    """
    Loads in the A/E parameters needed and applies calibration constants to energy
    """

    sto = lh5.LH5Store()

    params = ["A_max", "tp_0_est", "tp_99", "dt_eff", energy_param, cal_energy_param]

    table = sto.read_object(lh5_path, files)[0]
    df = table.eval(cal_dict).get_dataframe()

    param_dict = {}
    for param in params:
        # add cuts in here
        if param in df:
            param_dict[param] = df[param].to_numpy()
        else:
            param_dict.update(lh5.load_nda(files, [param], lh5_path))
    if cut_field in df.keys():
        for entry in param_dict:
            param_dict[entry] = param_dict[entry][df[cut_field].to_numpy()]

    aoe = np.divide(param_dict["A_max"], param_dict[energy_param])
    return aoe, param_dict[cal_energy_param], param_dict["dt_eff"]


def PDF_AoE(
    x: np.array,
    lambda_s: float,
    lambda_b: float,
    mu: float,
    sigma: float,
    tau: float,
    lower_range: float = np.inf,
    upper_range: float = np.inf,
    components: bool = False,
) -> tuple(float, np.array):
    """
    PDF for A/E consists of a gaussian signal with gaussian tail background
    """
    try:
        sig = lambda_s * pgf.gauss_norm(x, mu, sigma)
        bkg = lambda_b * pgf.gauss_tail_norm(
            x, mu, sigma, tau, lower_range, upper_range
        )
    except:
        sig = np.full_like(x, np.nan)
        bkg = np.full_like(x, np.nan)

    if components == False:
        pdf = sig + bkg
        return lambda_s + lambda_b, pdf
    else:
        return lambda_s + lambda_b, sig, bkg


def unbinned_aoe_fit(
    aoe: np.array, display: int = 0, verbose: bool = False
) -> tuple(np.array, np.array):

    """
    Fitting function for A/E, first fits just a gaussian before using the full pdf to fit
    if fails will return NaN values
    """

    hist, bins, var = pgh.get_hist(aoe, bins=500)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    pars, cov = pgf.gauss_mode_max(hist, bins)
    mu = bin_centers[np.argmax(hist)]
    _, sigma, _ = pgh.get_gaussian_guess(hist, bins)
    ls_guess = 2 * np.sum(hist[(bin_centers > mu) & (bin_centers < (mu + 2.5 * sigma))])
    c1_min = mu - 2 * sigma
    c1_max = mu + 5 * sigma

    # Initial fit just using Gaussian
    c1 = cost.UnbinnedNLL(aoe[(aoe < c1_max) & (aoe > c1_min)], pgf.gauss_pdf)
    m1 = Minuit(c1, mu, sigma, ls_guess)
    m1.limits = [
        (mu * 0.8, mu * 1.2),
        (0.8 * sigma, sigma * 1.2),
        (0, len(aoe[(aoe < c1_max) & (aoe > c1_min)])),
    ]
    m1.migrad()
    mu, sigma, ls_guess = m1.values
    if verbose:
        print(m1)

    # Range to fit over, below this tail behaviour more exponential, few events above
    fmin = mu - 15 * sigma
    fmax = mu + 5 * sigma

    bg_guess = len(aoe[(aoe < fmax) & (aoe > fmin)]) - ls_guess
    x0 = [ls_guess, bg_guess, mu, sigma, 0.01, fmin, fmax, 0]
    if verbose:
        print(x0)

    # Full fit using gaussian signal with gaussian tail background
    c = cost.ExtendedUnbinnedNLL(aoe[(aoe < fmax) & (aoe > fmin)], PDF_AoE)
    m = Minuit(c, *x0)
    m.fixed[5:] = True
    m.simplex().migrad()
    m.hesse()
    if verbose:
        print(m)

    valid = m.valid & m.accurate
    if valid == False:
        return np.full_like(x0, np.nan), np.full_like(x0, np.nan)

    if display > 1:
        plt.figure()
        xs = np.linspace(fmin, fmax, 1000)
        counts, bins, bars = plt.hist(
            aoe[(aoe < fmax) & (aoe > fmin)], bins=400, histtype="step", label="Data"
        )
        dx = np.diff(bins)
        plt.plot(xs, PDF_AoE(xs, *m.values)[1] * dx[0], label="Full fit")
        n_events, sig, bkg = PDF_AoE(xs, *m.values[:-1], True)
        plt.plot(xs, sig * dx[0], label="Signal")
        plt.plot(xs, bkg * dx[0], label="Background")
        plt.plot(xs, pgf.gauss_pdf(xs, *m1.values) * dx[0], label="Initial Gaussian")
        plt.legend(loc="upper left")
        plt.show()

        plt.figure()
        bin_centers = (bins[1:] + bins[:-1]) / 2
        res = (PDF_AoE(bin_centers, *m.values)[1] * dx[0]) - counts
        plt.plot(
            bin_centers,
            [re / count if count != 0 else re for re, count in zip(res, counts)],
            label="Normalised Residuals",
        )
        plt.legend(loc="upper left")
        plt.show()
        return m.values, m.errors

    else:
        return m.values, m.errors


def pol1(x: np.array, a: float, b: float) -> np.array:
    """Basic Polynomial for fitting A/E centroid against energy"""
    return a * x + b


def sigma_fit(x: np.array, a: float, b: float) -> np.array:
    """Function definition for fitting A/E sigma against energy"""
    return np.sqrt(a + (b / x) ** 2)


def AoEcorrection(
    energy: np.array, aoe: np.array, eres: list, plot_dict: dict = {}, display: int = 0
) -> tuple(np.array, np.array):

    """
    Calculates the corrections needed for the energy dependence of the A/E.
    Does this by fitting the compton continuum in slices and then applies fits to the centroid and variance.
    """

    comptBands_width = 20
    comptBands = np.array(
        [
            940,
            960,
            980,
            1000,
            1020,
            1040,
            1130,
            1150,
            1170,
            1190,
            1210,
            1250,
            1270,
            1290,
            1310,
            1330,
            1370,
            1390,
            1420,
            1520,
            1540,
            1650,
            1700,
            1780,
            1810,
            1850,
            1870,
            1890,
            1910,
            1930,
            1950,
            1970,
            1990,
            2010,
            2030,
            2050,
            2150,
            2170,
            2190,
            2210,
            2230,
            2250,
            2270,
            2290,
            2310,
            2330,
            2350,
        ]
    )
    results_dict = {}
    comptBands = comptBands[::-1]  # Flip so color gets darker when plotting
    # peaks = np.array([1080,1094,1459,1512, 1552, 1592,1620, 1650, 1670,1830,2105])
    compt_aoe = np.zeros(len(comptBands))
    aoe_sigmas = np.zeros(len(comptBands))
    compt_aoe_err = np.zeros(len(comptBands))
    aoe_sigmas_err = np.zeros(len(comptBands))
    ratio = np.zeros(len(comptBands))
    ratio_err = np.zeros(len(comptBands))

    copper = cm = plt.get_cmap("copper")
    cNorm = mcolors.Normalize(vmin=0, vmax=len(comptBands))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=copper)

    if display > 0:
        fits_fig = plt.figure()

    # Fit each compton band
    for i, band in enumerate(comptBands):
        aoe_tmp = aoe[
            (energy > band) & (energy < band + comptBands_width) & (aoe > 0)
        ]  # [:20000]
        pars, errs = unbinned_aoe_fit(aoe_tmp, display=display)
        compt_aoe[i] = pars[2]
        aoe_sigmas[i] = pars[3]
        compt_aoe_err[i] = errs[2]
        aoe_sigmas_err[i] = errs[3]
        ratio[i] = pars[0] / pars[1]
        ratio_err[i] = ratio[i] * np.sqrt(
            (errs[0] / pars[0]) ** 2 + (errs[1] / pars[1]) ** 2
        )

        if display > 0:
            if np.isnan(errs[2]) | np.isnan(errs[3]) | (errs[2] == 0) | (errs[3] == 0):
                pass
            else:
                xs = np.arange(
                    pars[2] - 4 * pars[3], pars[2] + 3 * pars[3], pars[3] / 10
                )
                colorVal = scalarMap.to_rgba(i)
                plt.plot(xs, PDF_AoE(xs, *pars)[1], color=colorVal)

    if display > 0:
        plt.xlabel("A/E")
        plt.ylabel("Expected Counts")
        plt.title("Compton Band Fits")
        cbar = plt.colorbar(
            cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap("copper_r")),
            orientation="horizontal",
            label="Compton Band Energy",
            ticks=[0, 16, 32, len(comptBands)],
        )  # cax=ax,
        cbar.ax.set_xticklabels(
            [
                comptBands[::-1][0],
                comptBands[::-1][16],
                comptBands[::-1][32],
                comptBands[::-1][-1],
            ]
        )
        plot_dict["band_fits"] = fits_fig
        if display > 1:
            plt.show()
        else:
            plt.close()

    ids = (
        np.isnan(compt_aoe_err)
        | np.isnan(aoe_sigmas_err)
        | (aoe_sigmas_err == 0)
        | (compt_aoe_err == 0)
    )
    results_dict["n_of_valid_fits"] = len(np.where(~ids)[0])
    # Fit mus against energy
    p0_mu = [-1e-06, 5e-01]
    c_mu = cost.LeastSquares(
        comptBands[~ids], compt_aoe[~ids], compt_aoe_err[~ids], pol1
    )
    c_mu.loss = "soft_l1"
    m_mu = Minuit(c_mu, *p0_mu)
    m_mu.simplex()
    m_mu.migrad()
    m_mu.hesse()

    pars = m_mu.values
    errs = m_mu.errors

    csqr_mu = np.sum(
        ((compt_aoe[~ids] - pol1(comptBands[~ids], *pars)) ** 2) / compt_aoe_err[~ids]
    )
    dof_mu = len(compt_aoe[~ids]) - len(pars)
    results_dict["p_val_mu"] = chi2.sf(csqr_mu, dof_mu)
    results_dict["csqr_mu"] = (csqr_mu, dof_mu)

    # Fit sigma against energy
    p0_sig = [np.nanpercentile(aoe_sigmas[~ids], 50) ** 2, 2]
    c_sig = cost.LeastSquares(
        comptBands[~ids], aoe_sigmas[~ids], aoe_sigmas_err[~ids], sigma_fit
    )
    c_sig.loss = "soft_l1"
    m_sig = Minuit(c_sig, *p0_sig)
    m_sig.simplex()
    m_sig.migrad()
    m_sig.hesse()

    sig_pars = m_sig.values
    sig_errs = m_sig.errors

    csqr_sig = np.sum(
        ((aoe_sigmas[~ids] - sigma_fit(comptBands[~ids], *sig_pars)) ** 2)
        / aoe_sigmas_err[~ids]
    )
    dof_sig = len(aoe_sigmas[~ids]) - len(sig_pars)
    results_dict["p_val_sig"] = chi2.sf(csqr_sig, dof_sig)
    results_dict["csqr_sig"] = (csqr_sig, dof_sig)

    model = pol1(comptBands, *pars)
    sig_model = sigma_fit(comptBands, *sig_pars)

    # Get DEP fit
    sigma = np.sqrt(eres[0] + 1592 * eres[1]) / 2.355
    n_sigma = 4
    peak = 1592
    emin = peak - n_sigma * sigma
    emax = peak + n_sigma * sigma
    dep_pars, dep_err = unbinned_aoe_fit(
        aoe[(energy > emin) & (energy < emax) & (aoe > 0)][:10000]
    )

    if display > 0:
        mean_fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.errorbar(
            comptBands[~ids] + 10,
            compt_aoe[~ids],
            yerr=compt_aoe_err[~ids],
            xerr=10,
            label="data",
            linestyle=" ",
        )
        ax1.plot(comptBands[~ids] + 10, model[~ids], label="linear model")
        ax1.errorbar(
            1592,
            dep_pars[2],
            xerr=n_sigma * sigma,
            yerr=dep_err[2],
            label="DEP",
            color="green",
            linestyle=" ",
        )

        ax1.legend(title="A/E mu energy dependence", frameon=False)

        ax1.set_ylabel("raw A/E (a.u.)", ha="right", y=1)
        ax2.scatter(
            comptBands[~ids] + 10,
            100 * (compt_aoe[~ids] - model[~ids]) / compt_aoe_err[~ids],
            lw=1,
            c="b",
        )
        ax2.scatter(
            1592, 100 * (dep_pars[2] - pol1(1592, *pars)) / dep_err[2], lw=1, c="g"
        )
        ax2.set_ylabel("Residuals %", ha="right", y=1)
        ax2.set_xlabel("Energy (keV)", ha="right", x=1)
        plt.tight_layout()
        plot_dict["mean_fit"] = mean_fig
        if display > 1:
            plt.show()
        else:
            plt.close()

    if display > 0:
        sig_fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.errorbar(
            comptBands[~ids] + 10,
            aoe_sigmas[~ids],
            yerr=aoe_sigmas_err[~ids],
            xerr=10,
            label="data",
            linestyle=" ",
        )
        ax1.plot(
            comptBands[~ids],
            sig_model[~ids],
            label=f"sqrt model: sqrt({sig_pars[0]:1.4f}+({sig_pars[1]:1.1f}/E)^2)",
        )  # {sig_pars[2]:1.1f}
        ax1.errorbar(
            1592,
            dep_pars[3],
            xerr=n_sigma * sigma,
            yerr=dep_err[3],
            label="DEP",
            color="green",
        )
        ax1.set_ylabel("A/E stdev (a.u.)", ha="right", y=1)
        ax1.legend(title="A/E stdev energy dependence", frameon=False)
        ax2.scatter(
            comptBands[~ids] + 10,
            100 * (aoe_sigmas[~ids] - sig_model[~ids]) / aoe_sigmas_err[~ids],
            lw=1,
            c="b",
        )
        ax2.scatter(
            1592,
            100 * (dep_pars[3] - sigma_fit(1592, *sig_pars)) / dep_err[3],
            lw=1,
            c="g",
        )
        ax2.set_ylabel("Residuals", ha="right", y=1)
        ax2.set_xlabel("Energy (keV)", ha="right", x=1)
        plt.tight_layout()
        plot_dict["sigma_fit"] = sig_fig
        if display > 1:
            plt.show()
        else:
            plt.close()
        return pars, sig_pars, results_dict, plot_dict
    else:
        return pars, sig_pars, results_dict


def plot_compt_bands_overlayed(
    aoe: np.array, energy: np.array, eranges: list[tuple], aoe_range: list[float] = None
) -> None:

    """
    Function to plot various compton bands to check energy dependence and corrections
    """

    for erange in eranges:
        hist, bins, var = pgh.get_hist(
            aoe[(energy > erange - 10) & (energy < erange + 10) & (~np.isnan(aoe))],
            bins=500,
        )
        bin_cs = (bins[1:] + bins[:-1]) / 2
        mu = bin_cs[np.argmax(hist)]
        if aoe_range is None:
            aoe_range = [mu * 0.97, mu * 1.02]
        idxs = (
            (energy > erange - 10)
            & (energy < erange + 10)
            & (aoe > aoe_range[0])
            & (aoe < aoe_range[1])
        )
        plt.hist(aoe[idxs], bins=50, histtype="step", label=f"{erange-10}-{erange+10}")


def plot_dt_dep(
    aoe: np.array, energy: np.array, dt: np.array, erange: list[tuple], title: str
) -> None:

    """
    Function to produce 2d histograms of A/E against drift time to check dependencies
    """

    hist, bins, var = pgh.get_hist(
        aoe[(energy > erange[0]) & (energy < erange[1]) & (~np.isnan(aoe))], bins=500
    )
    bin_cs = (bins[1:] + bins[:-1]) / 2
    mu = bin_cs[np.argmax(hist)]
    aoe_range = [mu * 0.9, mu * 1.1]

    idxs = (
        (energy > erange[0])
        & (energy < erange[1])
        & (aoe > aoe_range[0])
        & (aoe < aoe_range[1])
        & (dt < 2000)
    )

    plt.hist2d(aoe[idxs], dt[idxs], bins=[200, 100], norm=LogNorm())
    plt.ylabel("Drift Time (ns)")
    plt.xlabel("A/E")
    plt.title(title)


def energy_guess(hist, bins, var, func_i, peak, eres_pars, fit_range):
    """
    Simple guess for peak fitting
    """
    if func_i == pgf.extended_radford_pdf:
        bin_cs = (bins[1:] + bins[:-1]) / 2
        sigma = thc.fwhm_slope(peak, *eres_pars) / 2.355
        i_0 = np.nanargmax(hist)
        mu = peak
        height = hist[i_0]
        bg0 = np.mean(hist[-10:])
        step = np.mean(hist[:10]) - bg0
        htail = 1.0 / 5
        tau = 0.5 * sigma

        hstep = step / (bg0 + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((3 * sigma) // dx)
        nsig_guess = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
        nbkg_guess = np.sum(hist) - nsig_guess
        if nbkg_guess < 0:
            nbkg_guess = 0
        if nsig_guess < 0:
            nsig_guess = 0
        parguess = [
            nsig_guess,
            mu,
            sigma,
            htail,
            tau,
            nbkg_guess,
            hstep,
            fit_range[0],
            fit_range[1],
            0,
        ]  #
        return parguess

    elif func_i == pgf.extended_gauss_step_pdf:
        mu = peak
        sigma = thc.fwhm_slope(peak, *eres_pars) / 2.355
        i_0 = np.argmax(hist)
        bg = np.mean(hist[-10:])
        step = bg - np.mean(hist[:10])
        hstep = step / (bg + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((3 * sigma) // dx)
        nsig_guess = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
        nbkg_guess = np.sum(hist) - nsig_guess
        if nbkg_guess < 0:
            nbkg_guess = 0
        if nsig_guess < 0:
            nsig_guess = 0
        return [nsig_guess, mu, sigma, nbkg_guess, hstep, fit_range[0], fit_range[1], 0]


def unbinned_energy_fit(
    energy: np.array,
    peak: float,
    eres_pars: list = None,
    simplex=False,
    guess=None,
    verbose: bool = False,
) -> tuple(np.array, np.array):

    """
    Fitting function for energy peaks used to calculate survival fractions
    """
    hist, bins, var = pgh.get_hist(
        energy, dx=0.5, range=(np.nanmin(energy), np.nanmax(energy))
    )
    if guess is None:

        x0 = energy_guess(
            hist,
            bins,
            var,
            pgf.extended_gauss_step_pdf,
            peak,
            eres_pars,
            (np.nanmin(energy), np.nanmax(energy)),
        )
        c = cost.ExtendedUnbinnedNLL(energy, pgf.extended_gauss_step_pdf)
        m = Minuit(c, *x0)
        m.fixed[-3:] = True
        m.simplex().migrad()
        m.hesse()
        x0 = m.values[:3]
        x0 += [1 / 5, 0.2 * m.values[2]]
        x0 += m.values[3:]
        if verbose:
            print(m)
    else:
        x0 = guess
    if len(x0) == 0:
        return [np.nan], [np.nan]

    fixed, mask = pgc.get_hpge_E_fixed(pgf.extended_radford_pdf)
    if verbose:
        print(x0)
    c = cost.ExtendedUnbinnedNLL(energy, pgf.extended_radford_pdf)
    m = Minuit(c, *x0)
    if x0[0] <= 0:
        sig_lim = 0.2 * len(energy)
    else:
        sig_lim = 1.2 * x0[0]
    m.limits = [
        (0, sig_lim),
        (None, None),
        (None, None),
        (0, 1),
        (0, None),
        (0, len(energy) * 1.1),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
    ]
    for fix in fixed:
        m.fixed[fix] = True
    if simplex == True:
        m.simplex().migrad()
    else:
        m.migrad()

    m.hesse()
    if verbose:
        print(m)

    valid = m.valid

    if valid == True and not np.isnan(m.errors[:-3]).all():
        return m.values, m.errors
    else:
        x0 = energy_guess(
            hist,
            bins,
            var,
            pgf.extended_radford_pdf,
            peak,
            eres_pars,
            (np.nanmin(energy), np.nanmax(energy)),
        )
        c = cost.ExtendedUnbinnedNLL(energy, pgf.extended_radford_pdf)
        m = Minuit(c, *x0)
        if x0[0] <= 0:
            sig_lim = 0.2 * len(energy)
        else:
            sig_lim = 1.2 * x0[0]
        m.limits = [
            (0, sig_lim),
            (None, None),
            (None, None),
            (0, 1),
            (0, None),
            (0, len(energy) * 1.1),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
        ]
        for fix in fixed:
            m.fixed[fix] = True
        m.simplex().migrad()
        m.hesse()
        valid = m.valid
        if verbose:
            print(m)
        if valid == True:
            return m.values, m.errors
        else:
            return np.full_like(x0, np.nan), np.full_like(x0, np.nan)


def get_peak_label(peak: float) -> str:
    if peak == 2039:
        return "CC @"
    elif peak == 1592.5:
        return "Tl DEP @"
    elif peak == 1620.5:
        return "Bi FEP @"
    elif peak == 2103.53:
        return "Tl SEP @"
    elif peak == 2614.5:
        return "Tl FEP @"


def get_survival_fraction(
    energy,
    aoe,
    cut_val,
    peak,
    eres_pars,
    high_cut=None,
    guess_pars_cut=None,
    guess_pars_surv=None,
):
    if high_cut is not None:
        idxs = (aoe > cut_val) & (aoe < high_cut)
    else:
        idxs = aoe > cut_val
    cut_pars, ct_errs = unbinned_energy_fit(
        energy[~idxs], peak, eres_pars, guess=guess_pars_cut, simplex=True
    )
    surv_pars, surv_errs = unbinned_energy_fit(
        energy[idxs], peak, eres_pars, guess=guess_pars_surv, simplex=True
    )

    ct_n = cut_pars[0]
    ct_err = ct_errs[0]
    surv_n = surv_pars[0]
    surv_err = surv_errs[0]

    pc_n = ct_n + surv_n
    pc_err = np.sqrt(surv_err**2 + ct_err**2)

    sf = (surv_n / pc_n) * 100
    err = sf * np.sqrt((pc_err / pc_n) ** 2 + (surv_err / surv_n) ** 2)
    return sf, err, cut_pars, surv_pars


def get_aoe_cut_fit(
    energy: np.array,
    aoe: np.array,
    peak: float,
    ranges: tuple(int, int),
    dep_acc: float,
    eres_pars: list,
    display: int = 1,
) -> float:

    """
    Determines A/E cut by sweeping through values and for each one fitting the DEP to determine how many events survive.
    Then interpolates to get cut value at desired DEP survival fraction (typically 90%)
    """

    min_range, max_range = ranges

    peak_energy = energy[(energy > peak - min_range) & (energy < peak + max_range)][
        :20000
    ]
    peak_aoe = aoe[(energy > peak - min_range) & (energy < peak + max_range)][:20000]

    cut_vals = np.arange(-8, 0, 0.2)
    pars, errors = unbinned_energy_fit(peak_energy, peak, eres_pars, simplex=True)
    if np.isnan(pars[0]):
        print("first fit failed")
        pars, errors = unbinned_energy_fit(peak_energy, peak, eres_pars, simplex=False)
    pc_n = pars[0]
    pc_err = errors[0]
    sfs = []
    sf_errs = []
    for cut_val in cut_vals:
        sf, err, cut_pars, surv_pars = get_survival_fraction(
            peak_energy,
            peak_aoe,
            cut_val,
            peak,
            eres_pars,
            guess_pars_cut=None,
            guess_pars_surv=None,
        )
        sfs.append(sf)
        sf_errs.append(err)

    # return cut_vals, sfs, sf_errs
    ids = (
        (sf_errs < (1.5 * np.nanpercentile(sf_errs, 70)))
        & (~np.isnan(sf_errs))
        & (np.array(sfs) < 95)
    )
    fit = np.polynomial.polynomial.polyfit(
        cut_vals[ids], np.array(sfs)[ids], w=1 / np.array(sf_errs)[ids], deg=6
    )

    xs = np.arange(np.nanmin(cut_vals[ids]), np.nanmax(cut_vals[ids]), 0.01)
    p = np.polynomial.polynomial.polyval(xs, fit)
    cut_val = xs[np.argmin(np.abs(p - (100 * dep_acc)))]

    if display > 0:
        plt.figure()
        plt.errorbar(
            cut_vals[ids],
            np.array(sfs)[ids],
            yerr=np.array(sf_errs)[ids],
            linestyle=" ",
        )

        plt.plot(xs, p)
        plt.show()

    return cut_val


def get_sf(
    energy: np.array,
    aoe: np.array,
    peak: float,
    fit_width: tuple(int, int),
    aoe_cut_val: float,
    eres_pars: list,
    display: int = 0,
) -> tuple(np.array, np.array, np.array, float, float):

    """
    Calculates survival fraction for gamma lines using fitting method as in cut determination
    """

    # fwhm = np.sqrt(eres[0]+peak*eres[1])
    min_range = peak - fit_width[0]
    max_range = peak + fit_width[1]
    if peak == "1592.5":
        peak_energy = energy[(energy > min_range) & (energy < max_range)][:20000]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range)][:20000]
    else:
        peak_energy = energy[(energy > min_range) & (energy < max_range)][:50000]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range)][:50000]
    pars, errors = unbinned_energy_fit(peak_energy, peak, eres_pars, simplex=False)
    pc_n = pars[0]
    pc_err = errors[0]
    sfs = []
    sf_errs = []

    cut_vals = np.arange(-5, 5, 0.2)
    # cut_vals = np.append(cut_vals, aoe_cut_val)
    final_cut_vals = []
    for cut_val in cut_vals:
        try:

            sf, err, cut_pars, surv_pars = get_survival_fraction(
                peak_energy, peak_aoe, cut_val, peak, eres_pars
            )
            if np.isnan(cut_pars).all() == False and np.isnan(surv_pars).all() == False:
                guess_pars_cut = cut_pars
                guess_pars_surv = surv_pars
        except:
            sf = np.nan
            err = np.nan
        sfs.append(sf)
        sf_errs.append(err)
        final_cut_vals.append(cut_val)
    ids = (
        (sf_errs < (5 * np.nanpercentile(sf_errs, 50)))
        & (~np.isnan(sf_errs))
        & (np.array(sfs) < 100)
    )
    sf, sf_err, cut_pars, surv_pars = get_survival_fraction(
        peak_energy, peak_aoe, aoe_cut_val, peak, eres_pars
    )

    if display > 0:
        plt.figure()
        plt.errorbar(cut_vals, sfs, sf_errs)
        plt.show()

    return (
        np.array(final_cut_vals)[ids],
        np.array(sfs)[ids],
        np.array(sf_errs)[ids],
        sf,
        sf_err,
    )


def compton_sf(
    energy: np.array,
    aoe: np.array,
    cut: float,
    peak: float,
    eres: list[float, float],
    display: int = 1,
) -> tuple(float, np.array, list):

    """
    Determines survival fraction for compton continuum by basic counting
    """

    fwhm = np.sqrt(eres[0] + peak * eres[1])

    emin = peak - 2 * fwhm
    emax = peak + 2 * fwhm
    sfs = []
    aoe = aoe[(energy > emin) & (energy < emax)]
    cut_vals = np.arange(-5, 5, 0.1)
    for cut_val in cut_vals:
        sfs.append(100 * len(aoe[(aoe > cut_val)]) / len(aoe))
    sf = 100 * len(aoe[(aoe > cut)]) / len(aoe)
    return sf, cut_vals, sfs


def get_sf_no_sweep(
    energy: np.array,
    aoe: np.array,
    peak: float,
    fit_width: tuple(int, int),
    eres_pars: list,
    aoe_low_cut_val: float,
    aoe_high_cut_val: float = None,
    display: int = 1,
) -> tuple(float, float):

    """
    Calculates survival fraction for gamma line without sweeping through values
    """

    min_range = peak - fit_width[0]
    max_range = peak + fit_width[1]
    if peak == "1592.5":
        peak_energy = energy[(energy > min_range) & (energy < max_range)][:20000]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range)][:20000]
    else:
        peak_energy = energy[(energy > min_range) & (energy < max_range)][:50000]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range)][:50000]

    sf, sf_err, cut_pars, surv_pars = get_survival_fraction(
        peak_energy,
        peak_aoe,
        aoe_low_cut_val,
        peak,
        eres_pars,
        high_cut=aoe_high_cut_val,
    )
    return sf, sf_err


def compton_sf_no_sweep(
    energy: np.array,
    aoe: np.array,
    peak: float,
    eres: list[float, float],
    aoe_low_cut_val: float,
    aoe_high_cut_val: float = None,
    display: int = 1,
) -> float:

    """
    Calculates survival fraction for compton contiuum without sweeping through values
    """

    fwhm = np.sqrt(eres[0] + peak * eres[1])

    emin = peak - 2 * fwhm
    emax = peak + 2 * fwhm
    sfs = []
    aoe = aoe[(energy > emin) & (energy < emax)]
    cut_vals = np.arange(-5, 5, 0.1)
    if aoe_high_cut_val is None:
        sf = 100 * len(aoe[(aoe > aoe_low_cut_val)]) / len(aoe)
    else:
        sf = (
            100
            * len(aoe[(aoe > aoe_low_cut_val) & (aoe < aoe_high_cut_val)])
            / len(aoe)
        )
    return sf


def get_classifier(
    aoe: np.array,
    energy: np.array,
    mu_pars: list[float, float],
    sigma_pars: list[float, float],
) -> np.array:

    """
    Applies correction to A/E energy dependence
    """

    classifier = aoe / (mu_pars[0] * energy + mu_pars[1])
    classifier = (classifier - 1) / sigma_fit(energy, *sigma_pars)
    return classifier


def get_dt_guess(hist: np.array, bins: np.array, var: np.array) -> list:
    """
    Guess for fitting dt spectrum
    """

    mu, sigma, amp = pgh.get_gaussian_guess(hist, bins)
    i_0 = np.argmax(hist)
    bg = np.mean(hist[-10:])
    step = bg - np.mean(hist[:10])
    hstep = step / (bg + np.mean(hist[:10]))
    dx = np.diff(bins)[0]
    n_bins_range = int((3 * sigma) // dx)
    nsig_guess = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
    nbkg_guess = np.sum(hist) - nsig_guess
    return [
        nsig_guess * np.diff(bins)[0],
        mu,
        sigma,
        nbkg_guess * np.diff(bins)[0],
        hstep,
        np.inf,
        np.inf,
        0,
    ]


def apply_dtcorr(aoe: np.array, dt: np.array, alpha: float) -> np.array:
    """Aligns dt regions"""
    return aoe * (1 + alpha * dt)


def drift_time_correction(
    aoe: np.array,
    energy: np.array,
    dt: np.array,
    display: int = 0,
    plot_dict: dict = {},
) -> tuple(np.array, float):
    """
    Calculates the correction needed to align the two drift time regions for ICPC detectors
    """
    hist, bins, var = pgh.get_hist(aoe[(energy > 1582) & (energy < 1602)], bins=500)
    bin_cs = (bins[1:] + bins[:-1]) / 2
    mu = bin_cs[np.argmax(hist)]
    aoe_range = [mu * 0.9, mu * 1.1]

    idxs = (
        (energy > 1582) & (energy < 1602) & (aoe > aoe_range[0]) & (aoe < aoe_range[1])
    )

    mask = (
        (idxs)
        & (dt < np.nanpercentile(dt[idxs], 55))
        & (dt > np.nanpercentile(dt[idxs], 1))
    )

    hist, bins, var = pgh.get_hist(
        dt[mask], dx=10, range=(np.nanmin(dt[mask]), np.nanmax(dt[mask]))
    )

    gpars = get_dt_guess(hist, bins, var)
    dt_pars, dt_errs, dt_cov = pgf.fit_binned(
        pgf.gauss_step_pdf,
        hist,
        bins,
        guess=gpars,
        fixed=[-3, -2, -1],
        cost_func="Least Squares",
    )

    aoe_mask = (
        (idxs) & (dt > dt_pars[1] - 2 * dt_pars[2]) & (dt < dt_pars[1] + 2 * dt_pars[2])
    )
    aoe_tmp = aoe[aoe_mask]
    aoe_pars, aoe_errs = unbinned_aoe_fit(aoe_tmp, display=display)

    mask2 = (
        (idxs)
        & (dt > np.nanpercentile(dt[idxs], 50))
        & (dt < np.nanpercentile(dt[idxs], 99))
    )
    hist2, bins2, var2 = pgh.get_hist(
        dt[mask2], dx=10, range=(np.nanmin(dt[mask2]), np.nanmax(dt[mask2]))
    )
    gpars2 = get_dt_guess(hist2, bins2, var2)

    dt_pars2, dt_errs2, dt_cov2 = pgf.fit_binned(
        pgf.gauss_step_pdf,
        hist2,
        bins2,
        guess=gpars2,
        fixed=[-3, -2, -1],
        cost_func="Least Squares",
    )

    aoe_mask2 = (
        (idxs)
        & (dt > dt_pars2[1] - 2 * dt_pars2[2])
        & (dt < dt_pars2[1] + 2 * dt_pars2[2])
    )
    aoe_tmp2 = aoe[aoe_mask2]
    aoe_pars2, aoe_errs2 = unbinned_aoe_fit(aoe_tmp2, display=display)

    alpha = (aoe_pars[2] - aoe_pars2[2]) / (
        dt_pars2[1] * aoe_pars2[2] - dt_pars[1] * aoe_pars[2]
    )
    aoe_corrected = apply_dtcorr(aoe, dt, alpha)

    if display > 0:
        dt_fig = plt.figure()
        plt.subplot(3, 2, 1)
        plt.step(pgh.get_bin_centers(bins), hist, label="Data")
        plt.plot(
            pgh.get_bin_centers(bins),
            pgf.gauss_step_pdf(pgh.get_bin_centers(bins), *gpars),
            label="Guess",
        )
        plt.plot(
            pgh.get_bin_centers(bins),
            pgf.gauss_step_pdf(pgh.get_bin_centers(bins), *dt_pars),
            label="Fit",
        )
        plt.xlabel("Drift Time (ns)")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")

        plt.subplot(3, 2, 2)
        plt.step(pgh.get_bin_centers(bins2), hist2, label="Data")
        plt.plot(
            pgh.get_bin_centers(bins2),
            pgf.gauss_step_pdf(pgh.get_bin_centers(bins2), *gpars2),
            label="Guess",
        )
        plt.plot(
            pgh.get_bin_centers(bins2),
            pgf.gauss_step_pdf(pgh.get_bin_centers(bins2), *dt_pars2),
            label="Fit",
        )
        plt.xlabel("Drift Time (ns)")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")

        plt.subplot(3, 2, 3)
        xs = np.linspace(aoe_pars[-3], aoe_pars[-2], 1000)
        counts, aoe_bins, bars = plt.hist(
            aoe[(aoe < aoe_pars[-2]) & (aoe > aoe_pars[-3]) & aoe_mask],
            bins=400,
            histtype="step",
            label="Data",
        )
        dx = np.diff(aoe_bins)
        plt.plot(xs, PDF_AoE(xs, *aoe_pars)[1] * dx[0], label="Full fit")
        # plt.yscale('log')
        n_events, sig, bkg = PDF_AoE(xs, *aoe_pars[:-1], True)
        plt.plot(xs, sig * dx[0], label="Peak fit")
        plt.plot(xs, bkg * dx[0], label="Bkg fit")
        plt.legend(loc="upper left")
        plt.xlabel("A/E")
        plt.ylabel("Counts")

        plt.subplot(3, 2, 4)
        xs = np.linspace(aoe_pars2[-3], aoe_pars2[-2], 1000)
        counts, aoe_bins2, bars = plt.hist(
            aoe[(aoe < aoe_pars2[-2]) & (aoe > aoe_pars2[-3]) & aoe_mask2],
            bins=400,
            histtype="step",
            label="Data",
        )
        dx = np.diff(aoe_bins2)
        plt.plot(xs, PDF_AoE(xs, *aoe_pars2)[1] * dx[0], label="Full fit")
        # plt.yscale('log')
        n_events, sig, bkg = PDF_AoE(xs, *aoe_pars2[:-1], True)
        plt.plot(xs, sig * dx[0], label="Peak fit")
        plt.plot(xs, bkg * dx[0], label="Bkg fit")
        plt.legend(loc="upper left")
        plt.xlabel("A/E")
        plt.ylabel("Counts")

        plt.subplot(3, 2, 5)
        counts, bins, bars = plt.hist(
            aoe[idxs], bins=200, histtype="step", label="Uncorrected"
        )
        plt.hist(aoe_corrected[idxs], bins=bins, histtype="step", label="Corrected")
        plt.xlabel("A/E")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.xlim([aoe_pars[-3], aoe_pars[-2] * 1.01])
        plot_dict["dt_corr"] = dt_fig
        if display > 1:
            plt.show()
        else:
            plt.close()
        return aoe_corrected, alpha, plot_dict
    else:
        return aoe_corrected, alpha


def cal_aoe(
    files: list,
    lh5_path,
    cal_dict: dict,
    energy_param: str,
    cal_energy_param: str,
    eres_pars: list,
    cut_field: str = "is_valid_cal",
    dt_corr: bool = False,
    aoe_high_cut: int = 4,
    display: int = 0,
) -> tuple(dict, dict):
    """
    Main function for running the a/e correction and cut determination.
    """

    aoe_uncorr, energy, dt = load_aoe(
        files, lh5_path, cal_dict, energy_param, cal_energy_param, cut_field=cut_field
    )

    if dt_corr == True:
        aoe, alpha = drift_time_correction(aoe_uncorr, energy, dt)
    else:
        aoe = aoe_uncorr

    aoe_tmp = aoe[(energy > 1000) & (energy < 1300) & (aoe > 0)]  # [:20000]
    bulk_pars, bulk_errs = unbinned_aoe_fit(aoe_tmp, display=0)

    log.info("Starting A/E correction")
    mu_pars, sigma_pars, results_dict = AoEcorrection(energy, aoe, eres_pars)
    log.info("Finished A/E correction")

    classifier = get_classifier(aoe, energy, mu_pars, sigma_pars)

    cut = get_aoe_cut_fit(energy, classifier, 1592, (40, 20), 0.9, eres_pars, display=0)

    cal_dict.update(
        {
            "AoE_Corrected": {
                "expression": f"(((A_max/{energy_param})/(a*{cal_energy_param} +b))-1)",
                "parameters": {"a": mu_pars[0], "b": mu_pars[1]},
            }
        }
    )

    cal_dict.update(
        {
            "AoE_Classifier": {
                "expression": f"AoE_Corrected/(sqrt(c+(d/{cal_energy_param})**2)/(a*{cal_energy_param} +b))",
                "parameters": {
                    "a": mu_pars[0],
                    "b": mu_pars[1],
                    "c": sigma_pars[0],
                    "d": sigma_pars[1],
                },
            }
        }
    )

    cal_dict.update(
        {
            "AoE_Low_Cut": {
                "expression": "AoE_Classifier>a",
                "parameters": {"a": cut},
            }
        }
    )

    cal_dict.update(
        {
            "AoE_Double_Sided_Cut": {
                "expression": "(b>AoE_Classifier)&(AoE_Classifier>a)",
                "parameters": {"a": cut, "b": aoe_high_cut},
            }
        }
    )

    try:
        log.info("  Compute low side survival fractions: ")

        peaks_of_interest = [1592.5, 1620.5, 2039, 2103.53, 2614.50]
        sf = np.zeros(len(peaks_of_interest))
        sferr = np.zeros(len(peaks_of_interest))
        fit_widths = [(40, 25), (25, 40), (0, 0), (25, 40), (50, 50)]
        full_sfs = []
        full_sf_errs = []
        full_cut_vals = []

        for i, peak in enumerate(peaks_of_interest):
            if peak == 2039:
                sf[i], cut_vals, sfs = compton_sf(
                    energy, classifier, cut, peak, eres_pars
                )
                sferr[i] = 0

                full_cut_vals.append(cut_vals)
                full_sfs.append(sfs)
                full_sf_errs.append(None)
            else:
                cut_vals, sfs, sf_errs, sf[i], sferr[i] = get_sf(
                    energy, classifier, peak, fit_widths[i], cut, eres_pars
                )
                full_cut_vals.append(cut_vals)
                full_sfs.append(sfs)
                full_sf_errs.append(sf_errs)

            log.info(f"{peak}keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %")

        sf_2side = np.zeros(len(peaks_of_interest))
        sferr_2side = np.zeros(len(peaks_of_interest))
        log.info("Calculating 2 sided cut sfs")
        for i, peak in enumerate(peaks_of_interest):
            if peak == 2039:
                sf_2side[i] = compton_sf_no_sweep(
                    energy,
                    classifier,
                    peak,
                    eres_pars,
                    cut,
                    aoe_high_cut_val=aoe_high_cut,
                )
                sferr_2side[i] = 0
            else:
                sf_2side[i], sferr_2side[i] = get_sf_no_sweep(
                    energy,
                    classifier,
                    peak,
                    fit_widths[i],
                    eres_pars,
                    cut,
                    aoe_high_cut_val=aoe_high_cut,
                )

            log.info(f"{peak}keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %")

        def convert_sfs_to_dict(peaks_of_interest, sfs, sf_errs):
            out_dict = {}
            for i, peak in enumerate(peaks_of_interest):
                out_dict[str(peak)] = {
                    "sf": f"{sfs[i]:2f}",
                    "sf_err": f"{sf_errs[i]:2f}",
                }
            return out_dict

        out_dict = {
            "correction_fit_results": results_dict,
            "A/E_Energy_param": energy_param,
            "Cal_energy_param": cal_energy_param,
            "dt_param": "dt_eff",
            "rt_correction": dt_corr,
            "1000-1300keV_mean": bulk_pars[2],
            "Mean_pars": list(mu_pars),
            "Sigma_pars": list(sigma_pars),
            "Low_cut": cut,
            "High_cut": aoe_high_cut,
            "Low_side_sfs": convert_sfs_to_dict(peaks_of_interest, sf, sferr),
            "2_side_sfs": convert_sfs_to_dict(peaks_of_interest, sf_2side, sferr_2side),
        }
        log.info("Done")
        log.info(f"Results are {out_dict}")

        if display > 0:
            plot_dict = {}

            plt.rcParams["figure.figsize"] = (12, 8)
            plt.rcParams["font.size"] = 16

            fig1 = plt.figure()
            plt.subplot(3, 2, 1)
            plot_dt_dep(aoe, energy, dt, [1582, 1602], f"Tl DEP")
            plt.subplot(3, 2, 2)
            plot_dt_dep(aoe, energy, dt, [1510, 1630], f"Bi FEP")
            plt.subplot(3, 2, 3)
            plot_dt_dep(aoe, energy, dt, [2030, 2050], "Qbb")
            plt.subplot(3, 2, 4)
            plot_dt_dep(aoe, energy, dt, [2080, 2120], f"Tl SEP")
            plt.subplot(3, 2, 5)
            plot_dt_dep(aoe, energy, dt, [2584, 2638], f"Tl FEP")
            plt.tight_layout()
            plot_dict["dt_deps"] = fig1
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig2 = plt.figure()
            plot_compt_bands_overlayed(aoe, energy, [950, 1250, 1460, 1660, 1860, 2060])
            plt.ylabel("Counts")
            plt.xlabel("Raw A/E")
            plt.title(f"Compton Bands before Correction")
            plt.legend(loc="upper left")
            plot_dict["compt_bands_nocorr"] = fig2
            if display > 1:
                plt.show()
            else:
                plt.close()

            if dt_corr == True:
                _, plot_dict = drift_time_correction(
                    aoe_uncorr, energy, dt, display=display, plot_dict=plot_dict
                )

            mu_pars, sigma_pars, results_dict, plot_dict = AoEcorrection(
                energy, aoe, eres_pars, plot_dict, display=display
            )

            fig3 = plt.figure()
            plot_compt_bands_overlayed(
                classifier, energy, [950, 1250, 1460, 1660, 1860, 2060], [-5, 5]
            )
            plt.ylabel("Counts")
            plt.xlabel("Corrected A/E")
            plt.title(f"Compton Bands after Correction")
            plt.legend(loc="upper left")
            plot_dict["compt_bands_corr"] = fig3
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig4 = plt.figure()
            plt.vlines(cut, 0, 100, label=f"Cut Value: {cut:1.2f}", color="black")

            for i, peak in enumerate(peaks_of_interest):
                if peak == 2039:
                    plt.plot(
                        full_cut_vals[i],
                        full_sfs[i],
                        label=f"{get_peak_label(peak)} {peak} keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %",
                    )
                else:
                    plt.errorbar(
                        full_cut_vals[i],
                        full_sfs[i],
                        yerr=full_sf_errs[i],
                        label=f"{get_peak_label(peak)} {peak} keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %",
                    )

            handles, labels = plt.gca().get_legend_handles_labels()
            order = [1, 2, 3, 0, 4, 5]
            plt.legend(
                [handles[idx] for idx in order],
                [labels[idx] for idx in order],
                loc="upper right",
            )
            plt.xlabel("Cut Value")
            plt.ylabel("Survival Fraction %")
            plot_dict["surv_fracs"] = fig4
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig5, ax = plt.subplots()
            bins = np.linspace(1000, 3000, 2000)
            ax.hist(energy, bins=bins, histtype="step", label="Before PSD")
            ax.hist(
                energy[classifier > cut],
                bins=bins,
                histtype="step",
                label="Low side PSD cut",
            )
            ax.hist(
                energy[(classifier > cut) & (classifier < 4)],
                bins=bins,
                histtype="step",
                label="Double sided PSD cut",
            )
            ax.hist(
                energy[(classifier < cut) | (classifier > 4)],
                bins=bins,
                histtype="step",
                label="Rejected by PSD cut",
            )

            axins = ax.inset_axes([0.25, 0.07, 0.4, 0.3])
            bins = np.linspace(1580, 1640, 200)
            axins.hist(energy, bins=bins, histtype="step")
            axins.hist(energy[classifier > cut], bins=bins, histtype="step")
            axins.hist(
                energy[(classifier > cut) & (classifier < aoe_high_cut)],
                bins=bins,
                histtype="step",
            )
            axins.hist(
                energy[(classifier < cut) | (classifier > aoe_high_cut)],
                bins=bins,
                histtype="step",
            )
            ax.set_xlim([1000, 3000])
            ax.set_yscale("log")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.legend(loc="upper left")
            plot_dict["PSD_spectrum"] = fig5
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig6 = plt.figure()
            bins = np.linspace(1000, 3000, 1000)
            counts_pass, bins_pass, _ = pgh.get_hist(
                energy[(classifier > cut) & (classifier < aoe_high_cut)], bins=bins
            )
            counts, bins, _ = pgh.get_hist(energy, bins=bins)
            survival_fracs = counts_pass / (counts)

            plt.step(pgh.get_bin_centers(bins_pass), survival_fracs)
            plt.xlabel("Energy (keV)")
            plt.ylabel("Survival Fraction")
            plt.ylim([0, 1])
            plot_dict["psd_sf"] = fig6
            if display > 1:
                plt.show()
            else:
                plt.close()

            return cal_dict, out_dict, plot_dict
        else:
            return cal_dict, out_dict

    except:
        log.error("survival fraction determination failed")
        out_dict = {
            "correction_fit_results": results_dict,
            "A/E_Energy_param": "cuspEmax",
            "Cal_energy_param": "cuspEmax_ctc",
            "dt_param": "dt_eff",
            "rt_correction": False,
            "1000-1300keV_mean": bulk_pars[2],
            "Mean_pars": list(mu_pars),
            "Sigma_pars": list(sigma_pars),
            "Low_cut": cut,
            "High_cut": aoe_high_cut,
        }
        if display > 0:
            plot_dict = {}
            return cal_dict, out_dict, plot_dict
        else:
            return cal_dict, out_dict
