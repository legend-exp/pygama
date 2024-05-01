"""
This module contains the functions for performing the energy optimisation.
This happens in 2 steps, firstly a grid search is performed on each peak
separately using the optimiser, then the resulting grids are interpolated
to provide the best energy resolution at Qbb
"""

import logging

import lgdo.lh5 as lh5
import matplotlib.pyplot as plt
import numpy as np

import pygama.math.distributions as pgd
import pygama.math.histogram as pgh
import pygama.pargen.energy_cal as pgc
from pygama.pargen.utils import convert_to_minuit, return_nans

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


def simple_guess(energy, func, fit_range=None, bin_width=None):
    """
    Simple guess for peak fitting
    """
    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))

    energy = energy[(energy >= fit_range[0]) & (energy <= fit_range[1])]
    if bin_width is None:
        init_bin_width = (
            2
            * (np.nanpercentile(energy, 75) - np.nanpercentile(energy, 25))
            * len(energy) ** (-1 / 3)
        )
        init_hist, init_bins, _ = pgh.get_hist(
            energy, dx=init_bin_width, range=fit_range
        )
        try:
            _, init_sigma, _ = pgh.get_gaussian_guess(init_hist, init_bins)
        except IndexError:
            init_hist, init_bins, _ = pgh.get_hist(
                energy, dx=init_bin_width / 2, range=fit_range
            )
            try:
                _, init_sigma, _ = pgh.get_gaussian_guess(init_hist, init_bins)
            except IndexError:
                init_sigma = np.nanstd(energy)
        bin_width = (init_sigma) * len(energy) ** (-1 / 3)

    hist, bins, var = pgh.get_hist(energy, dx=bin_width, range=fit_range)

    # make binning dynamic based on max, % of events/ n of events?
    hist, bins, var = pgh.get_hist(energy, range=fit_range, dx=bin_width)

    if func == pgd.hpge_peak or func == pgd.gauss_on_step:
        mu, sigma, amp = pgh.get_gaussian_guess(hist, bins)
        i_0 = np.argmax(hist)
        bg = np.mean(hist[-10:])
        step = bg - np.mean(hist[:10])
        hstep = step / (bg + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((4 * sigma) // dx)
        nsig = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
        nbkg = np.sum(hist) - nsig

        parguess = {
            "n_sig": nsig,
            "mu": mu,
            "sigma": sigma,
            "n_bkg": nbkg,
            "hstep": hstep,
            "x_lo": fit_range[0],
            "x_hi": fit_range[1],
        }

        if func == pgd.hpge_peak:
            htail = 1.0 / 5
            tau = 0.5 * sigma
            parguess["htail"] = htail
            parguess["tau"] = tau

    else:
        log.error(f"simple_guess not implemented for {func.__name__}")
        return return_nans(func)

    return convert_to_minuit(parguess, func).values


def get_peak_fwhm_with_dt_corr(
    energies,
    alpha,
    dt,
    func,
    peak,
    kev_width,
    guess=None,
    kev=False,
    frac_max=0.5,
    bin_width=1,
    allow_tail_drop=False,
    display=0,
):
    """
    Applies the drift time correction and fits the peak returns the fwhm, fwhm/max and associated errors,
    along with the number of signal events and the reduced chi square of the fit. Can return result in ADC or keV.
    """

    correction = np.multiply(
        np.multiply(alpha, dt, dtype="float64"), energies, dtype="float64"
    )
    ct_energy = np.add(correction, energies)

    lower_bound = (np.nanmin(ct_energy) // bin_width) * bin_width
    upper_bound = ((np.nanmax(ct_energy) // bin_width) + 1) * bin_width
    hist, bins, var = pgh.get_hist(
        ct_energy, dx=bin_width, range=(lower_bound, upper_bound)
    )
    mu = bins[np.nanargmax(hist)]
    adc_to_kev = mu / peak
    # Making the window slightly smaller removes effects where as mu moves edge can be outside bin width
    lower_bound = mu - ((kev_width[0] - 2) * adc_to_kev)
    upper_bound = mu + ((kev_width[1] - 2) * adc_to_kev)
    win_idxs = (ct_energy > lower_bound) & (ct_energy < upper_bound)
    fit_range = (lower_bound, upper_bound)
    tol = None
    try:
        (
            energy_pars,
            energy_err,
            cov,
            chisqr,
            func,
            _,
            _,
            _,
        ) = pgc.unbinned_staged_energy_fit(
            ct_energy[win_idxs],
            func=func,
            fit_range=fit_range,
            guess_func=simple_guess,
            tol=tol,
            guess=guess,
            allow_tail_drop=allow_tail_drop,
            bin_width=bin_width,
            display=display,
        )
        if display > 0:
            plt.figure()
            xs = np.arange(lower_bound, upper_bound, bin_width)
            fit_hist, fit_bins, _ = pgh.get_hist(
                ct_energy, dx=bin_width, range=(lower_bound, upper_bound)
            )
            plt.step(pgh.get_bin_centers(fit_bins), fit_hist)
            plt.plot(xs, func.get_pdf(xs, *energy_pars))
            plt.show()

        fwhm = func.get_fwfm(energy_pars, frac_max=frac_max)

        xs = np.arange(lower_bound, upper_bound, 0.1)
        y = func.get_pdf(xs, *energy_pars)
        max_val = np.amax(y)
        fwhm_o_max = fwhm / max_val

        rng = np.random.default_rng(1)
        # generate set of bootstrapped parameters
        par_b = rng.multivariate_normal(energy_pars, cov, size=100)
        y_max = np.array([func.get_pdf(xs, *p) for p in par_b])
        maxs = np.nanmax(y_max, axis=1)

        y_b = np.zeros(len(par_b))
        for i, p in enumerate(par_b):
            try:
                y_b[i] = func.get_fwfm(p, frac_max=frac_max)
            except Exception:
                y_b[i] = np.nan
        fwhm_err = np.nanstd(y_b, axis=0)
        fwhm_o_max_err = np.nanstd(y_b / maxs, axis=0)

        if display > 1:
            plt.figure()
            plt.step(pgh.get_bin_centers(bins), hist)
            for i in range(100):
                plt.plot(xs, y_max[i, :])
            plt.show()

        if display > 0:
            plt.figure()
            hist, bins, var = pgh.get_hist(
                ct_energy, dx=bin_width, range=(lower_bound, upper_bound)
            )
            plt.step(pgh.get_bin_centers(bins), hist)
            plt.plot(xs, y, color="orange")
            yerr_boot = np.nanstd(y_max, axis=0)
            plt.fill_between(
                xs, y - yerr_boot, y + yerr_boot, facecolor="C1", alpha=0.5
            )
            plt.show()

    except Exception:
        return np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, None

    if kev is True:
        fwhm *= peak / energy_pars["mu"]
        fwhm_err *= peak / energy_pars["mu"]

    return (
        fwhm,
        fwhm_o_max,
        fwhm_err,
        fwhm_o_max_err,
        chisqr,
        energy_pars["n_sig"],
        energy_err["n_sig"],
        energy_pars,
    )


def fom_fwhm_with_alpha_fit(
    tb_in, kwarg_dict, ctc_parameter, nsteps=11, idxs=None, frac_max=0.2, display=0
):
    """
    FOM for sweeping over ctc values to find the best value, returns the best found fwhm with its error,
    the corresponding alpha value and the number of events in the fitted peak, also the reduced chisquare of the
    """
    parameter = kwarg_dict["parameter"]
    func = kwarg_dict["func"]
    energies = tb_in[parameter].nda
    energies = energies.astype("float64")
    peak = kwarg_dict["peak"]
    kev_width = kwarg_dict["kev_width"]
    bin_width = kwarg_dict.get("bin_width", 1)
    min_alpha = 0
    max_alpha = 3.50e-06
    alphas = np.linspace(min_alpha, max_alpha, nsteps, dtype="float64")
    try:
        dt = tb_in[ctc_parameter].nda
    except KeyError:
        dt = tb_in.eval(ctc_parameter)
    if idxs is not None:
        energies = energies[idxs]
        dt = dt[idxs]
    try:
        if np.isnan(energies).any():
            log.debug("nan in energies")
            raise RuntimeError
        if np.isnan(dt).any():
            log.debug("nan in dts")
            raise RuntimeError
        fwhms = np.array([])
        final_alphas = np.array([])
        fwhm_errs = np.array([])
        best_fwhm = np.inf
        early_break = False
        for alpha in alphas:
            (
                _,
                fwhm_o_max,
                _,
                fwhm_o_max_err,
                _,
                _,
                _,
                fit_pars,
            ) = get_peak_fwhm_with_dt_corr(
                energies,
                alpha,
                dt,
                func,
                peak,
                kev_width,
                guess=None,
                frac_max=0.5,
                bin_width=bin_width,
                allow_tail_drop=False,
            )
            if not np.isnan(fwhm_o_max):
                fwhms = np.append(fwhms, fwhm_o_max)
                final_alphas = np.append(final_alphas, alpha)
                fwhm_errs = np.append(fwhm_errs, fwhm_o_max_err)
                if fwhms[-1] < best_fwhm:
                    best_fwhm = fwhms[-1]
            log.info(f"alpha: {alpha}, fwhm/max:{fwhm_o_max:.4f}+-{fwhm_o_max_err:.4f}")

            ids = (fwhm_errs < 2 * np.nanpercentile(fwhm_errs, 50)) & (
                fwhm_errs > 1e-10
            )
            if len(fwhms[ids]) > 5:
                if (np.diff(fwhms[ids])[-3:] > 0).all():
                    early_break = True
                    break

        # Make sure fit isn't based on only a few points
        if len(fwhms) < nsteps * 0.2 and early_break is False:
            log.debug("less than 20% fits successful")
            raise RuntimeError

        ids = (fwhm_errs < 2 * np.nanpercentile(fwhm_errs, 50)) & (fwhm_errs > 1e-10)
        # Fit alpha curve to get best alpha

        try:
            alphas = np.linspace(
                final_alphas[ids][0],
                final_alphas[ids][-1],
                nsteps * 20,
                dtype="float64",
            )
            alpha_fit, cov = np.polyfit(
                final_alphas[ids], fwhms[ids], w=1 / fwhm_errs[ids], deg=4, cov=True
            )
            fit_vals = np.polynomial.polynomial.polyval(alphas, alpha_fit[::-1])
            alpha = alphas[np.nanargmin(fit_vals)]

            rng = np.random.default_rng(1)
            alpha_pars_b = rng.multivariate_normal(alpha_fit, cov, size=1000)
            fits = np.array(
                [
                    np.polynomial.polynomial.polyval(alphas, pars[::-1])
                    for pars in alpha_pars_b
                ]
            )
            min_alphas = np.array([alphas[np.nanargmin(fit)] for fit in fits])
            alpha_err = np.nanstd(min_alphas)
            if display > 0:
                plt.figure()
                yerr_boot = np.nanstd(fits, axis=0)
                plt.errorbar(final_alphas, fwhms, yerr=fwhm_errs, linestyle=" ")
                plt.plot(alphas, fit_vals)
                plt.fill_between(
                    alphas,
                    fit_vals - yerr_boot,
                    fit_vals + yerr_boot,
                    facecolor="C1",
                    alpha=0.5,
                )
                plt.show()

        except Exception:
            log.debug("alpha fit failed")

        if np.isnan(fit_vals).all():
            log.debug("alpha fit all nan")
            raise RuntimeError
        (
            final_fwhm,
            _,
            final_err,
            _,
            csqr,
            n_sig,
            n_sig_err,
            _,
        ) = get_peak_fwhm_with_dt_corr(
            energies,
            alpha,
            dt,
            func,
            peak,
            kev_width,
            guess=None,
            kev=True,
            frac_max=frac_max,
            allow_tail_drop=True,
            bin_width=bin_width,
            display=display,
        )
        if np.isnan(final_fwhm) or np.isnan(final_err):
            log.debug(f"final fit failed, alpha was {alpha}")
            raise RuntimeError
        return {
            "fwhm": final_fwhm,
            "fwhm_err": final_err,
            "alpha": alpha,
            "alpha_err": alpha_err,
            "chisquare": csqr,
            "n_sig": n_sig,
            "n_sig_err": n_sig_err,
        }
    except Exception:
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": 0,
            "alpha_err": np.nan,
            "chisquare": (np.nan, np.nan),
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }


def fom_fwhm_no_alpha_sweep(
    tb_in, kwarg_dict, ctc_param=None, alpha=0, idxs=None, frac_max=0.5, display=0
):
    """
    FOM with no ctc sweep, used for optimising ftp.
    """
    parameter = kwarg_dict["parameter"]
    func = kwarg_dict["func"]
    energies = tb_in[parameter].nda
    energies = energies.astype("float64")
    peak = kwarg_dict["peak"]
    kev_width = kwarg_dict["kev_width"]
    alpha = kwarg_dict.get("alpha", alpha)
    bin_width = kwarg_dict.get("bin_width", 1)
    if isinstance(alpha, dict):
        alpha = alpha[parameter]
    if "ctc_param" in kwarg_dict or ctc_param is not None:
        ctc_param = kwarg_dict.get("ctc_param", ctc_param)
        try:
            dt = tb_in[ctc_param].nda
        except KeyError:
            dt = tb_in.eval(ctc_param)
            dt = tb_in[ctc_param].nda
    else:
        dt = 0

    if idxs is not None:
        energies = energies[idxs]
        dt = dt[idxs]

    if np.isnan(energies).any():
        return {
            "fwhm": np.nan,
            "fwhm_o_max": np.nan,
            "fwhm_err": np.nan,
            "fwhm_o_max_err": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }
    (
        fwhm,
        final_fwhm_o_max,
        fwhm_err,
        final_fwhm_o_max_err,
        csqr,
        n_sig,
        n_sig_err,
        fit_pars,
    ) = get_peak_fwhm_with_dt_corr(
        energies,
        alpha,
        dt,
        func,
        peak=peak,
        kev_width=kev_width,
        frac_max=frac_max,
        kev=True,
        bin_width=bin_width,
        display=display,
    )
    return {
        "fwhm": fwhm,
        "fwhm_o_max": final_fwhm_o_max,
        "fwhm_err": fwhm_err,
        "fwhm_o_max_err": final_fwhm_o_max_err,
        "chisquare": csqr,
        "n_sig": n_sig,
        "n_sig_err": n_sig_err,
    }


def fom_single_peak_alpha_sweep(data, kwarg_dict, display=0):
    idx_list = kwarg_dict["idx_list"]
    ctc_param = kwarg_dict["ctc_param"]
    peak_dicts = kwarg_dict["peak_dicts"]
    frac_max = kwarg_dict.get("frac_max", 0.2)
    out_dict = fom_fwhm_with_alpha_fit(
        data,
        peak_dicts[0],
        ctc_param,
        idxs=idx_list[0],
        frac_max=frac_max,
        display=display,
    )
    return out_dict


def fom_interpolate_energy_res_with_single_peak_alpha_sweep(
    data, kwarg_dict, display=0
):
    peaks = kwarg_dict["peaks_kev"]
    idx_list = kwarg_dict["idx_list"]
    ctc_param = kwarg_dict["ctc_param"]
    peak_dicts = kwarg_dict["peak_dicts"]
    interp_energy = kwarg_dict.get("interp_energy", {"Qbb": 2039})
    fwhm_func = kwarg_dict.get("fwhm_func", pgc.FWHMLinear)
    frac_max = kwarg_dict.get("frac_max", 0.2)

    out_dict = fom_fwhm_with_alpha_fit(
        data,
        peak_dicts[-1],
        ctc_param,
        idxs=idx_list[-1],
        frac_max=frac_max,
        display=display,
    )
    alpha = out_dict["alpha"]
    log.info(alpha)
    fwhms = []
    fwhm_errs = []
    n_sig = []
    n_sig_err = []
    for i, _ in enumerate(peaks[:-1]):
        out_peak_dict = fom_fwhm_no_alpha_sweep(
            data,
            peak_dicts[i],
            ctc_param,
            alpha=alpha,
            idxs=idx_list[i],
            frac_max=frac_max,
            display=display,
        )
        fwhms.append(out_peak_dict["fwhm"])
        fwhm_errs.append(out_peak_dict["fwhm_err"])
        n_sig.append(out_peak_dict["n_sig"])
        n_sig_err.append(out_peak_dict["n_sig_err"])
    fwhms.append(out_dict["fwhm"])
    fwhm_errs.append(out_dict["fwhm_err"])
    n_sig.append(out_dict["n_sig"])
    n_sig_err.append(out_dict["n_sig_err"])
    log.info(f"fwhms are {fwhms}keV +- {fwhm_errs}")

    fwhms = np.array(fwhms)
    fwhm_errs = np.array(fwhm_errs)
    n_sig = np.array(n_sig)
    n_sig_err = np.array(n_sig_err)
    peaks = np.array(peaks)

    nan_mask = np.isnan(fwhms) | (fwhms < 0)
    if len(fwhms[~nan_mask]) < 2:
        return np.nan, np.nan, np.nan
    else:
        results = pgc.HPGeCalibration.fit_energy_res_curve(
            fwhm_func, peaks[~nan_mask], fwhms[~nan_mask], fwhm_errs[~nan_mask]
        )
        results = pgc.HPGeCalibration.interpolate_energy_res(
            fwhm_func, peaks[~nan_mask], results, interp_energy
        )
        interp_res = results[f"{list(interp_energy)[0]}_fwhm_in_kev"]
        interp_res_err = results[f"{list(interp_energy)[0]}_fwhm_err_in_kev"]

        if nan_mask[-1] is True or nan_mask[-2] is True:
            interp_res_err = np.nan
        if interp_res_err / interp_res > 0.1:
            interp_res_err = np.nan

    log.info(f"{list(interp_energy)[0]} fwhm is {interp_res} keV +- {interp_res_err}")

    return {
        f"{list(interp_energy)[0]}_fwhm": interp_res,
        f"{list(interp_energy)[0]}_fwhm_err": interp_res_err,
        "alpha": alpha,
        "peaks": peaks.tolist(),
        "fwhms": fwhms,
        "fwhm_errs": fwhm_errs,
        "n_sig": n_sig,
        "n_sig_err": n_sig_err,
    }
