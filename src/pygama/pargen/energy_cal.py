"""routines for automatic calibration.

- hpge_find_energy_peaks (Find uncalibrated E peaks whose E spacing matches the pattern in peaks_kev)
- hpge_get_energy_peaks (Get uncalibrated E peaks at the energies of peaks_kev)
- hpge_fit_energy_peaks (fits the energy peals)
- hpge_E_calibration (main routine -- finds and fits peaks specified)
"""
from __future__ import annotations

import inspect
import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from iminuit import Minuit, cost
from iminuit.util import ValueView
from matplotlib.colors import LogNorm
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import binned_statistic, chi2

import pygama.math.binned_fitting as pgb
import pygama.math.distributions as pgf
import pygama.math.histogram as pgh
from pygama.math.histogram import get_i_local_extrema, get_i_local_maxima
import pygama.math.utils as pgu
from pygama.math.least_squares import fit_simple_scaling
from pygama.pargen.utils import convert_to_minuit, return_nans

log = logging.getLogger(__name__)


class HPGeCalibration:

    """
    Calibrate HPGe data to a set of known peaks. Class stores the calibration parameters
    as well as the peaks locations and energies used. Each function called updates a results
    dictionary with any additional information which is stored in the class.

    Parameters
    ----------
    e_uncal : array
        uncalibrated energy data
    glines : array
        list of peak energies to be fit to. Each must be in the data
    guess_kev : float
        a rough initial guess at the conversion factor from e_uncal to kev. Must
        be positive
    deg : non-negative int
        degree of the polynomial for the E_cal function E_kev = poly(e_uncal).
        deg = 0 corresponds to a simple scaling E_kev = scale * e_uncal.
        Otherwise follows the convention in np.polyfit
    uncal_is_int : bool
        if True, attempts will be made to avoid picket-fencing when binning
        e_uncal
    fixed : dict
        dictionary of fixed parameters for the calibration function
    plot_options : dict
        dictionary of options for plotting the calibration results

    """

    def __init__(
        self,
        energy_param,
        glines,
        guess_kev: float,
        deg: int = 1,
        uncal_is_int: bool = False,
        fixed=None,
        plot_options: dict = None,
    ):
        self.energy_param = energy_param

        if deg < -1:
            log.error(f"hpge_E_cal warning: invalid deg = {deg}")
            return
        self.deg = int(deg)

        # could change these to tuples of (kev, adc)?
        self.peaks_kev = np.asarray(glines)
        self.peak_locs = []

        if guess_kev <= 0:
            log.error(f"hpge_E_cal warning: invalid guess_kev = {guess_kev}")
        if deg ==-1:
            self.pars = np.zeros(2, dtype=float)
            self.pars[0] = guess_kev
            self.fixed = {1:1}
        elif deg == 0:
            self.pars = np.zeros(2, dtype=float)
            self.pars[1] = guess_kev
            self.fixed = {0:0}
        else:
            self.pars = np.zeros(self.deg + 1, dtype=float)
            self.pars[1] = guess_kev
            self.fixed = fixed
        self.results = {}
        

        self.uncal_is_int = uncal_is_int
        self.plot_options = plot_options

        # if not isinstance(range_kev, list):
        #     if np.isscalar(range_kev):
        #         range_kev = (range_kev / 2, range_kev / 2)
        #     range_kev = [range_kev for peak in glines]

        # if not hasattr(funcs, "__len__"):
        #     funcs = [funcs for peak in glines]

        # self.peak_args = [
        #     (peak, ranges, func) for peak, ranges, func in zip(glines, range_kev, funcs)
        # ]

    def gen_pars_dict(self):
        # rewrite this
        out_dict = {}

        return out_dict

    def fill_plot_dict(self, data, plot_dict=None):
        if plot_dict is not None:
            for key, item in self.plot_options.items():
                if item["options"] is not None:
                    plot_dict[key] = item["function"](self, data, **item["options"])
                else:
                    plot_dict[key] = item["function"](self, data)
        else:
            plot_dict = {}
        return plot_dict

    def update_results_dict(self, results_dict):
        name = inspect.stack()[1][3]
        if name in self.results:
            it = 0
            for n in self.results:
                if name in n:
                    if name == n:
                        pass
                    else:
                        new_it = int(n.split("_")[-1])
                        if new_it > it:
                            it = new_it
            it += 1
            name += f"_{it}"
        self.results[name] = results_dict

    def hpge_find_energy_peaks(
        self,
        e_uncal,
        peaks_kev=None,
        n_sigma=5,
        etol_kev=None,
        bin_width_kev=1,
        erange=None,
        var_zero=1,
        update_cal_pars=True,
    ):
        """Find uncalibrated E peaks whose E spacing matches the pattern in peaks_kev
        Note: the specialization here to units "kev" in peaks and Etol is
        unnecessary. However it is kept so that the default value for etol_kev has
        an unambiguous interpretation.

        Parameters
        ----------
        hist, bins, var : array, array, array
            Histogram of uncalibrated energies, see pgh.get_hist()
            var cannot contain any zero entries.
        peaks_kev : array
            Energies of peaks to search for (in kev)
        n_sigma : float
            Threshold for detecting a peak in sigma (i.e. sqrt(var))
        deg : int
            deg arg to pass to poly_match
        etol_kev : float
            absolute tolerance in energy for matching peaks
        var_zero : float
            number used to replace zeros of var to avoid divide-by-zero in
            hist/sqrt(var). Default value is 1. Usually when var = 0 its because
            hist = 0, and any value here is fine.
        """

        if peaks_kev is None:
            peaks_kev = self.peaks_kev

        peaks_adc = [(Polynomial(self.pars) - ei).roots() for ei in peaks_kev]

        # bin the histogram in ~1 kev bins for the initial rough peak search
        if erange is None:
            euc_min = np.nanmin(peaks_adc) * 0.6
            euc_max = np.nanmax(peaks_adc) * 1.1
        else:
            euc_min, euc_max = erange
        d_euc = bin_width_kev / self.pars[1]
        if self.uncal_is_int:
            euc_min, euc_max, d_euc = pgh.better_int_binning(
                x_lo=euc_min, x_hi=euc_max, dx=d_euc
            )
        hist, bins, var = pgh.get_hist(e_uncal, range=(euc_min, euc_max), dx=d_euc)

        # clean up var if necessary
        if np.any(var == 0):
            log.debug(f"hpge_find_energy_peaks: replacing var zeros with {var_zero}")
            var[np.where(var == 0)] = var_zero
        peaks_kev = np.asarray(peaks_kev)

        # Find all maxes with > n_sigma significance
        imaxes = get_i_local_maxima(hist / np.sqrt(var), n_sigma)

        # Now pattern match to peaks_kev within etol_kev using poly_match
        detected_max_locs = pgh.get_bin_centers(bins)[imaxes]

        if etol_kev is None:
            # estimate etol_kev
            pt_pars, pt_covs = hpge_fit_energy_peak_tops(
                hist, bins, var, detected_max_locs, n_to_fit=15
            )
            if (
                sum(np.sum(c.flatten()) if c.ndim != 0 else 0 for c in pt_covs)
                == np.inf
                or sum(np.sum(c.flatten()) if c.ndim != 0 else 0 for c in pt_covs) == 0
            ):
                log.debug(
                    "hpge_find_energy_peaks: can safely ignore previous covariance warning, not used"
                )
            pt_pars = pt_pars[np.array([x is not None for x in pt_pars])]
            med_sigma_ratio = np.median(
                np.stack(pt_pars)[:, 1] / np.stack(pt_pars)[:, 0]
            )

            etol_kev = 5.0 * (med_sigma_ratio / 0.003)
        pars, ixtup, iytup = poly_match(
            detected_max_locs, peaks_kev, deg=self.deg, atol=etol_kev, fixed=self.fixed
        )

        if len(ixtup) != len(peaks_kev):
            log.info(
                f"hpge_find_energy_peaks: only found {len(ixtup)} of {len(peaks_kev)} expected peaks"
            )

        self.update_results_dict(
            {
                "input_peaks_kev": peaks_kev,
                "found_peaks_kev": peaks_kev[iytup],
                "found_peaks_locs": detected_max_locs[iytup],
            }
        )
        log.info(f"{len(peaks_kev[iytup])} peaks found:")
        log.info("\t   Energy   | Position  ")
        for i, (li, ei) in enumerate(zip(detected_max_locs[ixtup], peaks_kev[iytup])):
            log.info(f"\t{i}".ljust(4) + str(ei).ljust(9) + f"| {li:g}".ljust(5))

        if update_cal_pars is False:
            return

        self.peak_locs = detected_max_locs[ixtup]
        self.peaks_kev = peaks_kev[iytup]
        self.pars = np.array(pars)

    def hpge_get_energy_peaks(
        self,
        e_uncal,
        peaks_kev=None,
        n_sigma=3,
        etol_kev=5,
        var_zero=1,
        bin_width_kev=0.2,
        update_cal_pars=True,
        erange=None,
    ):
        """Get uncalibrated E peaks at the energies of peaks_kev

        Parameters
        ----------
        hist, bins, var : array, array, array
            Histogram of uncalibrated energies, see pgh.get_hist()
            var cannot contain any zero entries.
        cal_pars : array
            Estimated energy calibration parameters used to search for peaks
        peaks_kev : array
            Energies of peaks to search for (in kev)
        n_sigma : float
            Threshold for detecting a peak in sigma (i.e. sqrt(var))
        etol_kev : float
            absolute tolerance in energy for matching peaks
        var_zero : float
            number used to replace zeros of var to avoid divide-by-zero in
            hist/sqrt(var). Default value is 1. Usually when var = 0 its because
            hist = 0, and any value here is fine.
        """
        if peaks_kev is None:
            peaks_kev = self.peaks_kev

        # re-bin the histogram in ~0.2 kev bins with updated E scale par for peak-top fits
        if erange is None:
            euc_min, euc_max = (
                (Polynomial(self.pars) - i).roots()
                for i in (peaks_kev[0] * 0.9, peaks_kev[-1] * 1.1)
            )
            euc_min = euc_min[np.logical_and(euc_min >= 0, euc_min <= max(euc_max))][0]
            euc_max = euc_max[
                np.logical_and(euc_max >= euc_min, euc_max <= np.nanmax(e_uncal) * 1.1)
            ][0]
        else:
            euc_min, euc_max = erange

        d_euc = bin_width_kev / self.pars[1]

        if self.uncal_is_int:
            euc_min, euc_max, d_euc = pgh.better_int_binning(
                x_lo=euc_min, x_hi=euc_max, dx=d_euc
            )
        hist, bins, var = pgh.get_hist(e_uncal, range=(euc_min, euc_max), dx=d_euc)

        # clean up var if necessary
        if np.any(var == 0):
            log.debug(f"hpge_find_energy_peaks: replacing var zeros with {var_zero}")
            var[np.where(var == 0)] = var_zero

        # Find all maxes with > n_sigma significance
        imaxes = get_i_local_maxima(hist / np.sqrt(var), n_sigma)

        # Keep maxes if they coincide with expected peaks
        test_peaks_kev = np.asarray([pgf.nb_poly(i, self.pars) for i in bins[imaxes]])
        imatch = [abs(peaks_kev - i).min() < etol_kev for i in test_peaks_kev]

        got_peak_locations = bins[imaxes[imatch]]
        got_peak_energies = test_peaks_kev[imatch]

        # Match calculated and true peak energies
        matched_energies = peaks_kev[
            [np.argmin(abs(peaks_kev - i)) for i in got_peak_energies]
        ]
        while not all([list(matched_energies).count(x) == 1 for x in matched_energies]):
            for i in range(len(matched_energies)):
                if matched_energies[i + 1] == matched_energies[i]:
                    # remove duplicates
                    if np.argmin(
                        abs(got_peak_energies[i : i + 2] - matched_energies[i])
                    ):  # i+1 is best match
                        got_peak_locations = np.delete(got_peak_locations, i)
                        got_peak_energies = np.delete(got_peak_energies, i)
                    else:  # i is best match
                        got_peak_locations = np.delete(got_peak_locations, i + 1)
                        got_peak_energies = np.delete(got_peak_energies, i + 1)
                    matched_energies = np.delete(matched_energies, i)
                    break
                i += 1

        input_peaks = peaks_kev.copy()

        self.update_results_dict(
            {
                "input_peaks_kev": input_peaks,
                "got_peaks_kev": got_peak_locations,
                "got_peaks_locs": matched_energies,
            }
        )

        if update_cal_pars is False:
            return

        self.peak_locs = got_peak_locations
        self.peaks_kev = matched_energies

        # Calculate updated calibration curve
        poly_pars = (
            Polynomial.fit(got_peak_locations, matched_energies, len(self.pars))
            .convert()
            .coef
        )
        c = cost.LeastSquares(
            matched_energies,
            got_peak_locations,
            np.full_like(got_peak_locations, 1),
            poly_wrapper,
        )
        if self.fixed is not None:
            for idx, val in self.fixed.items():
                if val is True or val is None:
                    pass
                else:
                    poly_pars[idx] = val
        m = Minuit(c, *poly_pars)
        if self.fixed is not None:
            for idx in list(self.fixed):
                m.fixed[idx] = True

        log.info(f"{len(self.peak_locs)} peaks obtained:")
        log.info("\t   Energy   | Position  ")
        for i, (li, ei) in enumerate(zip(self.peak_locs, self.peaks_kev)):
            log.info(f"\t{i}".ljust(4) + str(ei).ljust(9) + f"| {li:g}".ljust(5))

    def hpge_cal_energy_peak_tops(
        self,
        e_uncal,
    ):
        """
        Calibrates using energy peak top fits
        """

    def hpge_fit_energy_peaks(
        self,
        e_uncal,
        peaks_kev=None,
        peak_pars=None,
        default_n_bins=50,
        peak_param="mode",
        method="unbinned",
        n_events=None,
        allowed_p_val=0.05,
        tail_weight=0,
        update_cal_pars=True,
    ):
        """Fit the Energy peaks specified using the function given

        Parameters
        ----------
        e_uncal : array
            unbinned energy data to be fit
        mode_guesses : array
            array of guesses for modes of each peak
        wwidths : float or array of float
            array of widths to use for the fit windows (in units of e_uncal),
            typically on the order of 10 sigma where sigma is the peak width
        n_bins : int or array of ints
            array of number of bins to use for the fit window histogramming
        peak_pars : list of tuples
            list containing tuples of form (peak, range, func) where peak is the energy of the peak to fit,
            range is the range in kev to fit, and func is the function to fit
        method : str
            default is unbinned fit can specify to use binned fit method instead
        uncal_is_int : bool
            if True, attempts will be made to avoid picket-fencing when binning
            e_uncal
        simplex : bool determining whether to do a round of simpson minimisation before gradient minimisation
        n_events : int number of events to use for unbinned fit
        allowed_p_val : lower limit on p_val of fit
        """

        results_dict = {}
        # check no peaks in self.peaks_kev missing from peak_pars

        if peaks_kev is None:
            peaks_kev = self.peaks_kev

        # convert peak pars to array of tuples
        tmp = np.empty(len(peak_pars), dtype=object)
        tmp[:] = peak_pars
        peak_pars = tmp

        peak_pars_lines = [i[0] for i in peak_pars]
        peaks_mask = np.array(
            [True if peak in peaks_kev else False for peak in peak_pars_lines],
            dtype=bool,
        )
        peak_pars = peak_pars[peaks_mask]

        fit_peaks_mask = np.array(
            [True for i in peak_pars if i[1] is not None and i[2] is not None],
            dtype=bool,
        )
        peak_pars = peak_pars[fit_peaks_mask]

        # First calculate range around peaks to fit

        uncal_peak_pars = []
        derco = Polynomial(self.pars).deriv().coef
        for pars in peak_pars:
            peak, fit_range, func = pars

            if peak in self.peaks_kev:
                loc = self.peak_locs[np.where(peak == self.peaks_kev)][0]
            else:
                loc = (Polynomial(self.pars) - peak).roots()[0]

            if fit_range is None:
                euc_min, euc_max = (
                    (Polynomial(self.pars) - i).roots()
                    for i in (peaks_kev[0] * 0.9, peaks_kev[-1] * 1.1)
                )
                euc_min = euc_min[
                    np.logical_and(euc_min >= 0, euc_min <= max(euc_max))
                ][0]
                euc_max = euc_max[
                    np.logical_and(
                        euc_max >= euc_min, euc_max <= np.nanmax(e_uncal) * 1.1
                    )
                ][0]
                d_euc = 1 / self.pars[1]
                if self.uncal_is_int:
                    euc_min, euc_max, d_euc = pgh.better_int_binning(
                        x_lo=euc_min, x_hi=euc_max, dx=d_euc
                    )
                hist, bins, var = pgh.get_hist(
                    e_uncal, range=(euc_min, euc_max), dx=d_euc
                )
                # Need to do initial fit
                pt_pars, _ = hpge_fit_energy_peak_tops(hist, bins, var, loc, n_to_fit=7)
                # Drop failed fits
                if pt_pars[0] is not None:
                    range_uncal = float(pt_pars[0][1]) * 20
                    n_bins = default_n_bins
                else:
                    range_uncal = None
            elif isinstance(fit_range, tuple):
                der = pgf.nb_poly(peak, derco)
                range_uncal = (fit_range[0] / der, fit_range[1] / der)
                n_bins = int(sum(fit_range) / der)

            if range_uncal is not None:
                uncal_peak_pars.append((peak, loc, range_uncal, n_bins, func))

        fit_dict = {}

        for i_peak, uncal_peak_par in enumerate(uncal_peak_pars):
            peak_kev, mode_guess, wwidth_i, n_bins_i, func_i = uncal_peak_par
            wleft_i, wright_i = wwidth_i
            try:
                euc_min = mode_guess - wleft_i
                euc_max = mode_guess + wright_i

                if self.uncal_is_int is True:
                    euc_min, euc_max, n_bins_i = pgh.better_int_binning(
                        x_lo=euc_min, x_hi=euc_max, n_bins=n_bins_i
                    )
                energies = e_uncal[(e_uncal > euc_min) & (e_uncal < euc_max)][:n_events]
                binw_1 = (euc_max - euc_min) / n_bins_i
                if method == "unbinned":
                    (
                        pars_i,
                        errs_i,
                        cov_i,
                        csqr_i,
                        func_i,
                        mask,
                        valid_fit,
                        _,
                    ) = unbinned_staged_energy_fit(
                        energies,
                        func=func_i,
                        fit_range=(euc_min, euc_max),
                        guess_func=get_hpge_energy_peak_par_guess,
                        bounds_func=get_hpge_energy_bounds,
                        fixed_func=get_hpge_energy_fixed,
                        allow_tail_drop=True,
                        bin_width=binw_1,
                        tail_weight=tail_weight,
                        guess_kwargs={"mode_guess": mode_guess},
                    )
                    if pars_i["n_sig"] < 100:
                        valid_fit = False
                    csqr = csqr_i

                else:
                    hist, bins, var = pgh.get_hist(
                        energies, bins=n_bins_i, range=(euc_min, euc_max)
                    )
                    binw_1 = (bins[-1] - bins[0]) / (len(bins) - 1)
                    par_guesses = get_hpge_energy_peak_par_guess(
                        hist, bins, var, func_i, mode_guess=mode_guess
                    )
                    bounds = get_hpge_energy_bounds(func_i, par_guesses)
                    fixed, mask = get_hpge_energy_fixed(func_i)

                    x0 = get_hpge_energy_peak_par_guess(
                        energies, func_i, (euc_min, euc_max), bin_width=binw_1
                    )
                    fixed, mask = get_hpge_energy_fixed(func_i)
                    bounds = get_hpge_energy_bounds(func_i, x0)

                    pars_i, errs_i, cov_i = pgb.fit_binned(
                        func_i.get_pdf,
                        hist,
                        bins,
                        var=var,
                        guess=x0,
                        cost_func=method,
                        Extended=True,
                        fixed=fixed,
                        bounds=bounds,
                    )
                    valid_fit = True

                    csqr = pgb.goodness_of_fit(
                        hist,
                        bins,
                        None,
                        func_i.get_pdf,
                        pars_i,
                        method="Pearson",
                        scale_bins=False,
                    )
                    csqr = (csqr[0], csqr[1] + len(np.where(mask)[0]))

                if np.isnan(pars_i).any():
                    log.debug(
                        f"hpge_fit_energy_peaks: fit failed for i_peak={i_peak} at loc {mode_guess:g}, par is nan : {pars_i}"
                    )
                    raise RuntimeError

                p_val = scipy.stats.chi2.sf(csqr[0], csqr[1])

                total_events = func_i.get_total_events(pars_i, errors=errs_i)
                if (
                    cov_i is None
                    or cov_i.ndim == 0
                    or sum(sum(c) for c in cov_i[mask, :][:, mask]) == np.inf
                    or sum(sum(c) for c in cov_i[mask, :][:, mask]) == 0
                    or np.isnan(sum(sum(c) for c in cov_i[mask, :][:, mask]))
                ):
                    log.debug(
                        f"hpge_fit_energy_peaks: cov estimation failed for i_peak={i_peak} at loc {mode_guess:g}"
                    )
                    valid_pk = False

                elif valid_fit is False:
                    log.debug(
                        f"hpge_fit_energy_peaks: peak fitting failed for i_peak={i_peak} at loc {mode_guess:g}"
                    )
                    valid_pk = False

                elif (
                    errs_i is None
                    or pars_i is None
                    or np.abs(np.array(errs_i)[mask] / np.array(pars_i)[mask]) < 1e-7
                ).any() or np.isnan(np.array(errs_i)[mask]).any():
                    log.debug(
                        f"hpge_fit_energy_peaks: failed for i_peak={i_peak} at loc {mode_guess:g}, parameter error too low"
                    )
                    valid_pk = False

                elif np.abs(total_events[0] - len(energies)) / len(energies) > 0.1:
                    log.debug(
                        f"hpge_fit_energy_peaks: fit failed for i_peak={i_peak} at loc {mode_guess:g}, total_events is outside limit"
                    )
                    valid_pk = False

                elif p_val < allowed_p_val or np.isnan(p_val):
                    log.debug(
                        f"hpge_fit_energy_peaks: fit failed for i_peak={i_peak}, p-value too low: {p_val}"
                    )
                    valid_pk = False
                else:
                    valid_pk = True

            except Exception:
                log.debug(
                    f"hpge_fit_energy_peaks: fit failed for i_peak={i_peak}, unknown error"
                )
                valid_pk = False
                pars_i, errs_i, cov_i = return_nans(func_i)
                p_val = 0

            fit_dict[peak_kev] = {
                "function": func_i,
                "validity": valid_pk,
                "parameters": pars_i,
                "uncertainties": errs_i,
                "covariance": cov_i,
                "nbins": binw_1,
                "range": [euc_min, euc_max],
                "p_value": p_val,
            }

        results_dict["peak_parameters"] = fit_dict

        fitted_peaks_kev = np.array(
            [peak for peak in fit_dict if fit_dict[peak]["validity"]]
        )

        log.info(f"{len(fitted_peaks_kev)} peaks fitted:")
        for peak, peak_dict in fit_dict.items():
            if peak_dict["validity"] is True:
                varnames = peak_dict["function"].required_args()
                pars = np.asarray(peak_dict["parameters"], dtype=float)
                errors = np.asarray(peak_dict["uncertainties"], dtype=float)
                log.info(f"\tEnergy: {str(peak)}")
                log.info("\t\tParameter  |    Value +/- Sigma  ")
                for vari, pari, errorsi in zip(varnames, pars, errors):
                    log.info(
                        f'\t\t{str(vari).ljust(10)} | {("%4.2f" % pari).rjust(8)} +/- {("%4.2f" % errorsi).ljust(8)}'
                    )

        # Drop failed fits
        pk_funcs = np.array(
            [
                fit_dict[peak]["function"]
                for peak in fit_dict
                if fit_dict[peak]["validity"]
            ]
        )
        pk_pars = np.array(
            [
                fit_dict[peak]["parameters"]
                for peak in fit_dict
                if fit_dict[peak]["validity"]
            ]
        )
        pk_errors = np.array(
            [
                fit_dict[peak]["uncertainties"]
                for peak in fit_dict
                if fit_dict[peak]["validity"]
            ]
        )
        pk_covs = np.array(
            [
                fit_dict[peak]["covariance"]
                for peak in fit_dict
                if fit_dict[peak]["validity"]
            ]
        )

        if len(pk_pars) == 0:
            log.error("hpge_fit_energy_peaks: no peaks fitted")
            self.update_results_dict(results_dict)
            return

        # Do a second calibration to the results of the full peak fits
        # First, calculate the peak positions
        if peak_param == "mu":
            mus = [
                func_i.get_mu(pars_i, errors=errors_i)
                for func_i, pars_i, errors_i in zip(pk_funcs, pk_pars, pk_errors)
            ]
            mus, mu_vars = zip(*mus)

        elif peak_param == "mode":
            mus = [
                func_i.get_mode(pars_i, cov=cov_i)
                for func_i, pars_i, cov_i in zip(pk_funcs, pk_pars, pk_covs)
            ]
            mus, mu_vars = zip(*mus)
        else:
            log.error(f"hpge_E_calibration: mode {self.peak_param} not recognized")
            self.update_results_dict(results_dict)
            return

        results_dict["peak_param"] = peak_param
        mus = results_dict["pk_pos"] = np.asarray(mus)
        mu_vars = results_dict["pk_pos_uncertainties"] = np.asarray(mu_vars) ** 2

        if update_cal_pars is False:
            self.update_results_dict(results_dict)
            return

        self.peaks_kev = np.asarray(fitted_peaks_kev)
        self.peak_locs = np.asarray(mus)

        # Now fit the E scale
        try:
            pars, errs, cov = hpge_fit_energy_scale(
                mus, mu_vars, fitted_peaks_kev, deg=self.deg, fixed=self.fixed
            )

            results_dict["pk_cal_pars"] = pars
            results_dict["pk_cal_errs"] = errs
            results_dict["pk_cal_cov"] = cov

            # Invert the E scale fit to get a calibration function
            pars, errs, cov = hpge_fit_energy_cal_func(
                mus,
                mu_vars,
                fitted_peaks_kev,
                pars,
                deg=self.deg,
                fixed=self.fixed,
            )
            self.pars = np.array(pars)

        except ValueError:
            log.error("Failed to fit enough peaks to get accurate calibration")

        self.update_results_dict(results_dict)

    def get_fwhms(self):
        """
        Updates last results dictionary with fwhms in kev
        """

        peak_parameters = self.results[list(self.results)[-1]].get(
            "peak_parameters", None
        )

        if peak_parameters is None:
            log.error("No peak parameters found")
            return

        cal_fwhms = []
        cal_fwhm_errs = []
        for peak, peak_dict in peak_parameters.items():
            # Calculate the uncalibrated fwhm
            uncal_fwhm, uncal_fwhm_err = peak_dict["function"].get_fwhm(
                peak_dict["parameters"],
                cov=peak_dict["covariance"],
            )

            # Apply calibration

            derco = Polynomial(self.pars).deriv().coef
            der = pgf.nb_poly(peak, derco)
            cal_fwhm = uncal_fwhm * der
            cal_fwhm_err = uncal_fwhm_err * der

            peak_dict.update({"fwhm_in_kev": cal_fwhm, "fwhm_err_in_kev": cal_fwhm_err})

            if peak_dict["validity"] is True:
                cal_fwhms.append(cal_fwhm)
                cal_fwhm_errs.append(cal_fwhm_err)

        cal_fwhms = np.array(cal_fwhms)
        cal_fwhm_errs = np.array(cal_fwhm_errs)
        fitted_peaks_kev = np.array(
            [
                peak
                for peak, peak_dict in peak_parameters.items()
                if peak_dict["validity"]
            ]
        )

        log.info(f"{len(cal_fwhms)} FWHMs found:")
        log.info("\t   Energy   | FWHM  ")
        for i, (ei, fwhm, fwhme) in enumerate(
            zip(fitted_peaks_kev, cal_fwhms, cal_fwhm_errs)
        ):
            log.info(
                f"\t{i}".ljust(4)
                + str(ei).ljust(9)
                + f"| {fwhm:.2f}+-{fwhme:.2f} kev".ljust(5)
            )

    def fit_energy_res_curve(self, fwhm_func, interp_energy_kev=None):
        peak_parameters = self.results[list(self.results)[-1]].get(
            "peak_parameters", None
        )
        if peak_parameters is None:
            log.error("No peak parameters found")
            return
        fitted_peaks_kev = np.array(
            [
                peak
                for peak, peak_dict in peak_parameters.items()
                if peak_dict["validity"]
            ]
        )
        if "fwhm_in_kev" not in peak_parameters[fitted_peaks_kev[0]]:
            self.get_fwhms()
            peak_parameters = self.results[list(self.results)[-1]].get(
                "peak_parameters", None
            )

        fwhm_peaks = np.array([], dtype=np.float32)
        fwhms = np.array([], dtype=np.float32)
        dfwhms = np.array([], dtype=np.float32)
        all_peaks = np.array([], dtype=np.float32)
        #####
        # Remove the Doppler Broadened peaks from calibration if found
        for peak, peak_dict in peak_parameters.items():
            all_peaks = np.append(all_peaks, peak)
            if np.abs(peak - 2103.5) < 1:
                log.info("Tl SEP removed from fwhm fitting")
                pass
            elif np.abs(peak - 1592.53) < 1:
                log.info("Tl DEP removed from fwhm fitting")
                pass
            elif np.abs(peak - 511.0) < 1:
                log.info("e annihilation removed from fwhm fitting")
                pass
            elif np.isnan(peak_dict["fwhm_in_kev"]) or np.isnan(
                peak_dict["fwhm_err_in_kev"]
            ):
                log.info(f"{peak} failed, removed from fwhm fitting")
                pass
            else:
                fwhm_peaks = np.append(fwhm_peaks, peak)
                fwhms = np.append(fwhms, peak_dict["fwhm_in_kev"])
                dfwhms = np.append(dfwhms, peak_dict["fwhm_err_in_kev"])

        log.info(f"Running FWHM fit for : {fwhm_func.__name__}")

        try:
            c_lin = cost.LeastSquares(fwhm_peaks, fwhms, dfwhms, fwhm_func.func)
            c_lin.loss = "soft_l1"
            m = Minuit(c_lin, *fwhm_func.guess(fwhm_peaks, fwhms, dfwhms))
            bounds = fwhm_func.bounds()
            for arg, val in bounds:
                m.limits[arg] = val
            m.simplex()
            m.migrad()
            m.hesse()

            p_val = scipy.stats.chi2.sf(m.fval, len(fwhm_peaks) - len(m.values))

            results = {
                "function": fwhm_func.__name__,
                "module": fwhm_func.__module__,
                "expression": fwhm_func.string_func("x"),
                "parameters": m.values,
                "uncertainties": m.errors,
                "cov": m.covariance,
                "csqr": (m.fval, len(fwhm_peaks) - len(m.values)),
                "p_val": p_val,
            }

            log.info(f'FWHM fit: {results["parameters"].to_dict()}')
            log.info("FWHM fit values:")
            log.info("\t   Energy   | FWHM (kev)  | Predicted (kev)")
            for i, (peak, fwhm, fwhme) in enumerate(zip(fwhm_peaks, fwhms, dfwhms)):
                log.info(
                    f"\t{i}".ljust(4)
                    + str(peak).ljust(9)
                    + f"| {fwhm:.2f}+-{fwhme:.2f}  ".ljust(5)
                    + f"| {fwhm_func.func(peak, *results['parameters']):.2f}".ljust(5)
                )
        except RuntimeError:
            pars, errs, cov = return_nans(fwhm_func.func)
            results = {
                "function": fwhm_func.__name__,
                "module": fwhm_func.__module__,
                "expression": fwhm_func.string_func("x"),
                "parameters": pars,
                "uncertainties": errs,
                "cov": cov,
                "csqr": (np.nan, np.nan),
                "p_val": 0,
            }
            log.error("FWHM fit failed to converge")
        if interp_energy_kev is not None:
            results = self.interpolate_energy_res(
                fwhm_peaks, fwhm_func, results, interp_energy_kev
            )
        self.results[list(self.results)[-1]].update({fwhm_func.__name__: results})

    def interpolate_energy_res(
        self, fwhm_peaks, fwhm_func, fwhm_results, interp_energy_kev=None
    ):
        if interp_energy_kev is not None:
            for key, energy in interp_energy_kev.items():
                try:
                    if energy > np.nanmax(fwhm_peaks) or energy < np.nanmin(fwhm_peaks):
                        raise RuntimeError(
                            "Interpolating energy out of range of fitted peaks"
                        )
                    rng = np.random.default_rng(1)
                    pars_b = rng.multivariate_normal(
                        fwhm_results["parameters"], fwhm_results["cov"], size=1000
                    )
                    interp_vals = np.array(
                        [fwhm_func.func(energy, *par_b) for par_b in pars_b]
                    )
                    interp_err = np.nanstd(interp_vals)
                    interp_fwhm = fwhm_func.func(energy, *fwhm_results["parameters"])
                except Exception:
                    interp_fwhm = np.nan
                    interp_err = np.nan
                fwhm_results = fwhm_results.update(
                    {
                        f"{key}_fwhm_in_kev": interp_fwhm,
                        f"{key}_fwhm_err_in_kev": interp_err,
                    }
                )
                log.info(
                    f"FWHM {key} energy resolution at {energy} : {interp_fwhm:1.2f} +- {interp_err:1.2f} kev"
                )
        return fwhm_results

    def full_calibration(
        self,
        e_uncal,
        peak_pars,
        allowed_p_val=10**-20,
        tail_weight=0,
        peak_param="mode",
        n_events=None,
    ):
        log.debug(f"Find peaks and compute calibration curve for {self.energy_param}")
        log.debug(f"Guess is {self.pars[1]:.3f}")
        self.hpge_find_energy_peaks(e_uncal)
        self.hpge_get_energy_peaks(e_uncal)

        got_peaks_kev = self.peaks_kev.copy()
        self.hpge_fit_energy_peaks(
            e_uncal,
            peak_pars=peak_pars,
            allowed_p_val=allowed_p_val,
            tail_weight=tail_weight,
            peak_param=peak_param,
            n_events=n_events,
        )
        if len(self.peaks_kev) != len(got_peaks_kev):
            for i, peak in enumerate(got_peaks_kev):
                if peak not in self.peaks_kev:
                    for i, peak_par in enumerate(peak_pars):
                        if peak_par[0] == peak:
                            new_kev_ranges = (peak_par[1][0] - 5, peak_par[1][1] - 5)
                            peak_pars[i] = (peak, new_kev_ranges, peak_par[2])
            for i, peak in enumerate(self.peaks_kev):
                try:
                    if (
                        self.results["pk_fwhms"][:, 1][i]
                        / self.results["pk_fwhms"][:, 0][i]
                        > 0.05
                    ):
                        for i, peak_par in enumerate(peak_pars):
                            if peak_par[0] == peak:
                                new_kev_ranges = (
                                    peak_par[1][0] - 5,
                                    peak_par[1][1] - 5,
                                )
                                peak_pars[i] = (peak, new_kev_ranges, peak_par[2])
                except Exception:
                    pass

            self.hpge_fit_energy_peaks(
                e_uncal,
                peaks=got_peaks_kev,
                peak_pars=peak_pars,
                allowed_p_val=allowed_p_val,
                tail_weight=tail_weight,
                peak_param=peak_param,
                n_events=n_events,
            )

            if self.pars is None:
                if self.deg <1:
                    self.pars = np.full(2, np.nan)
                else:
                    self.pars = np.full(self.deg+1, np.nan)

                log.error(f"Calibration failed completely for {self.energy_param}")
                return

        log.debug("Calibrated found")
        log.info(f"Calibration pars are {self.pars}")

        self.fit_energy_res_curve(
            FWHMLinear,
            interp_energy_kev={"Qbb": 2039.0},
        )
        self.fit_energy_res_curve(
            FWHMQuadratic,
            interp_energy_kev={"Qbb": 2039.0},
        )

        # these go in dataflow
        # self.hit_dict = {self.cal_energy_param: self.gen_pars_dict()}
        # data[self.cal_energy_param] = pgf.nb_poly(data[self.energy_param], self.pars)

    def fit_calibrated_peaks(self, e_uncal):
        log.debug(f"Fitting {self.energy_param}")
        self.hpge_get_energy_peaks(e_uncal, update_cal_pars=False)
        self.hpge_fit_energy_peaks(e_uncal, update_cal_pars=False)
        self.fit_energy_res_curve(
            FWHMLinear,
            interp_energy_kev={"Qbb": 2039.0},
        )
        self.fit_energy_res_curve(
            FWHMQuadratic,
            interp_energy_kev={"Qbb": 2039.0},
        )

    def calibrate_prominent_peak(
        self,
        e_uncal,
        peak,
        peak_pars,
        allowed_p_val=10**-20,
        tail_weight=0,
        peak_param="mode",
        n_events=None,
    ):
        log.debug(f"Find peaks and compute calibration curve for {self.energy_param}")
        log.debug(f"Guess is {self.pars[1]:.3f}")
        if self.deg != 0:
            log.error("deg must be 0 for calibrate_prominent_peak")
            return
        self.hpge_find_energy_peaks(e_uncal)
        self.hpge_get_energy_peaks(e_uncal)

        got_peaks_kev = self.peaks_kev.copy()
        self.hpge_fit_energy_peaks(
            e_uncal,
            peaks_kev=[peak],
            peak_pars=peak_pars,
            allowed_p_val=allowed_p_val,
            tail_weight=tail_weight,
            peak_param=peak_param,
            n_events=n_events,
        )
        self.hpge_fit_energy_peaks(
            e_uncal,
            peaks_kev=got_peaks_kev,
            peak_pars=peak_pars,
            allowed_p_val=allowed_p_val,
            tail_weight=tail_weight,
            peak_param=peak_param,
            n_events=n_events,
            update_cal_pars=False,
        )
        self.fit_energy_res_curve(
            FWHMLinear,
            interp_energy_kev={"Qbb": 2039.0},
        )
        self.fit_energy_res_curve(
            FWHMQuadratic,
            interp_energy_kev={"Qbb": 2039.0},
        )


def fwhm_slope(x: np.array, m0: float, m1: float, m2: float = None) -> np.array:
    """
    Fit the energy resolution curve
    """
    if m2 is None:
        return np.sqrt(m0 + m1 * x)
    else:
        return np.sqrt(m0 + m1 * x + m2 * x**2)


class FWHMLinear:
    @staticmethod
    def func(x, a, b):
        return np.sqrt(a + b * x)

    @staticmethod
    def string_func(input_param):
        return f"(a+b*{input_param})**(0.5)"

    @staticmethod
    def guess(xs, ys, y_errs):
        return [np.nanmin(ys), 10**-3]

    @staticmethod
    def bounds():
        return [(0, None), (0, None)]


class FWHMQuadratic:
    @staticmethod
    def func(x, a, b, c):
        return np.sqrt(a + b * x + c * x**2)

    @staticmethod
    def string_func(input_param):
        return f"(a+b*{input_param}+c*{input_param}**2)**(0.5)"

    @staticmethod
    def guess(xs, ys, y_errs):
        return [np.nanmin(ys), 10**-3, 10**-5]

    @staticmethod
    def bounds():
        return [(0, None), (0, None), (0, None)]


def hpge_fit_energy_peak_tops(
    hist,
    bins,
    var,
    peak_locs,
    n_to_fit=7,
    cost_func="Least Squares",
    inflate_errors=False,
    gof_method="var",
):
    """Fit gaussians to the tops of peaks

    Parameters
    ----------
    hist, bins, var : array, array, array
        Histogram of uncalibrated energies, see pgh.get_hist()
    peak_locs : array
        locations of peaks in hist. Must be accurate two within +/- 2*n_to_fit
    n_to_fit : int
        number of hist bins near the peak top to include in the gaussian fit
    cost_func : bool (optional)
        Flag passed to gauss_mode_width_max()
    inflate_errors : bool (optional)
        Flag passed to gauss_mode_width_max()
    gof_method : str (optional)
        method flag passed to gauss_mode_width_max()

    Returns
    -------
    pars_list : list of array
        a list of best-fit parameters (mode, sigma, max) for each peak-top fit
    cov_list : list of 2D arrays
        a list of covariance matrices for each pars
    """
    pars_list = []
    cov_list = []
    for e_peak in peak_locs:
        try:
            pars, cov = pgb.gauss_mode_width_max(
                hist,
                bins,
                var,
                mode_guess=e_peak,
                n_bins=n_to_fit,
                cost_func=cost_func,
                inflate_errors=inflate_errors,
                gof_method=gof_method,
            )
        except Exception:
            pars, cov = None, None

        pars_list.append(pars)
        cov_list.append(cov)
    return np.array(pars_list, dtype=object), np.array(cov_list, dtype=object)


def get_hpge_energy_peak_par_guess(
    energy, func, fit_range=None, bin_width=1, mode_guess=None
):
    """Get parameter guesses for func fit to peak in hist

    Parameters
    ----------
    hist, bins, var : array, array, array
        Histogram of uncalibrated energies, see pgh.get_hist(). Should be
        windowed around the peak.
    func : function
        The function to be fit to the peak in the (windowed) hist
    """
    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))
    hist, bins, var = pgh.get_hist(energy, dx=bin_width, range=fit_range)
    if func == pgf.gauss_on_step:
        # get mu and height from a gauss fit, also sigma as fallback
        pars, cov = pgb.gauss_mode_width_max(
            hist, bins, var, mode_guess=mode_guess, n_bins=10
        )

        bin_centres = pgh.get_bin_centers(bins)
        if pars is None:
            log.info("get_hpge_energy_peak_par_guess: gauss_mode_width_max failed")
            i_0 = np.argmax(hist)
            mu = bin_centres[i_0]
            height = hist[i_0]
            sigma_guess = None
        else:
            mu = mode_guess if mode_guess is not None else pars[0]
            sigma_guess = pars[1]
            height = pars[2]
        # get bg and step from edges of hist
        bg = np.mean(hist[-10:])
        step = bg - np.mean(hist[:10])
        # get sigma from fwfm with f = 1/sqrt(e)
        try:
            sigma = pgh.get_fwfm(
                0.6065,
                hist,
                bins,
                var,
                mx=height,
                bl=bg - step / 2,
                method="interpolate",
            )[0]
            if sigma <= 0:
                raise ValueError
        except ValueError:
            sigma = pgh.get_fwfm(
                0.6065,
                hist,
                bins,
                var,
                mx=height,
                bl=bg - step / 2,
                method="fit_slopes",
            )[0]
            if sigma <= 0 or sigma > 1000:
                log.info("get_hpge_energy_peak_par_guess: sigma estimation failed")
                if sigma_guess is not None and sigma_guess > 0 and sigma_guess < 1000:
                    sigma = sigma_guess
                else:
                    (
                        _,
                        sigma,
                    ) = pgh.get_gaussian_guess(hist, bins)
                    if sigma is not None and sigma_guess > 0 and sigma_guess < 1000:
                        pass
                    else:
                        return {}
        # now compute amp and return
        n_sig = np.sum(
            hist[(bin_centres > mu - 3 * sigma) & (bin_centres < mu + 3 * sigma)]
        )
        n_bkg = np.sum(hist) - n_sig

        hstep = step / (bg + np.mean(hist[:10]))

        parguess = {
            "n_sig": n_sig,
            "mu": mu,
            "sigma": sigma,
            "n_bkg": n_bkg,
            "hstep": hstep,
            "x_lo": bins[0],
            "x_hi": bins[-1],
        }

        for name, guess in parguess.items():
            if np.isnan(guess):
                parguess[name] = 0

    elif func == pgf.hpge_peak:
        # guess mu, height
        pars, cov = pgb.gauss_mode_width_max(hist, bins, var, n_bins=10)
        bin_centres = pgh.get_bin_centers(bins)
        if pars is None:
            log.info("get_hpge_energy_peak_par_guess: gauss_mode_width_max failed")
            sigma_guess = None
        else:
            mu = mode_guess if mode_guess is not None else bin_centres[i_0]
            sigma_guess = pars[1]
            # height=pars[2]
        i_0 = np.argmax(hist)
        height = hist[i_0]

        # get bg and step from edges of hist
        bg0 = np.mean(hist[-10:])
        step = bg0 - np.mean(hist[:10])

        # get sigma from fwfm with f = 1/sqrt(e)
        try:
            sigma = pgh.get_fwfm(
                0.6065,
                hist,
                bins,
                var,
                mx=height,
                bl=bg0 + step / 2,
                method="interpolate",
            )[0]
            if sigma <= 0:
                raise ValueError
        except Exception:
            sigma = pgh.get_fwfm(
                0.6065,
                hist,
                bins,
                var,
                mx=height,
                bl=bg0 + step / 2,
                method="fit_slopes",
            )[0]
            if sigma <= 0 or sigma > 1000:
                log.info("get_hpge_energy_peak_par_guess: sigma estimation failed")
                if sigma_guess is not None and sigma_guess > 0 and sigma_guess < 1000:
                    sigma = sigma_guess
                else:
                    (
                        _,
                        sigma,
                    ) = pgh.get_gaussian_guess(hist, bins)
                    if sigma is not None and sigma_guess > 0 and sigma_guess < 1000:
                        pass
                    else:
                        return {}
        sigma = sigma * 0.8  # roughly remove some amount due to tail

        # for now hard-coded
        htail = 1.0 / 5
        tau = sigma / 2

        hstep = step / (bg0 + np.mean(hist[:10]))

        n_sig = np.sum(
            hist[(bin_centres > mu - 3 * sigma) & (bin_centres < mu + 3 * sigma)]
        )
        n_bkg = np.sum(hist) - n_sig

        parguess = {
            "n_sig": n_sig,
            "mu": mu,
            "sigma": sigma,
            "htail": htail,
            "tau": tau,
            "n_bkg": n_bkg,
            "hstep": hstep,
            "x_lo": bins[0],
            "x_hi": bins[-1],
        }

        for name, guess in parguess.items():
            if np.isnan(guess):
                parguess[name] = 0

    else:
        log.error(f"get_hpge_energy_peak_par_guess not implemented for {func.__name__}")
        return return_nans(func)

    return convert_to_minuit(parguess, func).values


def get_hpge_energy_fixed(func):
    """
    Returns: Sequence list of fixed indexes for fitting and mask for parameters
    """

    if func == pgf.gauss_on_step:
        # pars are: n_sig, mu, sigma, n_bkg, hstep, components
        fixed = ["x_lo", "x_hi"]

    elif func == pgf.hpge_peak:
        # pars are: n_sig, mu, sigma, htail,tau, n_bkg, hstep, components
        fixed = ["x_lo", "x_hi"]

    else:
        log.error(f"get_hpge_energy_fixed not implemented for {func.__name__}")
        return None
    mask = ~np.in1d(func.required_args(), fixed)
    return fixed, mask


def get_hpge_energy_bounds(func, parguess):
    if func == pgf.gauss_on_step:
        return {
            "n_sig": (0, None),
            "mu": (parguess["x_lo"], parguess["x_hi"]),
            "sigma": (0, None),
            "n_bkg": (None, None),
            "hstep": (-1, 1),
            "x_lo": (None, None),
            "x_hi": (None, None),
        }

    elif func == pgf.hpge_peak:
        return {
            "n_sig": (0, None),
            "mu": (parguess["x_lo"], parguess["x_hi"]),
            "sigma": (0, None),
            "htail": (0, 0.5),
            "tau": (0.1 * parguess["sigma"], 10 * parguess["sigma"]),
            "n_bkg": (None, None),
            "hstep": (-1, 1),
            "x_lo": (None, None),
            "x_hi": (None, None),
        }
    else:
        log.error(f"get_hpge_energy_bounds not implemented for {func.__name__}")
        return []


class TailPrior:
    """
    Generic least-squares cost function with error.
    """

    verbose = 0
    errordef = Minuit.LIKELIHOOD  # for Minuit to compute errors correctly

    def __init__(self, data, model, tail_weight=0):
        self.model = model  # model predicts y for given x
        self.data = data
        self.tail_weight = tail_weight

    def _call(self, *pars):
        return self.__call__(*pars[0])

    def __call__(
        self,
        x_lo,
        x_hi,
        n_sig,
        mu,
        sigma,
        htail,
        tau,
        n_bkg,
        hstep,
    ):
        return self.tail_weight * np.log(htail + 0.1)  # len(self.data)/


def unbinned_staged_energy_fit(
    energy,
    func,
    gof_range=None,
    fit_range=None,
    guess=None,
    guess_func=get_hpge_energy_peak_par_guess,
    bounds_func=get_hpge_energy_bounds,
    fixed_func=get_hpge_energy_fixed,
    guess_kwargs=None,
    bounds_kwargs=None,
    fixed_kwargs=None,
    tol=None,
    tail_weight=0,
    allow_tail_drop=True,
    bin_width=1,
    lock_guess=False,
    display=0,
):
    """
    Unbinned fit to energy. This is different to the default fitting as
    it will try different fitting methods and choose the best. This is necessary for the lower statistics.
    """

    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))

    if gof_range is None:
        gof_range = fit_range

    hist, bins, _ = pgh.get_hist(energy, range=fit_range, dx=bin_width)
    bin_cs = (bins[:-1] + bins[1:]) / 2

    gof_hist, gof_bins, gof_var = pgh.get_hist(energy, range=gof_range, dx=bin_width)

    if guess is not None:
        if not isinstance(guess, ValueView):
            x0 = convert_to_minuit(guess, func)
        x0["x_lo"] = fit_range[0]
        x0["x_hi"] = fit_range[1]
        x1 = guess_func(
            energy,
            func,
            fit_range,
            bin_width=bin_width,
            **guess_kwargs if guess_kwargs is not None else {},
        )
        for arg, val in x1.items():
            if arg not in x0:
                x0[arg] = val
        if lock_guess is False:
            if len(x0) == len(x1):
                cs, _ = pgb.goodness_of_fit(
                    gof_hist, gof_bins, None, func.get_pdf, x0, method="Pearson"
                )
                cs2, _ = pgb.goodness_of_fit(
                    gof_hist, gof_bins, None, func.get_pdf, x1, method="Pearson"
                )
                if cs >= cs2:
                    x0 = x1
            else:
                x0 = x1
    else:
        if func == pgf.hpge_peak:
            x0_notail = guess_func(
                energy,
                pgf.gauss_on_step,
                fit_range,
                bin_width=bin_width,
                **guess_kwargs,
            )
            c = cost.ExtendedUnbinnedNLL(energy, pgf.gauss_on_step.pdf_ext)
            m = Minuit(c, *x0_notail)
            bounds = bounds_func(
                pgf.gauss_on_step,
                x0_notail,
                **bounds_kwargs if bounds_kwargs is not None else {},
            )
            for arg, val in bounds.items():
                m.limits[arg] = val
            fixed, mask = fixed_func(
                pgf.gauss_on_step,
                **fixed_kwargs if fixed_kwargs is not None else {},
            )
            for fix in fixed:
                m.fixed[fix] = True
            m.simplex().migrad()
            m.hesse()
            x0 = guess_func(
                energy,
                func,
                fit_range,
                bin_width=bin_width,
                **guess_kwargs if guess_kwargs is not None else {},
            )
            for arg in x0_notail.to_dict():
                x0[arg] = x0_notail[arg]

        else:
            x0 = guess_func(
                energy,
                func,
                fit_range,
                bin_width=bin_width,
                **guess_kwargs if guess_kwargs is not None else {},
            )

    if (func == pgf.hpge_peak) and allow_tail_drop is True:
        fit_no_tail = unbinned_staged_energy_fit(
            energy,
            func=pgf.gauss_on_step,
            gof_range=gof_range,
            fit_range=fit_range,
            guess=None,
            guess_func=guess_func,
            bounds_func=bounds_func,
            fixed_func=fixed_func,
            guess_kwargs=guess_kwargs,
            bounds_kwargs=bounds_kwargs,
            fixed_kwargs=fixed_kwargs,
            tol=tol,
            tail_weight=None,
            allow_tail_drop=False,
            bin_width=bin_width,
        )

        c = cost.ExtendedUnbinnedNLL(energy, func.pdf_ext) + TailPrior(
            energy, func, tail_weight=tail_weight
        )
    else:
        c = cost.ExtendedUnbinnedNLL(energy, func.pdf_ext)

    fixed, mask = fixed_func(func, **fixed_kwargs if fixed_kwargs is not None else {})
    bounds = bounds_func(func, x0, **bounds_kwargs if bounds_kwargs is not None else {})

    # try without simplex
    m = Minuit(c, *x0)
    if tol is not None:
        m.tol = tol
    for fix in fixed:
        m.fixed[fix] = True
    for arg, val in bounds.items():
        m.limits[arg] = val
    m.migrad()
    m.hesse()

    valid1 = (
        m.valid
        & (~np.isnan(np.array(m.errors)[mask]).any())
        & (~(np.array(m.errors)[mask] == 0).all())
    )

    cs = pgb.goodness_of_fit(
        gof_hist,
        gof_bins,
        gof_var,
        func.get_pdf,
        m.values,
        method="Pearson",
        scale_bins=True,
    )
    cs = (cs[0], cs[1] + len(np.where(mask)[0]))

    fit1 = (m.values, m.errors, m.covariance, cs, func, mask, valid1, m)

    # Now try with simplex
    m2 = Minuit(c, *x0)
    if tol is not None:
        m2.tol = tol
    for fix in fixed:
        m2.fixed[fix] = True
    for arg, val in bounds.items():
        m2.limits[arg] = val
    m2.simplex().migrad()
    m2.hesse()

    valid2 = (
        m2.valid
        & (~np.isnan(np.array(m2.errors)[mask]).any())
        & (~(np.array(m2.errors)[mask] == 0).all())
    )

    cs2 = pgb.goodness_of_fit(
        gof_hist,
        gof_bins,
        gof_var,
        func.get_pdf,
        m2.values,
        method="Pearson",
        scale_bins=True,
    )
    cs2 = (cs2[0], cs2[1] + len(np.where(mask)[0]))

    fit2 = (m2.values, m2.errors, m2.covariance, cs2, func, mask, valid2, m2)

    frac_errors1 = np.sum(np.abs(np.array(m.errors)[mask] / np.array(m.values)[mask]))
    frac_errors2 = np.sum(np.abs(np.array(m2.errors)[mask] / np.array(m2.values)[mask]))

    if display > 1:
        m_fit = func.get_pdf(bin_cs, *m.values) * np.diff(bin_cs)[0]
        m2_fit = func.get_pdf(bin_cs, *m2.values) * np.diff(bin_cs)[0]
        plt.figure()
        plt.step(bin_cs, hist, label="hist")
        plt.plot(bin_cs, func(bin_cs, *x0)[1], label="Guess")
        plt.plot(bin_cs, m_fit, label=f"Fit 1: {cs}")
        plt.plot(bin_cs, m2_fit, label=f"Fit 2: {cs2}")
        plt.legend()
        plt.show()

    if valid1 is False and valid2 is False:
        log.debug("Extra simplex needed")
        m = Minuit(c, *x0)
        if tol is not None:
            m.tol = tol
        for fix in fixed:
            m.fixed[fix] = True
        for arg, val in bounds.items():
            m.limits[arg] = val
        m.simplex().simplex().migrad()
        m.hesse()
        cs = pgb.goodness_of_fit(
            gof_hist,
            gof_bins,
            gof_var,
            func.get_pdf,
            m.values,
            method="Pearson",
            scale_bins=True,
        )
        cs = (cs[0], cs[1] + len(np.where(mask)[0]))
        valid3 = (
            m.valid
            & (~np.isnan(np.array(m.errors)[mask]).any())
            & (~(np.array(m.errors)[mask] == 0).all())
        )
        if valid3 is False:
            try:
                m.minos()
                valid3 = (
                    m.valid
                    & (~np.isnan(np.array(m.errors)[mask]).any())
                    & (~(np.array(m.errors)[mask] == 0).all())
                )
            except Exception:
                raise RuntimeError

        fit = (m.values, m.errors, m.covariance, cs, func, mask, valid3, m)

    elif valid2 is False:
        fit = fit1

    elif valid1 is False:
        fit = fit2

    elif cs[0] * 1.05 < cs2[0]:
        fit = fit1

    elif cs2[0] * 1.05 < cs[0]:
        fit = fit2

    elif frac_errors1 < frac_errors2:
        fit = fit1

    elif frac_errors1 > frac_errors2:
        fit = fit2

    else:
        raise RuntimeError

    if (func == pgf.hpge_peak) and allow_tail_drop is True:
        p_val = chi2.sf(fit[3][0], fit[3][1])
        p_val_no_tail = chi2.sf(fit_no_tail[3][0], fit_no_tail[3][1])
        if fit[0]["htail"] < fit[1]["htail"] or p_val_no_tail > p_val:
            debug_string = f'dropping tail tail val : {fit[0]["htail"]} tail err : {fit[1]["htail"]} '
            debug_string += f"p_val no tail: : {p_val_no_tail} p_val with tail: {p_val}"
            log.debug(debug_string)
            
            #if display > 0:
            m_fit = pgf.gauss_on_step.get_pdf(bin_cs, *fit_no_tail[0])
            m_fit_tail = pgf.hpge_peak.get_pdf(bin_cs, *fit[0])
            plt.figure()
            plt.step(bin_cs, hist, where="mid", label="hist")
            plt.plot(
                bin_cs,
                m_fit * np.diff(bin_cs)[0],
                label=f"Drop tail: {p_val_no_tail}",
            )
            plt.plot(
                bin_cs,
                m_fit_tail * np.diff(bin_cs)[0],
                label=f"Drop tail: {p_val}",
            )
            plt.plot(
                bin_cs,
                pgf.hpge_peak.pdf_ext(bin_cs, *fit[0])[1] * np.diff(bin_cs)[0],
                label=f"Drop tail: {p_val}",
            )
            plt.legend()
            plt.show()
            fit = fit_no_tail
    return fit


def poly_wrapper(x, *pars):
    return pgf.nb_poly(x, np.array(pars))


def hpge_fit_energy_scale(mus, mu_vars, energies_kev, deg=0, fixed=None):
    """Find best fit of poly(E) = mus +/- sqrt(mu_vars)
    Compare to hpge_fit_energy_cal_func which fits for E = poly(mu)

    Parameters
    ----------
    mus : array
        uncalibrated energies
    mu_vars : array
        variances in the mus
    energies_kev : array
        energies to fit to, in kev
    deg : int
        degree for energy scale fit. deg=0 corresponds to a simple scaling
        mu = scale * E. Otherwise deg follows the definition in np.polyfit
    fixed : dict
        dict where keys are index of polyfit pars to fix and vals are the value
        to fix at, can be None to fix at guess value
    Returns
    -------
    pars : array
        parameters of the best fit. Follows the convention in np.polyfit
    cov : 2D array
        covariance matrix for the best fit parameters.
    """
    if deg == 0:
        scale, scale_cov = fit_simple_scaling(energies_kev, mus, var=mu_vars)
        pars = np.array([0,scale])
        cov = np.array([[0, 0], [0, scale_cov]])
        errs = np.diag(np.sqrt(cov))
    else:
        poly_pars = (
            Polynomial.fit(energies_kev, mus, deg=deg, w=1 / np.sqrt(mu_vars))
            .convert()
            .coef
        )
        c = cost.LeastSquares(energies_kev, mus, np.sqrt(mu_vars), poly_wrapper)
        if fixed is not None:
            for idx, val in fixed.items():
                if val is True or val is None:
                    pass
                else:
                    poly_pars[idx] = val
        m = Minuit(c, *poly_pars)
        if fixed is not None:
            for idx in list(fixed):
                m.fixed[idx] = True
        m.simplex()
        m.migrad()
        m.hesse()
        pars = m.values
        cov = m.covariance
        errs = m.errors
    return pars, errs, cov


def hpge_fit_energy_cal_func(
    mus, mu_vars, energies_kev, energy_scale_pars, deg=0, fixed=None
):
    """Find best fit of E = poly(mus +/- sqrt(mu_vars))
    This is an inversion of hpge_fit_energy_scale.
    E uncertainties are computed from mu_vars / dmu/dE where mu = poly(E) is the
    E_scale function

    Parameters
    ----------
    mus : array
        uncalibrated energies
    mu_vars : array
        variances in the mus
    energies_kev : array
        energies to fit to, in kev
    energy_scale_pars : array
        Parameters from the escale fit (kev to ADC) used for calculating
        uncertainties
    deg : int
        degree for energy scale fit. deg=0 corresponds to a simple scaling
        mu = scale * E. Otherwise deg follows the definition in np.polyfit
    fixed : dict
        dict where keys are index of polyfit pars to fix and vals are the value
        to fix at, can be None to fix at guess value

    Returns
    -------
    pars : array
        parameters of the best fit. Follows the convention in np.polyfit
    cov : 2D array
        covariance matrix for the best fit parameters.
    """
    if deg == 0:
        e_vars = mu_vars / energy_scale_pars[1] ** 2
        scale, scale_cov = fit_simple_scaling(mus, energies_kev, var=e_vars)
        pars = np.array([0,scale])
        cov = np.array([[0, 0], [0, scale_cov]])
        errs = np.diag(np.sqrt(cov))
    else:
        d_mu_d_es = np.zeros(len(mus))
        for n in range(len(energy_scale_pars) - 1):
            d_mu_d_es += energy_scale_pars[n] * mus ** (len(energy_scale_pars) - 2 - n)
        e_weights = d_mu_d_es * mu_vars
        poly_pars = (
            Polynomial.fit(mus, energies_kev, deg=deg, w=1 / e_weights).convert().coef
        )
        if fixed is not None:
            for idx, val in fixed.items():
                if val is True or val is None:
                    pass
                else:
                    poly_pars[idx] = val
        c = cost.LeastSquares(mus, energies_kev, e_weights, poly_wrapper)
        m = Minuit(c, *poly_pars)
        if fixed is not None:
            for idx in list(fixed):
                m.fixed[idx] = True
        m.simplex()
        m.migrad()
        m.hesse()
        pars = m.values
        errs = m.errors
        cov = m.covariance
    return pars, errs, cov


def poly_match(xx, yy, deg=-1, rtol=1e-5, atol=1e-8, fixed=None):
    """Find the polynomial function best matching pol(xx) = yy

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
        degree of the polynomial to be used. If deg = 0, will fit for a simple
        scaling: scale * xx = yy. If deg = -1, fits to a simple shift in the
        data: xx + shift = yy. Otherwise, deg is equivalent to the deg argument
        of np.polyfit()
    rtol : float
        the relative tolerance to be sent to np.isclose()
    atol : float
        the absolute tolerance to be sent to np.isclose(). Has the same units
        as yy.

    Returns
    -------
    pars: None or array of floats
        The parameters of the best fit of poly(xx) = yy.  Follows the convention
        used for the return value "p" of polyfit. Returns None when the inputs
        are bad.
    i_matches : list of int
        list of indices in xx for the matched values in the best match
    """

    # input handling
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    #    if len(xx) <= len(yy):
    #        print(f"poly_match error: len(xx)={len(xx)} <= len(yy)={len(yy)}")
    #        return None, 0
    deg = int(deg)
    if deg < -1:
        log.error(f"poly_match error: got bad deg = {deg}")
        return None, 0
    req_ylen = max(2, deg + 2)
    if len(yy) < req_ylen:
        log.error(
            f"poly_match error: len(yy) must be at least {req_ylen} for deg={deg}, got {len(yy)}"
        )
        return None, 0

    maxoverlap = min(len(xx), len(yy))

    # build ixtup: the indices in xx to compare with the values in yy
    ixtup = np.array(list(range(maxoverlap)))
    iytup = np.array(list(range(maxoverlap)))
    best_ixtup = None
    best_iytup = None
    n_close = 0
    gof = np.inf  # lower is better gof
    while True:
        xx_i = xx[ixtup]
        yy_i = yy[iytup]
        gof_i = np.inf

        # simple shift
        if deg == -1:
            pars_i = np.array([(np.sum(yy_i) - np.sum(xx_i)) / len(yy_i), 1])
            polxx = xx_i + pars_i[0]

        # simple scaling
        elif deg == 0:
            pars_i = np.array([0, np.sum(yy_i * xx_i) / np.sum(xx_i * xx_i)])
            polxx = pars_i[1] * xx_i

        # generic poly of degree >= 1
        else:
            poly_pars = Polynomial.fit(xx_i, yy_i, deg=deg).convert().coef
            c = cost.LeastSquares(xx_i, yy_i, np.full_like(yy_i, 1), poly_wrapper)
            if fixed is not None:
                for idx, val in fixed.items():
                    if val is True or val is None:
                        pass
                    else:
                        poly_pars[idx] = val
            m = Minuit(c, *poly_pars)
            if fixed is not None:
                for idx in list(fixed):
                    m.fixed[idx] = True
            pars_i = np.array(m.values)
            polxx = np.zeros(len(yy_i))
            polxx = pgf.nb_poly(xx_i, pars_i)

        # by here we have the best polxx. Search for matches and store pars_i if
        # its the best so far
        matches = np.isclose(polxx, yy_i, rtol=rtol, atol=atol)
        n_close_i = np.sum(matches)
        if n_close_i >= n_close:
            gof_i = np.sum(np.power(polxx[matches] - yy_i[matches], 2))
            if n_close_i > n_close or (n_close_i == n_close and gof_i < gof):
                n_close = n_close_i
                gof = gof_i
                pars = pars_i
                best_ixtup = np.copy(ixtup)
                best_iytup = np.copy(iytup)

        # increment ixtup
        # first find the index of ixtup that needs to be incremented
        ii = 0
        while ii < len(ixtup) - 1:
            if ixtup[ii] < ixtup[ii + 1] - 1:
                break
            ii += 1

        # quit if ii is the last index of ixtup and it's already maxed out
        if not (ii == len(ixtup) - 1 and ixtup[ii] == len(xx) - 1):
            # otherwise increment ii and reset indices < ii
            ixtup[ii] += 1
            ixtup[0:ii] = list(range(ii))
            continue

        # increment iytup
        # first find the index of iytup that needs to be incremented
        ii = 0
        while ii < len(iytup) - 1:
            if iytup[ii] < iytup[ii + 1] - 1:
                break
            ii += 1

        # quit if ii is the last index of iytup and it's already maxed out
        if not (ii == len(iytup) - 1 and iytup[ii] == len(yy) - 1):
            # otherwise increment ii and reset indices < ii
            iytup[ii] += 1
            iytup[0:ii] = list(range(ii))
            ixtup = np.array(list(range(len(iytup))))  # (reset ix)
            continue

        if n_close == len(iytup):  # found best
            break

        # reduce overlap
        new_len = len(iytup) - 1
        if new_len < req_ylen:
            break
        ixtup = np.array(list(range(new_len)))
        iytup = np.array(list(range(new_len)))

        best_ixtup = None
        best_iytup = None
        n_close = 0
        gof = np.inf

    return pars, best_ixtup, best_iytup

# move these to dataflow
def get_peak_labels(
    labels: list[str], pars: list[float]
) -> tuple(list[float], list[float]):
    out = []
    out_labels = []
    for i, label in enumerate(labels):
        if i % 2 == 1:
            continue
        else:
            out.append(f"{pgf.nb_poly(label, pars):.1f}")
            out_labels.append(label)
    return out_labels, out


def get_peak_label(peak: float) -> str:
    if peak == 583.191:
        return "Tl 583"
    elif peak == 727.33:
        return "Bi 727"
    elif peak == 860.564:
        return "Tl 860"
    elif peak == 1592.53:
        return "Tl DEP"
    elif peak == 1620.5:
        return "Bi FEP"
    elif peak == 2103.53:
        return "Tl SEP"
    elif peak == 2614.5:
        return "Tl FEP"
    else:
        return ""


def plot_fits(
    ecal_class, data, figsize=(12, 8), fontsize=12, ncols=3, nrows=3, binning_kev=5
):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fitted_peaks = ecal_class.results["got_peaks_keV"]
    pk_pars = ecal_class.results["pk_pars"]
    pk_ranges = ecal_class.results["pk_ranges"]
    p_vals = ecal_class.results["pk_pvals"]

    fitted_gof_funcs = []
    for i, peak in enumerate(ecal_class.glines):
        if peak in fitted_peaks:
            fitted_gof_funcs.append(ecal_class.gof_funcs[i])

    mus = [
        pgf.get_mu_func(func_i, pars_i) if pars_i is not None else np.nan
        for func_i, pars_i in zip(fitted_gof_funcs, pk_pars)
    ]

    fig = plt.figure()
    derco = Polynomial(ecal_class.pars).deriv().coef
    der = [pgf.nb_poly(5, derco) for _ in fitted_peaks]
    for i, peak in enumerate(mus):
        range_adu = 5 / der[i]
        plt.subplot(nrows, ncols, i + 1)
        try:
            binning = np.arange(pk_ranges[i][0], pk_ranges[i][1], 0.1 / der[i])
            bin_cs = (binning[1:] + binning[:-1]) / 2
            energies = data.query(
                f"{ecal_class.energy_param}>{pk_ranges[i][0]}&{ecal_class.energy_param}<{pk_ranges[i][1]}&{ecal_class.selection_string}"
            )[ecal_class.energy_param]
            energies = energies.iloc[: ecal_class.n_events]

            counts, bs, bars = plt.hist(energies, bins=binning, histtype="step")
            if pk_pars[i] is not None:
                fit_vals = (
                    fitted_gof_funcs[i](bin_cs, *pk_pars[i][:-1], 0) * np.diff(bs)[0]
                )
                plt.plot(bin_cs, fit_vals)
                plt.step(
                    bin_cs,
                    [
                        (fval - count) / count if count != 0 else (fval - count)
                        for count, fval in zip(counts, fit_vals)
                    ],
                    where="mid",
                )

                plt.annotate(
                    get_peak_label(fitted_peaks[i]),
                    (0.02, 0.9),
                    xycoords="axes fraction",
                )
                plt.annotate(
                    f"{fitted_peaks[i]:.1f} keV", (0.02, 0.8), xycoords="axes fraction"
                )
                plt.annotate(
                    f"p-value : {p_vals[i]:.4f}", (0.02, 0.7), xycoords="axes fraction"
                )
                plt.xlabel("Energy (keV)")
                plt.ylabel("Counts")
                plt.legend(loc="upper left", frameon=False)
                plt.xlim([peak - range_adu, peak + range_adu])
                locs, labels = plt.xticks()
                new_locs, new_labels = get_peak_labels(locs, ecal_class.pars)
                plt.xticks(ticks=new_locs, labels=new_labels)
        except Exception:
            pass

    plt.tight_layout()
    plt.close()
    return fig


def plot_2614_timemap(
    ecal_class,
    data,
    figsize=(12, 8),
    fontsize=12,
    erange=(2580, 2630),
    dx=1,
    time_dx=180,
):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    selection = data.query(
        f"{ecal_class.cal_energy_param}>2560&{ecal_class.cal_energy_param}<2660&{ecal_class.selection_string}"
    )

    fig = plt.figure()
    if len(selection) == 0:
        pass
    else:
        time_bins = np.arange(
            (np.amin(data["timestamp"]) // time_dx) * time_dx,
            ((np.amax(data["timestamp"]) // time_dx) + 2) * time_dx,
            time_dx,
        )

        plt.hist2d(
            selection["timestamp"],
            selection[ecal_class.cal_energy_param],
            bins=[time_bins, np.arange(erange[0], erange[1] + dx, dx)],
            norm=LogNorm(),
        )

    ticks, labels = plt.xticks()
    plt.xlabel(
        f"Time starting : {datetime.utcfromtimestamp(ticks[0]).strftime('%d/%m/%y %H:%M')}"
    )
    plt.ylabel("Energy(keV)")
    plt.ylim([erange[0], erange[1]])

    plt.xticks(
        ticks,
        [datetime.utcfromtimestamp(tick).strftime("%H:%M") for tick in ticks],
    )
    plt.close()
    return fig


def plot_pulser_timemap(
    ecal_class,
    data,
    pulser_field="is_pulser",
    figsize=(12, 8),
    fontsize=12,
    dx=0.2,
    time_dx=180,
    n_spread=3,
):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    time_bins = np.arange(
        (np.amin(data["timestamp"]) // time_dx) * time_dx,
        ((np.amax(data["timestamp"]) // time_dx) + 2) * time_dx,
        time_dx,
    )

    selection = data.query(pulser_field)
    fig = plt.figure()
    if len(selection) == 0:
        pass

    else:
        mean = np.nanpercentile(selection[ecal_class.cal_energy_param], 50)
        spread = mean - np.nanpercentile(selection[ecal_class.cal_energy_param], 10)

        plt.hist2d(
            selection["timestamp"],
            selection[ecal_class.cal_energy_param],
            bins=[
                time_bins,
                np.arange(mean - n_spread * spread, mean + n_spread * spread + dx, dx),
            ],
            norm=LogNorm(),
        )
        plt.ylim([mean - n_spread * spread, mean + n_spread * spread])
    ticks, labels = plt.xticks()
    plt.xlabel(
        f"Time starting : {datetime.utcfromtimestamp(ticks[0]).strftime('%d/%m/%y %H:%M')}"
    )
    plt.ylabel("Energy(keV)")

    plt.xticks(
        ticks,
        [datetime.utcfromtimestamp(tick).strftime("%H:%M") for tick in ticks],
    )
    plt.close()
    return fig


def bin_pulser_stability(ecal_class, data, pulser_field="is_pulser", time_slice=180):
    selection = data.query(pulser_field)

    utime_array = data["timestamp"]
    select_energies = selection[ecal_class.cal_energy_param].to_numpy()

    time_bins = np.arange(
        (np.amin(utime_array) // time_slice) * time_slice,
        ((np.amax(utime_array) // time_slice) + 2) * time_slice,
        time_slice,
    )
    # bin time values
    times_average = (time_bins[:-1] + time_bins[1:]) / 2

    if len(selection) == 0:
        return {
            "time": times_average,
            "energy": np.full_like(times_average, np.nan),
            "spread": np.full_like(times_average, np.nan),
        }

    nanmedian = (
        lambda x: np.nanpercentile(x, 50) if len(x[~np.isnan(x)]) >= 10 else np.nan
    )
    error = (
        lambda x: np.nanvar(x) / np.sqrt(len(x))
        if len(x[~np.isnan(x)]) >= 10
        else np.nan
    )

    par_average, _, _ = binned_statistic(
        selection["timestamp"], select_energies, statistic=nanmedian, bins=time_bins
    )
    par_error, _, _ = binned_statistic(
        selection["timestamp"], select_energies, statistic=error, bins=time_bins
    )

    return {"time": times_average, "energy": par_average, "spread": par_error}


def bin_stability(ecal_class, data, time_slice=180, energy_range=(2585, 2660)):
    selection = data.query(
        f"{ecal_class.cal_energy_param}>{energy_range[0]}&{ecal_class.cal_energy_param}<{energy_range[1]}&{ecal_class.selection_string}"
    )

    utime_array = data["timestamp"]
    select_energies = selection[ecal_class.cal_energy_param].to_numpy()

    time_bins = np.arange(
        (np.amin(utime_array) // time_slice) * time_slice,
        ((np.amax(utime_array) // time_slice) + 2) * time_slice,
        time_slice,
    )
    # bin time values
    times_average = (time_bins[:-1] + time_bins[1:]) / 2

    if len(selection) == 0:
        return {
            "time": times_average,
            "energy": np.full_like(times_average, np.nan),
            "spread": np.full_like(times_average, np.nan),
        }

    nanmedian = (
        lambda x: np.nanpercentile(x, 50) if len(x[~np.isnan(x)]) >= 10 else np.nan
    )
    error = (
        lambda x: np.nanvar(x) / np.sqrt(len(x))
        if len(x[~np.isnan(x)]) >= 10
        else np.nan
    )

    par_average, _, _ = binned_statistic(
        selection["timestamp"], select_energies, statistic=nanmedian, bins=time_bins
    )
    par_error, _, _ = binned_statistic(
        selection["timestamp"], select_energies, statistic=error, bins=time_bins
    )

    return {"time": times_average, "energy": par_average, "spread": par_error}


def plot_cal_fit(ecal_class, data, figsize=(12, 8), fontsize=12, erange=(200, 2700)):
    valid_fits = ecal_class.results["pk_validities"]
    mus = ecal_class.results["pk_pos"]
    mu_errs = ecal_class.results["pk_pos_uncertainties"]
    fitted_peaks = ecal_class.results["got_peaks_keV"]

    fitted_gof_funcs = []
    for i, peak in enumerate(ecal_class.glines):
        if peak in fitted_peaks:
            fitted_gof_funcs.append(ecal_class.gof_funcs[i])

    fitted_gof_funcs = np.array(fitted_gof_funcs)[valid_fits]
    fitted_peaks = np.array(fitted_peaks)[valid_fits]

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    cal_bins = np.linspace(0, np.nanmax(mus) * 1.1, 20)

    ax1.scatter(fitted_peaks, mus, marker="x", c="b")

    ax1.plot(pgf.nb_poly(cal_bins, ecal_class.pars), cal_bins, lw=1, c="g")

    ax1.grid()
    ax1.set_xlim([erange[0], erange[1]])
    ax1.set_ylabel("Energy (ADC)")
    ax2.errorbar(
        fitted_peaks,
        pgf.nb_poly(np.array(mus), ecal_class.pars) - fitted_peaks,
        yerr=pgf.nb_poly(np.array(mus) + np.array(mu_errs), ecal_class.pars)
        - pgf.nb_poly(np.array(mus), ecal_class.pars),
        linestyle=" ",
        marker="x",
        c="b",
    )
    ax2.grid()
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Residuals (keV)")
    plt.close()
    return fig


def plot_eres_fit(ecal_class, data, erange=(200, 2700), figsize=(12, 8), fontsize=12):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fwhms = ecal_class.results["pk_fwhms"][:, 0]
    dfwhms = ecal_class.results["pk_fwhms"][:, 1]
    fitted_peaks = ecal_class.results["fitted_keV"]

    #####
    # Remove the Tl SEP and DEP from calibration if found
    fwhm_peaks = np.array([], dtype=np.float32)
    indexes = []
    for i, peak in enumerate(fitted_peaks):
        if peak == 2103.53:
            log.info(f"Tl SEP found at index {i}")
            indexes.append(i)
            continue
        elif peak == 1592.53:
            log.info(f"Tl DEP found at index {i}")
            indexes.append(i)
            continue
        elif np.isnan(dfwhms[i]):
            log.info(f"{peak} failed")
            indexes.append(i)
            continue
        elif peak == 511.0:
            log.info(f"e annihilation found at index {i}")
            indexes.append(i)
            continue
        else:
            fwhm_peaks = np.append(fwhm_peaks, peak)
    fit_fwhms = np.delete(fwhms, [indexes])
    fit_dfwhms = np.delete(dfwhms, [indexes])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    if len(np.where((~np.isnan(fit_fwhms)) & (~np.isnan(fit_dfwhms)))[0]) > 0:
        ax1.errorbar(
            fwhm_peaks, fit_fwhms, yerr=fit_dfwhms, marker="x", ls=" ", c="black"
        )

        fwhm_slope_bins = np.arange(erange[0], erange[1], 10)

        qbb_line_vx = [2039.0, 2039.0]
        qbb_line_vy = [
            0.9
            * np.nanmin(
                FWHMLinear.func(
                    fwhm_slope_bins, *ecal_class.fwhm_fit_linear["parameters"]
                )
            ),
            np.nanmax(
                [
                    ecal_class.fwhm_fit_linear["Qbb_fwhm_in_keV"],
                    ecal_class.fwhm_fit_quadratic["Qbb_fwhm_in_keV"],
                ]
            ),
        ]
        qbb_line_hx = [erange[0], 2039.0]

        ax1.plot(
            fwhm_slope_bins,
            FWHMLinear.func(fwhm_slope_bins, *ecal_class.fwhm_fit_linear["parameters"]),
            lw=1,
            c="g",
            label=f'linear, Qbb fwhm: {ecal_class.fwhm_fit_linear["Qbb_fwhm_in_keV"]:1.2f} +- {ecal_class.fwhm_fit_linear["Qbb_fwhm_err_in_keV"]:1.2f} keV',
        )
        ax1.plot(
            fwhm_slope_bins,
            FWHMQuadratic.func(
                fwhm_slope_bins, *ecal_class.fwhm_fit_quadratic["parameters"]
            ),
            lw=1,
            c="b",
            label=f'quadratic, Qbb fwhm: {ecal_class.fwhm_fit_quadratic["Qbb_fwhm_in_keV"]:1.2f} +- {ecal_class.fwhm_fit_quadratic["Qbb_fwhm_err_in_keV"]:1.2f} keV',
        )
        ax1.plot(
            qbb_line_hx,
            [
                ecal_class.fwhm_fit_linear["Qbb_fwhm_in_keV"],
                ecal_class.fwhm_fit_linear["Qbb_fwhm_in_keV"],
            ],
            lw=1,
            c="r",
            ls="--",
        )
        ax1.plot(
            qbb_line_hx,
            [
                ecal_class.fwhm_fit_quadratic["Qbb_fwhm_in_keV"],
                ecal_class.fwhm_fit_quadratic["Qbb_fwhm_in_keV"],
            ],
            lw=1,
            c="r",
            ls="--",
        )
        ax1.plot(qbb_line_vx, qbb_line_vy, lw=1, c="r", ls="--")

        ax1.legend(loc="upper left", frameon=False)
        if np.isnan(ecal_class.fwhm_fit_linear["parameters"]).all():
            ax1.set_ylim(
                [
                    0.9 * np.nanmin(fit_fwhms),
                    1.1 * np.nanmax(fit_fwhms),
                ]
            )
        else:
            ax1.set_ylim(
                [
                    0.9
                    * np.nanmin(
                        FWHMLinear.func(
                            fwhm_slope_bins, *ecal_class.fwhm_fit_linear["parameters"]
                        )
                    ),
                    1.1
                    * np.nanmax(
                        FWHMLinear.func(
                            fwhm_slope_bins, *ecal_class.fwhm_fit_linear["parameters"]
                        )
                    ),
                ]
            )
        ax1.set_xlim(erange)
        ax1.set_ylabel("FWHM energy resolution (keV)")
        ax2.plot(
            fwhm_peaks,
            (
                fit_fwhms
                - FWHMLinear.func(fwhm_peaks, *ecal_class.fwhm_fit_linear["parameters"])
            )
            / fit_dfwhms,
            lw=0,
            marker="x",
            c="g",
        )
        ax2.plot(
            fwhm_peaks,
            (
                fit_fwhms
                - FWHMQuadratic.func(
                    fwhm_peaks, *ecal_class.fwhm_fit_quadratic["parameters"]
                )
            )
            / fit_dfwhms,
            lw=0,
            marker="x",
            c="b",
        )
        ax2.plot(erange, [0, 0], color="black", lw=0.5)
        ax2.set_xlabel("Energy (keV)")
        ax2.set_ylabel("Normalised Residuals")
    plt.tight_layout()
    plt.close()
    return fig


def bin_spectrum(
    ecal_class,
    data,
    cut_field="is_valid_cal",
    pulser_field="is_pulser",
    erange=(0, 3000),
    dx=2,
):
    bins = np.arange(erange[0], erange[1] + dx, dx)
    return {
        "bins": pgh.get_bin_centers(bins),
        "counts": np.histogram(
            data.query(ecal_class.selection_string)[ecal_class.cal_energy_param], bins
        )[0],
        "cut_counts": np.histogram(
            data.query(f"(~{cut_field})&(~{pulser_field})")[
                ecal_class.cal_energy_param
            ],
            bins,
        )[0],
        "pulser_counts": np.histogram(
            data.query(pulser_field)[ecal_class.cal_energy_param],
            bins,
        )[0],
    }


def bin_survival_fraction(
    ecal_class,
    data,
    cut_field="is_valid_cal",
    pulser_field="is_pulser",
    erange=(0, 3000),
    dx=6,
):
    counts_pass, bins_pass, _ = pgh.get_hist(
        data.query(ecal_class.selection_string)[ecal_class.cal_energy_param],
        bins=np.arange(erange[0], erange[1] + dx, dx),
    )
    counts_fail, bins_fail, _ = pgh.get_hist(
        data.query(f"(~{cut_field})&(~{pulser_field})")[ecal_class.cal_energy_param],
        bins=np.arange(erange[0], erange[1] + dx, dx),
    )
    sf = 100 * (counts_pass + 10 ** (-6)) / (counts_pass + counts_fail + 10 ** (-6))
    return {"bins": pgh.get_bin_centers(bins_pass), "sf": sf}

