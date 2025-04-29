"""routines for automatic calibration.

- hpge_find_energy_peaks (Find uncalibrated E peaks whose E spacing matches the pattern in peaks_kev)
- hpge_get_energy_peaks (Get uncalibrated E peaks at the energies of peaks_kev)
- hpge_fit_energy_peaks (fits the energy peals)
- hpge_E_calibration (main routine -- finds and fits peaks specified)
"""

from __future__ import annotations

import inspect
import logging
import string

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from iminuit import Minuit, cost
from iminuit.util import ValueView
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import chi2

import pygama.math.binned_fitting as pgb
import pygama.math.distributions as pgf
import pygama.math.histogram as pgh
from pygama.math.histogram import get_i_local_maxima
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
        Otherwise follows the convention in np.polynomial.polynomial of
        lowest order to highest order
    uncal_is_int : bool
        if True, attempts will be made to avoid picket-fencing when binning
        e_uncal
    fixed : dict
        dictionary of fixed parameters for the calibration function

    """

    def __init__(
        self,
        energy_param,
        glines,
        guess_kev: float,
        deg: int = 1,
        uncal_is_int: bool = False,
        fixed=None,
        debug_mode: bool = False,
    ):
        self.energy_param = energy_param

        if deg < -1:
            log.error(f"hpge_E_cal warning: invalid deg = {deg}")
            return
        self.deg = int(deg)

        self.peaks_kev = np.asarray(sorted(glines))
        self.peak_locs = []

        if guess_kev <= 0:
            log.error(f"hpge_E_cal warning: invalid guess_kev = {guess_kev}")
        if deg == -1:
            self.pars = np.zeros(2, dtype=float)
            self.pars[0] = guess_kev
            self.fixed = {1: 1}
        elif deg == 0:
            self.pars = np.zeros(2, dtype=float)
            self.pars[1] = guess_kev
            self.fixed = {0: 0}
        else:
            self.pars = np.zeros(self.deg + 1, dtype=float)
            self.pars[1] = guess_kev
            self.fixed = fixed
        self.results = {}

        self.uncal_is_int = uncal_is_int
        self.debug_mode = debug_mode

    def gen_pars_dict(self):
        """
        Generate a dictionary containing the expression and parameters used for energy calibration.

        Returns:
            dict: A dictionary with keys 'expression' and 'parameters'.
                  'expression' is a string representing the energy calibration expression.
                  'parameters' is a dictionary containing the parameter values used in the expression.
        """
        expression = ""
        parameters = {}
        for i, coeff in enumerate(self.pars):
            parameter_name = string.ascii_lowercase[i]
            if i == 0:
                expression += f"{parameter_name}"
            elif i == 1:
                expression += f" + {parameter_name} * {self.energy_param}"
            else:
                expression += f" + {parameter_name} * {self.energy_param}**{i} "
            parameters[parameter_name] = coeff
        return {"expression": expression, "parameters": parameters}

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
        """
        Find uncalibrated energy peaks whose energy spacing matches the pattern in peaks_kev.

        Parameters
        ----------
        e_uncal (array-like):
            Uncalibrated energy values.
        peaks_kev (array-like, optional):
            Pattern of energy peaks to match. If not provided, the peaks from the object's attribute `peaks_kev` will be used.
        n_sigma (float, optional):
             Number of standard deviations above the mean to consider a peak significant. Default is 5.
        etol_kev (float, optional):
            Tolerance in energy units for matching peaks. If not provided, it will be estimated based on the peak widths.
        bin_width_kev (float, optional):
            Width of the energy bins for initial peak search. Default is 1 keV.
        erange (tuple, optional):
            Range of uncalibrated energy values to consider. If not provided, the range will be determined based on the peaks.
        var_zero (float, optional):
            Value to replace zero variance with. Default is 1.
        update_cal_pars (bool, optional):
            Whether to update the calibration parameters. Default is True.

        Returns:
            None

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
                "found_peaks_locs": detected_max_locs[ixtup],
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
        e_uncal : array
            Uncalibrated energy values.
        peaks_kev : array, optional
            Energies of peaks to search for (in keV). If not provided, the peaks_kev
            attribute of the object will be used.
        n_sigma : float, optional
            Threshold for detecting a peak in sigma (i.e. sqrt(var)). Default is 3.
        etol_kev : float, optional
            Absolute tolerance in energy for matching peaks. Default is 5.
        var_zero : float, optional
            Number used to replace zeros of var to avoid divide-by-zero in hist/sqrt(var).
            Default is 1. Usually when var = 0, it's because hist = 0, and any value here is fine.
        bin_width_kev : float, optional
            Width of the energy bins for re-binning the histogram. Default is 0.2 keV.
        update_cal_pars : bool, optional
            Flag indicating whether to update the calibration parameters. Default is True.
        erange : tuple, optional
            Range of energy values to consider for peak search. If not provided, the range
            will be determined automatically based on the peaks_kev values.

        Returns
        -------
        None

        Notes
        -----
        This method performs the following steps:
        1. Re-bins the histogram in ~0.2 keV bins with updated energy scale parameters for peak-top fits.
        2. Finds all local maxima in the histogram with significance greater than n_sigma.
        3. Matches the calculated peak energies with the expected peak energies.
        4. Removes duplicate peak matches.
        5. Updates the input peaks, got peaks, and got peak locations in the results dictionary.
        6. If update_cal_pars is True, calculates the updated calibration curve using the matched peak energies.

        """
        if peaks_kev is None:
            peaks_kev = self.peaks_kev

        peaks_kev = np.asarray(peaks_kev)

        # re-bin the histogram in ~0.2 kev bins with updated E scale par for peak-top fits
        if erange is None:
            euc_min, euc_max = (
                (Polynomial(self.pars) - i).roots()
                for i in (peaks_kev[0] * 0.9, peaks_kev[-1] * 1.1)
            )
            euc_min = euc_min[0]
            euc_max = euc_max[0]
            if euc_min < 0:
                euc_min = 0
            if euc_max > np.nanmax(e_uncal) * 1.1:
                euc_max = np.nanmax(e_uncal) * 1.1
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
                "got_peaks_kev": matched_energies,
                "got_peaks_locs": got_peak_locations,
            }
        )

        if update_cal_pars is False:
            return

        self.peak_locs = got_peak_locations
        self.peaks_kev = matched_energies

        # Calculate updated calibration curve
        if self.deg == 0:
            scale, _ = fit_simple_scaling(got_peak_locations, matched_energies)
            poly_pars = np.array([0, scale])
        else:
            # Calculate updated calibration curve
            poly_pars = (
                Polynomial.fit(got_peak_locations, matched_energies, len(self.pars) - 1)
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

        self.pars = np.array(m.values)

        log.info(f"{len(self.peak_locs)} peaks obtained:")
        log.info("\t   Energy   | Position  ")
        for i, (li, ei) in enumerate(zip(self.peak_locs, self.peaks_kev)):
            log.info(f"\t{i}".ljust(4) + str(ei).ljust(9) + f"| {li:g}".ljust(5))

    def hpge_cal_energy_peak_tops(
        self,
        e_uncal,
        n_sigmas=1.2,
        peaks_kev=None,
        default_n_bins=50,
        n_events=None,
        allowed_p_val=0.01,
        update_cal_pars=True,
    ):
        """
        Perform energy calibration for HPGe detector using peak fitting.

        Args:
            e_uncal (array-like):
                Uncalibrated energy values.
            n_sigmas (float, optional):
                Number of standard deviations to use for peak fitting range. Defaults to 1.2.
            peaks_kev (array-like, optional):
                Known peak positions in keV. If not provided, uses self.peaks_kev. Defaults to None.
            default_n_bins (int, optional):
                    Number of bins for histogram. Defaults to 50.
            n_events (int, optional):
                Number of events to consider for calibration. Defaults to None which uses all events.
            allowed_p_val (float, optional):
                Maximum p-value for a fit to be considered valid. Defaults to 0.05.
            update_cal_pars (bool, optional):
                Whether to update the calibration parameters. Defaults to True.
        """

        results_dict = {}

        # check no peaks in self.peaks_kev missing from peak_pars
        if peaks_kev is None:
            peaks_kev = self.peaks_kev

        peak_pars = [(peak, None, pgf.gauss_on_uniform) for peak in peaks_kev]

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
            [True for i in peak_pars if i[2] is not None],
            dtype=bool,
        )
        peak_pars = peak_pars[fit_peaks_mask]

        # First calculate range around peaks to fit

        euc_min, euc_max = (
            (Polynomial(self.pars) - i).roots()
            for i in (peaks_kev[0] * 0.9, peaks_kev[-1] * 1.1)
        )
        euc_min = np.nanmin(euc_min)
        euc_max = np.nanmax(euc_max)

        if euc_min < 0:
            euc_min = 0
        if euc_max > np.nanmax(e_uncal) * 1.1:
            euc_max = np.nanmax(e_uncal) * 1.1

        d_euc = 0.5 / self.pars[1]
        if self.uncal_is_int:
            euc_min, euc_max, d_euc = pgh.better_int_binning(
                x_lo=euc_min, x_hi=euc_max, dx=d_euc
            )

        hist, bins, var = pgh.get_hist(e_uncal, range=(euc_min, euc_max), dx=d_euc)

        uncal_peak_pars = []
        for pars in peak_pars:
            peak, fit_range, func = pars

            if peak in self.peaks_kev:
                loc = self.peak_locs[np.where(peak == self.peaks_kev)][0]
            else:
                loc = (Polynomial(self.pars) - peak).roots()[0]

            # Need to do initial fit
            pt_pars, _ = hpge_fit_energy_peak_tops(hist, bins, var, [loc], n_to_fit=7)
            # Drop failed fits
            if pt_pars[0] is not None:
                range_uncal = (float(pt_pars[0][1]) * 20, float(pt_pars[0][1]) * 20)
                n_bins = default_n_bins
            else:
                range_uncal = None
            if range_uncal is not None:
                uncal_peak_pars.append((peak, loc, range_uncal, n_bins, func))

        fit_dict = {}

        for i_peak, uncal_peak_par in enumerate(uncal_peak_pars):
            try:
                peak_kev, mode_guess, wwidth_i, n_bins_i, func_i = uncal_peak_par
                wleft_i, wright_i = wwidth_i
                euc_min = mode_guess - wleft_i
                euc_max = mode_guess + wright_i

                if self.uncal_is_int is True:
                    euc_min, euc_max, n_bins_i = pgh.better_int_binning(
                        x_lo=euc_min, x_hi=euc_max, n_bins=n_bins_i
                    )

                energies = e_uncal[(e_uncal > euc_min) & (e_uncal < euc_max)][:n_events]
                binw_1 = (euc_max - euc_min) / n_bins_i

                x0 = get_hpge_energy_peak_par_guess(
                    energies,
                    func_i,
                    (euc_min, euc_max),
                    bin_width=binw_1,
                    mode_guess=mode_guess,
                )

                euc_min = x0["mu"] - n_sigmas * x0["sigma"]
                euc_max = x0["mu"] + n_sigmas * x0["sigma"]

                bin_width = (x0["sigma"]) * len(energies) ** (-1 / 3)
                n_bins_i = int((euc_max - euc_min) / bin_width)

                if self.uncal_is_int is True:
                    euc_min, euc_max, n_bins_i = pgh.better_int_binning(
                        x_lo=euc_min, x_hi=euc_max, n_bins=n_bins_i
                    )

                hist, bins, var = pgh.get_hist(
                    energies, bins=n_bins_i, range=(euc_min, euc_max)
                )

                x0["x_lo"] = euc_min
                x0["x_hi"] = euc_max

                fixed, mask = get_hpge_energy_fixed(func_i)
                fixed.append("n_bkg")
                mask[np.where(np.array(func_i.required_args()) == "n_bkg")[0]] = True
                bounds = get_hpge_energy_bounds(func_i, x0)

                pars_i, errs_i, cov_i = pgb.fit_binned(
                    func_i.cdf_ext,
                    hist,
                    bins,
                    var=var,
                    guess=x0,
                    cost_func="LL",
                    extended=True,
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
                    scale_bins=True,
                )
                csqr = (csqr[0], csqr[1] + len(np.where(mask)[0]))

                if np.isnan(pars_i).any():
                    log.debug(
                        f"hpge_cal_energy_peak_tops: fit failed for i_peak={i_peak} at loc {mode_guess:g}, par is nan : {pars_i}"
                    )
                    raise RuntimeError

                p_val = scipy.stats.chi2.sf(csqr[0], csqr[1])

                if (
                    cov_i is None
                    or cov_i.ndim == 0
                    or sum(sum(c) for c in cov_i[mask, :][:, mask]) == np.inf
                    or sum(sum(c) for c in cov_i[mask, :][:, mask]) == 0
                    or np.isnan(sum(sum(c) for c in cov_i[mask, :][:, mask]))
                ):
                    log.debug(
                        f"hpge_cal_energy_peak_tops: cov estimation failed for i_peak={i_peak} at loc {mode_guess:g}"
                    )
                    valid_pk = False

                elif valid_fit is False:
                    log.debug(
                        f"hpge_cal_energy_peak_tops: peak fitting failed for i_peak={i_peak} at loc {mode_guess:g}"
                    )
                    valid_pk = False

                elif (
                    errs_i is None
                    or pars_i is None
                    or np.abs(np.array(errs_i)[mask] / np.array(pars_i)[mask]) < 1e-7
                ).any() or np.isnan(np.array(errs_i)[mask]).any():
                    log.debug(
                        f"hpge_cal_energy_peak_tops: failed for i_peak={i_peak} at loc {mode_guess:g}, parameter error too low"
                    )
                    valid_pk = False

                elif p_val < allowed_p_val or np.isnan(p_val):
                    log.debug(
                        f"hpge_cal_energy_peak_tops: fit failed for i_peak={i_peak}, p-value too low: {p_val}"
                    )
                    valid_pk = False
                else:
                    valid_pk = True

                mu, mu_err = func_i.get_mu(pars_i, errors=errs_i)

            except BaseException as e:
                if e == KeyboardInterrupt:
                    raise (e)
                elif self.debug_mode:
                    raise (e)
                log.debug(
                    f"hpge_cal_energy_peak_tops: fit failed for i_peak={i_peak}, unknown error"
                )
                valid_pk = False
                pars_i, errs_i, cov_i = return_nans(func_i)
                p_val = 0
                mu = np.nan
                mu_err = np.nan

            fit_dict[peak_kev] = {
                "function": func_i,
                "validity": valid_pk,
                "parameters": pars_i,
                "uncertainties": errs_i,
                "covariance": cov_i,
                "bin_width": binw_1,
                "range": [euc_min, euc_max],
                "p_value": p_val,
                "position": mu,
                "position_uncertainty": mu_err,
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

        if len(fitted_peaks_kev) == 0:
            log.error("hpge_fit_energy_peak_tops: no peaks fitted")
            self.update_results_dict(results_dict)
            return

        mus = [
            fit_dict[peak]["position"]
            for peak in fit_dict
            if fit_dict[peak]["validity"]
        ]
        mu_vars = [
            fit_dict[peak]["position_uncertainty"]
            for peak in fit_dict
            if fit_dict[peak]["validity"]
        ]

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
            results_dict["calibration_parameters"] = pars
            results_dict["calibration_uncertainties"] = errs
            results_dict["calibration_covariance"] = cov

        except ValueError:
            log.error("Failed to fit enough peaks to get accurate calibration")

        self.update_results_dict(results_dict)

    def hpge_fit_energy_peaks(
        self,
        e_uncal,
        peak_pars=None,
        peaks_kev=None,
        bin_width_kev=1,
        peak_param="mode",
        method="unbinned",
        n_events=None,
        allowed_p_val=0.01,
        tail_weight=0,
        update_cal_pars=True,
        use_bin_width_in_fit=True,
    ):
        """
        Fit the energy peaks specified using the given function.

        Parameters
        ----------
        e_uncal : array
            Unbinned energy data to be fit.
        peaks_kev : array, optional
            Array of energy values for the peaks to fit. If not provided, it uses the peaks_kev attribute of the class.
        peak_pars : list of tuples, optional
            List containing tuples of the form (peak, range, func) where peak is the energy of the peak to fit,
            range is the range in keV to fit, and func is the function to fit.
        bin_width_kev : int, optional
            Default binwidth to use for the fit window histogramming. Default is 1 keV.
        peak_param : str, optional
            Parameter to use for peak fitting. Default is "mode".
        method : str, optional
            Method to use for fitting. Default is "unbinned". Can specify to use binned fit method instead.
        n_events : int, optional
            Number of events to use for unbinned fit.
        allowed_p_val : float, optional
            Lower limit on p-value of fit.
        tail_weight : int, optional
            Weight to apply to the tail of the fit.
        update_cal_pars : bool, optional
            Whether to update the calibration parameters. Default is True.

        Returns
        -------
        results_dict : dict
            Dictionary containing the fit results for each peak.

        Raises
        ------
        RuntimeError
            If the fit fails.

        Notes
        -----
        This function fits the energy peaks specified using the given function. It calculates the range around each peak to fit,
        performs the fitting using either unbinned or binned method, and returns the fit results in a dictionary.

        """

        results_dict = {}
        # check no peaks in self.peaks_kev missing from peak_pars

        if peaks_kev is None:
            peaks_kev = self.peaks_kev

        if peak_pars is None:
            peak_pars = [(peak, None, pgf.gauss_on_step) for peak in peaks_kev]

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
                euc_min = np.nanmin(euc_min)
                euc_max = np.nanmax(euc_max)
                if euc_min < 0:
                    euc_min = 0
                if euc_max > np.nanmax(e_uncal) * 1.1:
                    euc_max = np.nanmax(e_uncal) * 1.1
                d_euc = 0.5 / self.pars[1]
                if self.uncal_is_int:
                    euc_min, euc_max, d_euc = pgh.better_int_binning(
                        x_lo=euc_min, x_hi=euc_max, dx=d_euc
                    )
                hist, bins, var = pgh.get_hist(
                    e_uncal, range=(euc_min, euc_max), dx=d_euc
                )
                # Need to do initial fit
                pt_pars, _ = hpge_fit_energy_peak_tops(
                    hist, bins, var, [loc], n_to_fit=7
                )
                # Drop failed fits
                if pt_pars[0] is not None:
                    range_uncal = (float(pt_pars[0][1]) * 20, float(pt_pars[0][1]) * 20)
                    n_bins = int(range_uncal / bin_width_kev)
                else:
                    range_uncal = None
            elif isinstance(fit_range, tuple):
                der = pgf.nb_poly(peak, derco)
                range_uncal = (fit_range[0] / der, fit_range[1] / der)
                n_bins = int(sum(fit_range) / (bin_width_kev))
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
                        tail_weight=tail_weight,
                        bin_width=binw_1 if use_bin_width_in_fit is True else None,
                        guess_kwargs={"mode_guess": mode_guess},
                        p_val_threshold=allowed_p_val,
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
                        extended=True,
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
                        scale_bins=True,
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

                elif (p_val < allowed_p_val and (csqr[0] / csqr[1]) > 10) or np.isnan(
                    p_val
                ):
                    log.debug(
                        f"hpge_fit_energy_peaks: fit failed for i_peak={i_peak}, p-value too low: {p_val}"
                    )
                    valid_pk = False
                else:
                    valid_pk = True

                if peak_param == "mu":
                    mu, mu_err = func_i.get_mu(pars_i, errors=errs_i)

                elif peak_param == "mode":
                    mu, mu_err = func_i.get_mode(pars_i, cov=cov_i)
                else:
                    log.error(
                        f"hpge_fit_energy_peaks: mode {self.peak_param} not recognized"
                    )
                    raise RuntimeError

            except BaseException as e:
                if e == KeyboardInterrupt:
                    raise (e)
                elif self.debug_mode:
                    raise (e)
                log.debug(
                    f"hpge_fit_energy_peaks: fit failed for i_peak={i_peak}, unknown error"
                )
                valid_pk = False
                pars_i, errs_i, cov_i = return_nans(func_i)
                p_val = 0
                mu = np.nan
                mu_err = np.nan
                csqr = (np.nan, np.nan)

            fit_dict[peak_kev] = {
                "function": func_i,
                "validity": valid_pk,
                "parameters": pars_i,
                "uncertainties": errs_i,
                "covariance": cov_i,
                "bin_width": binw_1,
                "range": [euc_min, euc_max],
                "chi_square": csqr,
                "p_value": p_val,
                "position": mu,
                "position_uncertainty": mu_err,
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

        if len(fitted_peaks_kev) == 0:
            log.error("hpge_fit_energy_peaks: no peaks fitted")
            self.update_results_dict(results_dict)
            return

        mus = [
            fit_dict[peak]["position"]
            for peak in fit_dict
            if fit_dict[peak]["validity"]
        ]
        mu_vars = [
            fit_dict[peak]["position_uncertainty"]
            for peak in fit_dict
            if fit_dict[peak]["validity"]
        ]

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
            results_dict["calibration_parameters"] = pars
            results_dict["calibration_uncertainties"] = errs
            results_dict["calibration_covariance"] = cov

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
            if peak_dict["validity"] is True:
                uncal_fwhm, uncal_fwhm_err = peak_dict["function"].get_fwhm(
                    peak_dict["parameters"],
                    cov=peak_dict["covariance"],
                )
            else:
                uncal_fwhm, uncal_fwhm_err = (np.nan, np.nan)

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

    @staticmethod
    def fit_energy_res_curve(fwhm_func, fwhm_peaks, fwhms, dfwhms):
        try:
            if len(fwhm_peaks) == 0:
                raise RuntimeError
            c_lin = cost.LeastSquares(fwhm_peaks, fwhms, dfwhms, fwhm_func.func)
            # c_lin.loss = "soft_l1"
            m = Minuit(c_lin, *fwhm_func.guess(fwhm_peaks, fwhms, dfwhms))
            bounds = fwhm_func.bounds(fwhms)
            for arg, val in enumerate(bounds):
                m.limits[arg] = val
            m.simplex()
            m.migrad()
            m.hesse()

            p_val = scipy.stats.chi2.sf(m.fval, len(fwhm_peaks) - len(m.values))

            results = {
                "function": fwhm_func,
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
                "function": fwhm_func,
                "module": fwhm_func.__module__,
                "expression": fwhm_func.string_func("x"),
                "parameters": pars,
                "uncertainties": errs,
                "cov": cov,
                "csqr": (np.nan, np.nan),
                "p_val": 0,
            }
            log.error("FWHM fit failed to converge")
        return results

    @staticmethod
    def interpolate_energy_res(
        fwhm_func, fwhm_peaks, fwhm_results, interp_energy_kev=None, debug_mode=False
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
                except BaseException as e:
                    if debug_mode:
                        raise (e)
                    interp_fwhm = np.nan
                    interp_err = np.nan
                fwhm_results.update(
                    {
                        "interp_energy_in_kev": energy,
                        f"{key}_fwhm_in_kev": interp_fwhm,
                        f"{key}_fwhm_err_in_kev": interp_err,
                    }
                )
                log.info(
                    f"FWHM {key} energy resolution at {energy} : {interp_fwhm:1.2f} +- {interp_err:1.2f} kev"
                )
        return fwhm_results

    def get_energy_res_curve(self, fwhm_func, interp_energy_kev=None):
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
        if len(fitted_peaks_kev) == 0:
            return
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
            elif np.abs(peak - 1592.53) < 1:
                log.info("Tl DEP removed from fwhm fitting")
            elif np.abs(peak - 511.0) < 1:
                log.info("e annihilation removed from fwhm fitting")
            elif np.isnan(peak_dict["fwhm_in_kev"]) or np.isnan(
                peak_dict["fwhm_err_in_kev"]
            ):
                log.info(f"{peak} failed, removed from fwhm fitting")
            else:
                fwhm_peaks = np.append(fwhm_peaks, peak)
                fwhms = np.append(fwhms, peak_dict["fwhm_in_kev"])
                dfwhms = np.append(dfwhms, peak_dict["fwhm_err_in_kev"])

        log.info(f"Running FWHM fit for : {fwhm_func.__name__}")

        results = self.fit_energy_res_curve(fwhm_func, fwhm_peaks, fwhms, dfwhms)
        if interp_energy_kev is not None:
            results = self.interpolate_energy_res(
                fwhm_func,
                fwhm_peaks,
                results,
                interp_energy_kev,
                debug_mode=self.debug_mode,
            )
        self.results[list(self.results)[-1]].update({fwhm_func.__name__: results})

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
                except BaseException as e:
                    if self.debug_mode:
                        raise (e)

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
                if self.deg < 1:
                    self.pars = np.full(2, np.nan)
                else:
                    self.pars = np.full(self.deg + 1, np.nan)

                log.error(f"Calibration failed completely for {self.energy_param}")
                return

        log.debug("Calibrated found")
        log.info(f"Calibration pars are {self.pars}")

        self.get_energy_res_curve(
            FWHMLinear,
            interp_energy_kev={"Qbb": 2039.0},
        )
        self.get_energy_res_curve(
            FWHMQuadratic,
            interp_energy_kev={"Qbb": 2039.0},
        )

    def fit_calibrated_peaks(self, e_uncal, peak_pars):
        log.debug(f"Fitting {self.energy_param}")
        self.hpge_get_energy_peaks(e_uncal, update_cal_pars=False)
        self.hpge_fit_energy_peaks(e_uncal, peak_pars=peak_pars, update_cal_pars=False)
        self.get_energy_res_curve(
            FWHMLinear,
            interp_energy_kev={"Qbb": 2039.0},
        )
        self.get_energy_res_curve(
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
        self.get_energy_res_curve(
            FWHMLinear,
            interp_energy_kev={"Qbb": 2039.0},
        )
        self.get_energy_res_curve(
            FWHMQuadratic,
            interp_energy_kev={"Qbb": 2039.0},
        )

    def plot_cal_fit(self, data, figsize=(12, 8), fontsize=12, erange=(200, 2700)):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, figsize=figsize
        )

        cal_bins = np.linspace(0, np.nanmax(self.peak_locs) * 1.1, 20)

        ax1.scatter(self.peaks_kev, self.peak_locs, marker="x", c="b")

        ax1.plot(pgf.nb_poly(cal_bins, self.pars), cal_bins, lw=1, c="g")

        ax1.grid()
        ax1.set_xlim([erange[0], erange[1]])
        ax1.set_ylabel("Energy (ADC)", fontsize=fontsize)
        ax2.scatter(
            self.peaks_kev,
            pgf.nb_poly(np.array(self.peak_locs), self.pars) - self.peaks_kev,
            marker="x",
            c="b",
        )
        ax2.grid()
        ax2.set_xlabel("Energy (keV)", fontsize=fontsize)
        ax2.set_ylabel("Residuals (keV)", fontsize=fontsize)
        plt.close()
        return fig

    def plot_cal_fit_with_errors(
        self, data, figsize=(10, 6), fontsize=12, erange=(200, 2700)
    ):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, figsize=figsize
        )
        pk_parameters = self.results[list(self.results)[-1]].get(
            "peak_parameters", None
        )

        cal_bins = np.linspace(0, np.nanmax(self.peak_locs) * 1.1, 20)

        ax1.errorbar(
            self.peaks_kev,
            self.peak_locs,
            yerr=[
                pk_dict["position_uncertainty"]
                for pk_dict in pk_parameters.values()
                if pk_dict["validity"]
            ],
            linestyle="",
            marker="x",
            c="b",
        )

        ax1.plot(pgf.nb_poly(cal_bins, self.pars), cal_bins, lw=1, c="g")

        ax1.grid()
        ax1.set_xlim([erange[0], erange[1]])
        ax1.set_ylabel("Energy (ADC)", fontsize=fontsize)

        reses = pgf.nb_poly(np.array(self.peak_locs), self.pars) - self.peaks_kev
        res_errs = (
            np.array(
                [
                    pgf.nb_poly(
                        np.array(
                            [pk_dict["position"] + pk_dict["position_uncertainty"]]
                        ),
                        self.pars,
                    )[0]
                    for pk_dict in pk_parameters.values()
                    if pk_dict["validity"]
                ]
            )
            - self.peaks_kev
        )
        res_errs -= reses

        ax2.errorbar(
            self.peaks_kev,
            pgf.nb_poly(np.array(self.peak_locs), self.pars) - self.peaks_kev,
            yerr=res_errs,
            linestyle="",
            marker="x",
            c="b",
        )
        ax2.fill_between([erange[0], erange[1]], -0.1, 0.1, color="green", alpha=0.2)
        ax2.fill_between([erange[0], erange[1]], -0.5, 0.5, color="yellow", alpha=0.2)
        ax2.set_ylim([-1, 1])
        ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax2.grid()
        ax2.set_xlabel("Energy (keV)", fontsize=fontsize)
        ax2.set_ylabel("Residuals (keV)", fontsize=fontsize)
        plt.tight_layout()
        plt.close()
        return fig

    def plot_fits(
        self, energies, figsize=(12, 8), fontsize=12, ncols=3, nrows=3, binning_kev=5
    ):
        plt.rcParams["font.size"] = fontsize

        pk_parameters = self.results[list(self.results)[-1]].get(
            "peak_parameters", None
        )

        fig = plt.figure(figsize=figsize)
        derco = Polynomial(self.pars).deriv().coef
        der = [pgf.nb_poly(5, derco) for _ in list(pk_parameters)]
        for i, peak in enumerate(pk_parameters):
            range_adu = 5 / der[i]
            plt.subplot(nrows, ncols, i + 1)
            pk_dict = pk_parameters[peak]
            pk_pars = pk_dict["parameters"]
            pk_ranges = pk_dict["range"]
            pk_func = pk_dict["function"]
            mu = pk_func.get_mu(pk_pars) if pk_pars is not None else np.nan

            try:
                binning = np.arange(pk_ranges[0], pk_ranges[1], 0.1 / der[i])
                bin_cs = (binning[1:] + binning[:-1]) / 2

                counts, bs, bars = plt.hist(energies, bins=binning, histtype="step")
                if pk_pars is not None:
                    fit_vals = pk_func.get_pdf(bin_cs, *pk_pars, 0) * np.diff(bs)[0]
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
                        f"{peak:.1f} keV", (0.02, 0.8), xycoords="axes fraction"
                    )
                    plt.annotate(
                        f"p-value : {pk_dict['p_value']:.4f}",
                        (0.02, 0.7),
                        xycoords="axes fraction",
                    )
                    plt.xlabel("Energy (keV)")
                    plt.ylabel("Counts")

                    plt.xlim([mu - range_adu, mu + range_adu])
                    locs, labels = plt.xticks()

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

                    new_locs, new_labels = get_peak_labels(locs, self.pars)
                    plt.xticks(ticks=new_locs, labels=new_labels)

            except BaseException as e:
                if self.debug_mode:
                    raise (e)

        plt.tight_layout()
        plt.close()
        return fig

    def plot_eres_fit(self, data, erange=(200, 2700), figsize=(12, 8), fontsize=12):
        plt.rcParams["font.size"] = fontsize

        pk_parameters = self.results[list(self.results)[-1]].get(
            "peak_parameters", None
        )

        if pk_parameters is None:
            fig = plt.figure()
            return fig

        #####
        # Remove the Tl SEP and DEP from calibration if found
        fwhm_peaks = np.array([], dtype=np.float32)
        fwhms = np.array([], dtype=np.float32)
        dfwhms = np.array([], dtype=np.float32)

        for peak, pk_dict in pk_parameters.items():
            if peak == 2103.53:
                pass
            elif peak == 1592.53:
                pass
            elif peak == 511.0:
                pass
            elif pk_dict["validity"] is False:
                pass
            elif np.isnan(pk_dict["fwhm_err_in_kev"]):
                pass
            else:
                fwhm_peaks = np.append(fwhm_peaks, peak)
                fwhms = np.append(fwhms, pk_dict["fwhm_in_kev"])
                dfwhms = np.append(dfwhms, pk_dict["fwhm_err_in_kev"])

        fwhm_dicts = {}
        interp_energy = None
        interp_fwhm_name = None
        for name, item in self.results[list(self.results)[-1]].items():
            if "FWHM" in name:
                fwhm_dicts[name] = item
                if "interp_energy_in_kev" in item:
                    interp_energy = item["interp_energy_in_kev"]
                    for field in item:
                        if "_fwhm_in_kev" in field:
                            interp_fwhm_name = field.replace("_fwhm_in_kev", "")

        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, figsize=figsize
        )
        if len(np.where((~np.isnan(fwhms)) & (~np.isnan(dfwhms)))[0]) > 0:
            ax1.errorbar(fwhm_peaks, fwhms, yerr=dfwhms, marker="x", ls=" ", c="black")

            fwhm_slope_bins = np.arange(erange[0], erange[1], 10)

            if interp_energy is not None:
                qbb_line_vx = [interp_energy, interp_energy]
                qbb_line_hx = [erange[0], interp_energy]
                for name, fwhm_dict in fwhm_dicts.items():
                    qbb_line_vy = [np.inf, -np.inf]
                    low_lim = 0.9 * np.nanmin(
                        fwhm_dict["function"].func(
                            fwhm_slope_bins, *fwhm_dict["parameters"]
                        )
                    )
                    up_lim = fwhm_dict[f"{interp_fwhm_name}_fwhm_in_kev"]
                    if low_lim < qbb_line_vy[0]:
                        qbb_line_vy[0] = low_lim
                    if up_lim > qbb_line_vy[1]:
                        qbb_line_vy[1] = up_lim
                    ax1.plot(
                        qbb_line_hx,
                        [
                            fwhm_dict[f"{interp_fwhm_name}_fwhm_in_kev"],
                            fwhm_dict[f"{interp_fwhm_name}_fwhm_in_kev"],
                        ],
                        lw=1,
                        c="r",
                        ls="--",
                    )
                    ax1.plot(
                        fwhm_slope_bins,
                        fwhm_dict["function"].func(
                            fwhm_slope_bins, *fwhm_dict["parameters"]
                        ),
                        lw=1,
                        label=f'{name}, {interp_fwhm_name} fwhm: {fwhm_dict[f"{interp_fwhm_name}_fwhm_in_kev"]:1.2f} +- {fwhm_dict[f"{interp_fwhm_name}_fwhm_err_in_kev"]:1.2f} keV',
                    )
                    ax1.plot(qbb_line_vx, qbb_line_vy, lw=1, c="r", ls="--")

            ax1.set_xlim(erange)
            if np.isnan(low_lim):
                low_lim = 0
            ax1.set_ylim([low_lim, None])
            ax1.set_ylabel("FWHM energy resolution (keV)")
            for _, fwhm_dict in fwhm_dicts.items():
                ax2.plot(
                    fwhm_peaks,
                    (
                        fwhms
                        - fwhm_dict["function"].func(
                            fwhm_peaks, *fwhm_dict["parameters"]
                        )
                    )
                    / dfwhms,
                    lw=0,
                    marker="x",
                )
            ax2.plot(erange, [0, 0], color="black", lw=0.5)
            ax2.set_xlabel("Energy (keV)")
            ax2.set_ylabel("Normalised Residuals")
        plt.tight_layout()
        plt.close()
        return fig


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
    def bounds(ys):
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
        return [np.nanmin(ys), 2 * 10**-3, 10**-8]

    @staticmethod
    def bounds(ys):
        return [(0, np.nanmin(ys) ** 2), (10**-3, None), (0, None)]


def hpge_fit_energy_peak_tops(
    hist,
    bins,
    var,
    peak_locs,
    n_to_fit=7,
    cost_func="Least Squares",
    inflate_errors=False,
    gof_method="var",
    debug_mode=False,
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
        except BaseException as e:
            if e == KeyboardInterrupt:
                raise (e)
            elif debug_mode:
                raise (e)
            pars, cov = None, None

        pars_list.append(pars)
        cov_list.append(cov)
    return np.array(pars_list, dtype=object), np.array(cov_list, dtype=object)


def get_hpge_energy_peak_par_guess(
    energy, func, fit_range=None, bin_width=None, mode_guess=None
):
    """
    Get parameter guesses for func fit to peak in hist

    Parameters
    ----------
    energy : array
        An array of energy values in the range around the peak for guessing.
    func : function
        The function to be fit to the peak in the histogram.
    fit_range : tuple, optional
        A tuple specifying the range around the peak to perform the fit. If not provided, the entire range of energy values will be used.
    bin_width : float, optional
        The width of the bins in the histogram. Default is 1.
    mode_guess : float, optional
        A guess for the mode (mu) parameter of the function. If not provided, it will be estimated from the data.

    Returns
    -------
    ValueView
        A ValueView object from iminuit containing the parameter guesses for the function fit.

    Notes
    -----
    This function calculates parameter guesses for fitting a function to a peak in a histogram. It uses various methods to estimate the parameters based on the provided energy values and the selected function.

    If the function is 'gauss_on_step', the following parameters will be estimated:
    - n_sig: Number of signal events in the peak.
    - mu: Mean of the peak.
    - sigma: Standard deviation of the peak.
    - n_bkg: Number of background events.
    - hstep: Height of the step between the peak and the background.
    - x_lo: Lower bound of the fit range.
    - x_hi: Upper bound of the fit range.

    If the function is 'hpge_peak', the following parameters will be estimated:
    - n_sig: Number of signal events in the peak.
    - mu: Mean of the peak.
    - sigma: Standard deviation of the peak.
    - htail: Height of the tail component.
    - tau: Decay constant of the tail component.
    - n_bkg: Number of background events.
    - hstep: Height of the step between the peak and the background.
    - x_lo: Lower bound of the fit range.
    - x_hi: Upper bound of the fit range.

    If the provided function is not implemented, an error will be raised.

    Examples
    --------
    >>> energy = [1, 2, 3, 4, 5]
    >>> func = pgf.gauss_on_step
    >>> fit_range = (2, 4)
    >>> bin_width = 0.5
    >>> mode_guess = 3.5
    >>> get_hpge_energy_peak_par_guess(energy, func, fit_range, bin_width, mode_guess)
    {'n_sig': 3, 'mu': 3.5, 'sigma': 0.5, 'n_bkg': 2, 'hstep': 0.5, 'x_lo': 2, 'x_hi': 4}
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
        bin_width = 2 * (init_sigma) * len(energy) ** (-1 / 3)

    hist, bins, var = pgh.get_hist(energy, dx=bin_width, range=fit_range)

    if (
        func == pgf.gauss_on_step
        or func == pgf.hpge_peak
        or func == pgf.gauss_on_uniform
        or func == pgf.gauss_on_linear
    ):
        # get mu and height from a gauss fit, also sigma as fallback
        pars, cov = pgb.gauss_mode_width_max(
            hist, bins, var, mode_guess=mode_guess, n_bins=5
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
            if (
                sigma <= 0
                or abs(sigma / sigma_guess) > 5
                or sigma > (fit_range[1] - fit_range[0]) / 2
            ):
                raise ValueError
        except ValueError:
            try:
                sigma = pgh.get_fwfm(
                    0.6065,
                    hist,
                    bins,
                    var,
                    mx=height,
                    bl=bg - step / 2,
                    method="fit_slopes",
                )[0]
            except RuntimeError:
                sigma = -1
            if (
                sigma <= 0
                or sigma > (fit_range[1] - fit_range[0]) / 2
                or abs(sigma / sigma_guess) > 5
            ):
                if (
                    sigma_guess is not None
                    and sigma_guess > 0
                    and sigma_guess < (fit_range[1] - fit_range[0]) / 2
                ):
                    sigma = sigma_guess
                else:
                    (_, sigma, _) = pgh.get_gaussian_guess(hist, bins)
                    if (
                        sigma is not None
                        and sigma_guess > 0
                        and sigma_guess < (fit_range[1] - fit_range[0]) / 2
                    ):
                        pass
                    else:
                        log.info(
                            "get_hpge_energy_peak_par_guess: sigma estimation failed"
                        )
                        return {}
        # now compute amp and return
        n_sig = np.sum(
            hist[(bin_centres > mu - 3 * sigma) & (bin_centres < mu + 3 * sigma)]
        )
        n_bkg = np.sum(hist) - n_sig

        parguess = {
            "n_sig": n_sig,
            "mu": mu,
            "sigma": sigma,
            "n_bkg": n_bkg,
            "x_lo": bins[0],
            "x_hi": bins[-1],
        }

        if func == pgf.gauss_on_linear:
            # bg1 = np.mean(hist[-10:])
            # bg2 = np.mean(hist[:10])
            # m = (bg1 - bg2) / (bins[-5] - bins[5])
            # b = bg1 - m * bins[-5]
            parguess["m"] = 0
            parguess["b"] = 1

        elif func == pgf.gauss_on_step or func == pgf.hpge_peak:
            hstep = step / (bg + np.mean(hist[:10]))
            parguess["hstep"] = hstep

            if func == pgf.hpge_peak:
                sigma = sigma * 0.8  # roughly remove some amount due to tail
                # for now hard-coded
                htail = 1.0 / 5
                tau = sigma / 2
                parguess["sigma"] = sigma
                parguess["htail"] = htail
                parguess["tau"] = tau

        for name, guess in parguess.items():
            if np.isnan(guess):
                parguess[name] = 0

    else:
        log.error(f"get_hpge_energy_peak_par_guess not implemented for {func.__name__}")
        return return_nans(func)

    return convert_to_minuit(parguess, func).values


def get_hpge_energy_fixed(func):
    """
    Get the fixed indexes for fitting and mask for parameters based on the given function.

    Parameters
    ----------
    func : function
        The function for which the fixed indexes and mask are to be determined.

    Returns
    -------
    fixed : list
        A sequence list of fixed indexes for fitting.
    mask : ndarray
        A boolean mask indicating which parameters are fixed (False) and which are not fixed (True).
    """

    if (
        func == pgf.gauss_on_step
        or func == pgf.hpge_peak
        or func == pgf.gauss_on_uniform
        or func == pgf.gauss_on_linear
    ):
        # pars are: n_sig, mu, sigma, n_bkg, hstep, components
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
            "sigma": (0, (parguess["x_hi"] - parguess["x_lo"]) / 2),
            "n_bkg": (0, None),
            "hstep": (-1, 1),
            "x_lo": (None, None),
            "x_hi": (None, None),
        }

    elif func == pgf.hpge_peak:
        return {
            "n_sig": (0, None),
            "mu": (parguess["x_lo"], parguess["x_hi"]),
            "sigma": (0, (parguess["x_hi"] - parguess["x_lo"]) / 2),
            "htail": (0, 0.5),
            "tau": (0.1 * parguess["sigma"], 5 * parguess["sigma"]),
            "n_bkg": (0, None),
            "hstep": (-1, 1),
            "x_lo": (None, None),
            "x_hi": (None, None),
        }

    elif func == pgf.gauss_on_uniform:
        return {
            "n_sig": (0, None),
            "mu": (parguess["x_lo"], parguess["x_hi"]),
            "sigma": (0, (parguess["x_hi"] - parguess["x_lo"]) / 2),
            "n_bkg": (0, None),
            "x_lo": (None, None),
            "x_hi": (None, None),
        }
    elif func == pgf.gauss_on_linear:
        return {
            "n_sig": (0, None),
            "mu": (parguess["x_lo"], parguess["x_hi"]),
            "sigma": (0, (parguess["x_hi"] - parguess["x_lo"]) / 2),
            "n_bkg": (0, None),
            "m": (None, None),
            "b": (None, None),
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

    def _value(self, *pars):
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


def sum_bins(hist, bins, var, threshold=5):
    removed_ids = []
    for idx in range(len(hist)):
        counter = 0
        if idx not in removed_ids and idx < len(hist) - 1:
            while hist[idx] < threshold and idx < len(hist) - 1:
                hist[idx] += hist[idx + 1]
                var[idx] += var[idx + 1]
                hist = np.delete(hist, idx + 1)
                bins = np.delete(bins, idx + 1)
                var = np.delete(var, idx + 1)
                counter += 1
                removed_ids.append([idx + 1 + counter])
        elif idx == len(hist) - 2:
            hist[idx - 1] += hist[idx]
            var[idx - 1] += var[idx]
            hist = np.delete(hist, idx)
            bins = np.delete(bins, idx)
            var = np.delete(var, idx)

    return hist, bins, var


def average_counts_check(hist, bins, var, threshold=1):
    for i, _bin_i in enumerate(hist):
        if hist[i] <= 1 and np.nanmean(hist[i:]) < threshold:
            return hist[:i], bins[: i + 1], var[:i]
    return hist, bins, var


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
    bin_width=None,
    lock_guess=False,
    p_val_threshold=10e-20,
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
        bin_width = 2 * (init_sigma) * len(energy) ** (-1 / 3)

    gof_hist, gof_bins, gof_var = pgh.get_hist(energy, range=gof_range, dx=bin_width)
    # remove remaining when average counts < 1
    gof_hist, gof_bins, gof_var = average_counts_check(gof_hist, gof_bins, gof_var)
    # sum bins with counts < 5
    gof_hist, gof_bins, gof_var = sum_bins(gof_hist, gof_bins, gof_var)

    if guess is not None:
        if not isinstance(guess, ValueView):
            x0 = convert_to_minuit(guess, func)
        if lock_guess is True:
            x0 = guess
            x0["x_lo"] = fit_range[0]
            x0["x_hi"] = fit_range[1]
        else:
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
            if len(x0) == len(x1):
                cs, _ = pgb.goodness_of_fit(
                    gof_hist, gof_bins, None, func.pdf_norm, x0, method="Pearson"
                )
                cs2, _ = pgb.goodness_of_fit(
                    gof_hist, gof_bins, None, func.pdf_norm, x1, method="Pearson"
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
                **guess_kwargs if guess_kwargs is not None else {},
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
            m.fixed[fixed] = True
            m.simplex().migrad()
            m.hesse()
            x0 = guess_func(
                energy,
                func,
                fit_range,
                bin_width=bin_width,
                **guess_kwargs if guess_kwargs is not None else {},
            )
            cs = pgb.goodness_of_fit(
                gof_hist,
                gof_bins,
                gof_var,
                pgf.gauss_on_step.get_pdf,
                m.values,
                method="Pearson",
                scale_bins=True,
            )
            cs = (cs[0], cs[1] + len(np.where(mask)[0]))
            p_val = chi2.sf(cs[0], cs[1])
            if m.valid and (p_val > 0):
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
    m.fixed[fixed] = True
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
    m2.fixed[fixed] = True
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
        hist, bins, _ = pgh.get_hist(energy, range=fit_range, dx=bin_width)
        bin_cs = (bins[:-1] + bins[1:]) / 2

        m_fit = func.get_pdf(bin_cs, *m.values) * np.diff(bin_cs)[0]
        m2_fit = func.get_pdf(bin_cs, *m2.values) * np.diff(bin_cs)[0]
        guess_fit = func.get_pdf(bin_cs, *x0) * np.diff(bin_cs)[0]
        plt.figure()
        plt.step(bin_cs, hist, label="hist")
        plt.plot(bin_cs, guess_fit, label="Guess")
        plt.plot(bin_cs, m_fit, label=f"Fit 1: {cs}")
        plt.plot(bin_cs, m2_fit, label=f"Fit 2: {cs2}")
        plt.legend()
        plt.show()

    if valid1 is False and valid2 is False:
        log.debug("Extra simplex needed")
        m = Minuit(c, *x0)
        if tol is not None:
            m.tol = tol
        m.fixed[fixed] = True
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
        if (
            (p_val_no_tail > p_val)
            or ((fit[0]["htail"] < fit[1]["htail"]) & (p_val_no_tail > p_val_threshold))
            or (
                (fit[0]["htail"] < fit[1]["htail"])
                & (p_val_no_tail < p_val_threshold)
                & (p_val < p_val_threshold)
            )
        ):
            debug_string = f'dropping tail tail val : {fit[0]["htail"]} tail err : {fit[1]["htail"]} '
            debug_string += f"p_val no tail: : {p_val_no_tail} p_val with tail: {p_val}"
            log.debug(debug_string)

            if display > 0:
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
        pars = np.array([0, scale])
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
        pars = np.array([0, scale])
        cov = np.array([[0, 0], [0, scale_cov]])
        errs = np.diag(np.sqrt(cov))
    else:
        d_mu_d_es = np.zeros(len(mus))
        for n in range(len(energy_scale_pars) - 1):
            d_mu_d_es += energy_scale_pars[n + 1] * mus ** (n + 1)
        e_weights = np.sqrt(d_mu_d_es * mu_vars)
        mask = np.isfinite(e_weights)
        poly_pars = (
            Polynomial.fit(
                mus[mask], energies_kev[mask], deg=deg, w=1 / e_weights[mask]
            )
            .convert()
            .coef
        )
        if fixed is not None:
            for idx, val in fixed.items():
                if val is True or val is None:
                    pass
                else:
                    poly_pars[idx] = val
        c = cost.LeastSquares(
            mus[mask], energies_kev[mask], e_weights[mask], poly_wrapper
        )
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
