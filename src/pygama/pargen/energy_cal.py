"""routines for automatic calibration.

- hpge_find_E_peaks (Find uncalibrated E peaks whose E spacing matches the pattern in peaks_keV)
- hpge_get_E_peaks (Get uncalibrated E peaks at the energies of peaks_keV)
- hpge_fit_E_peaks (fits the energy peals)
- hpge_E_calibration (main routine -- finds and fits peaks specified)
"""
import logging
import sys

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.signal import find_peaks_cwt, medfilt

import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.math.utils as pgu

log = logging.getLogger(__name__)


def hpge_find_E_peaks(
    hist,
    bins,
    var,
    peaks_keV,
    n_sigma=5,
    deg=0,
    Etol_keV=None,
    var_zero=1,
    verbose=False,
):
    """Find uncalibrated E peaks whose E spacing matches the pattern in peaks_keV
    Note: the specialization here to units "keV" in peaks and Etol is
    unnecessary. However it is kept so that the default value for Etol_keV has
    an unambiguous interpretation.

    Parameters
    ----------
    hist, bins, var : array, array, array
        Histogram of uncalibrated energies, see pgh.get_hist()
        var cannot contain any zero entries.
    peaks_keV : array
        Energies of peaks to search for (in keV)
    n_sigma : float
        Threshold for detecting a peak in sigma (i.e. sqrt(var))
    deg : int
        deg arg to pass to poly_match
    Etol_keV : float
        absolute tolerance in energy for matching peaks
    var_zero : float
        number used to replace zeros of var to avoid divide-by-zero in
        hist/sqrt(var). Default value is 1. Usually when var = 0 its because
        hist = 0, and any value here is fine.
    verbose : bool
        print debug messages

    Returns
    -------
    detected_peak_locations : list
        list of uncalibrated energies of detected peaks
    detected_peak_energies : list
        list of calibrated energies of detected peaks
    pars : list of floats
        the parameters for poly(peaks_uncal) = peaks_keV (polyfit convention)
    """
    # clean up var if necessary
    if np.any(var == 0):
        log.debug(f"hpge_find_E_peaks: replacing var zeros with {var_zero}")
        var[np.where(var == 0)] = var_zero
    peaks_keV = np.asarray(peaks_keV)

    # Find all maxes with > n_sigma significance
    imaxes = get_i_local_maxima(hist / np.sqrt(var), n_sigma)

    # Now pattern match to peaks_keV within Etol_keV using poly_match
    detected_max_locs = pgh.get_bin_centers(bins)[imaxes]

    if Etol_keV is None:
        # estimate Etol_keV
        pt_pars, pt_covs = hpge_fit_E_peak_tops(
            hist, bins, var, detected_max_locs, n_to_fit=15
        )
        if (
            sum(sum(sum(c) if (c != None).any() else 0 for c in pt_covs)) == np.inf
            or sum(sum(sum(c) if (c != None).any() else 0 for c in pt_covs)) == 0
        ):
            print(
                "hpge_find_E_peaks: can safely ignore previous covariance warning, not used"
            )
        pt_pars = pt_pars[np.array([x is not None for x in pt_pars])]
        med_sigma_ratio = np.median(np.stack(pt_pars)[:, 1] / np.stack(pt_pars)[:, 0])

        Etol_keV = 5.0 * (med_sigma_ratio / 0.003)
    pars, ixtup, iytup = poly_match(
        detected_max_locs, peaks_keV, deg=deg, atol=Etol_keV
    )

    if len(ixtup) != len(peaks_keV):
        log.warning(
            f"hpge_find_E_peaks: only found {len(ixtup)} of {len(peaks_keV)} expected peaks"
        )
    return detected_max_locs[ixtup], peaks_keV[iytup], pars


def hpge_get_E_peaks(
    hist,
    bins,
    var,
    cal_pars,
    peaks_keV,
    n_sigma=3,
    Etol_keV=5,
    var_zero=1,
    verbose=False,
):
    """Get uncalibrated E peaks at the energies of peaks_keV

    Parameters
    ----------
    hist, bins, var : array, array, array
        Histogram of uncalibrated energies, see pgh.get_hist()
        var cannot contain any zero entries.
    cal_pars : array
        Estimated energy calibration parameters used to search for peaks
    peaks_keV : array
        Energies of peaks to search for (in keV)
    n_sigma : float
        Threshold for detecting a peak in sigma (i.e. sqrt(var))
    Etol_keV : float
        absolute tolerance in energy for matching peaks
    var_zero : float
        number used to replace zeros of var to avoid divide-by-zero in
        hist/sqrt(var). Default value is 1. Usually when var = 0 its because
        hist = 0, and any value here is fine.
    verbose : bool
        print debug messages

    Returns
    -------
    got_peak_locations : list
        list of uncalibrated energies of found peaks
    got_peak_energies : list
        list of calibrated energies of found peaks
    pars : list of floats
        the parameters for poly(peaks_uncal) = peaks_keV (polyfit convention)
    """
    # clean up var if necessary
    if np.any(var == 0):
        log.debug(f"hpge_find_E_peaks: replacing var zeros with {var_zero}")
        var[np.where(var == 0)] = var_zero
    peaks_keV = np.asarray(peaks_keV)

    # Find all maxes with > n_sigma significance
    imaxes = get_i_local_maxima(hist / np.sqrt(var), n_sigma)

    # Keep maxes if they coincide with expected peaks
    test_peaks_keV = np.asarray([pgf.poly(i, cal_pars) for i in bins[imaxes]])
    imatch = [abs(peaks_keV - i).min() < Etol_keV for i in test_peaks_keV]

    got_peak_locations = bins[imaxes[imatch]]
    got_peak_energies = test_peaks_keV[imatch]

    # Match calculated and true peak energies
    matched_energies = peaks_keV[
        [np.argmin(abs(peaks_keV - i)) for i in got_peak_energies]
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

    # Calculate updated calibration curve
    pars = np.polyfit(got_peak_locations, matched_energies, len(cal_pars))

    return got_peak_locations, matched_energies, pars


def hpge_fit_E_peak_tops(
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
    for E_peak in peak_locs:
        try:
            pars, cov = pgf.gauss_mode_width_max(
                hist,
                bins,
                var,
                mode_guess=E_peak,
                n_bins=n_to_fit,
                cost_func=cost_func,
                inflate_errors=inflate_errors,
                gof_method=gof_method,
            )
        except:
            pars, cov = None, None

        pars_list.append(pars)
        cov_list.append(cov)
    return np.array(pars_list, dtype=object), np.array(cov_list, dtype=object)


def get_hpge_E_peak_par_guess(hist, bins, var, func):
    """Get parameter guesses for func fit to peak in hist

    Parameters
    ----------
    hist, bins, var : array, array, array
        Histogram of uncalibrated energies, see pgh.get_hist(). Should be
        windowed around the peak.
    func : function
        The function to be fit to the peak in the (windowed) hist
    """
    if (
        func == pgf.gauss_step_cdf
        or func == pgf.gauss_step_pdf
        or func == pgf.extended_gauss_step_pdf
    ):
        # get mu and height from a gauss fit, also sigma as fallback
        pars, cov = pgf.gauss_mode_width_max(hist, bins, var)
        bin_centres = pgh.get_bin_centers(bins)
        if pars is None:
            log.warning("get_hpge_E_peak_par_guess: gauss_mode_width_max failed")
            i_0 = np.argmax(hist)
            mu = bin_centres[i_0]
            height = hist[i_0]
            sigma_guess = None
        else:
            mu = pars[0]
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
            if sigma == 0:
                raise ValueError
        except:
            sigma = pgh.get_fwfm(
                0.6065,
                hist,
                bins,
                var,
                mx=height,
                bl=bg - step / 2,
                method="fit_slopes",
            )[0]
            if sigma == 0:
                log.warning("get_hpge_E_peak_par_guess: sigma estimation failed")
                if sigma_guess is not None:
                    sigma = sigma_guess
                else:
                    return []

        # now compute amp and return
        n_sig = np.sum(
            hist[(bin_centres > mu - 3 * sigma) & (bin_centres < mu + 3 * sigma)]
        )
        n_bkg = np.sum(hist) - n_sig

        hstep = step / (bg + np.mean(hist[:10]))
        return [n_sig, mu, sigma / 2, n_bkg, hstep, bins[0], bins[-1], 0]

    if (
        func == pgf.radford_cdf
        or func == pgf.radford_pdf
        or func == pgf.extended_radford_pdf
    ):

        # guess mu, height
        pars, cov = pgf.gauss_mode_width_max(hist, bins, var)
        bin_centres = pgh.get_bin_centers(bins)
        if pars is None:
            log.warning("get_hpge_E_peak_par_guess: gauss_mode_width_max failed")
            sigma_guess = None

        else:
            sigma_guess = pars[1]
            # mu=pars[0]
            # height=pars[2]
        i_0 = np.argmax(hist)
        mu = bin_centres[i_0]
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
            if sigma == 0:
                raise ValueError
        except:
            sigma = pgh.get_fwfm(
                0.6065,
                hist,
                bins,
                var,
                mx=height,
                bl=bg0 + step / 2,
                method="fit_slopes",
            )[0]
            if sigma == 0:
                log.warning("get_hpge_E_peak_par_guess: sigma estimation failed")
                if sigma_guess is not None:
                    sigma = sigma_guess
                else:
                    return []
        sigma = sigma * 0.6  # roughly remove some amount due to tail

        # for now hard-coded
        htail = 1.0 / 5
        tau = 0.5 * sigma

        hstep = step / (bg0 + np.mean(hist[:10]))

        n_sig = np.sum(
            hist[(bin_centres > mu - 3 * sigma) & (bin_centres < mu + 3 * sigma)]
        )
        n_bkg = np.sum(hist) - n_sig

        parguess = [n_sig, mu, sigma, htail, tau, n_bkg, hstep, bins[0], bins[-1], 0]

        return parguess

    else:
        log.error(f"get_hpge_E_peak_par_guess not implemented for {func.__name__}")
        return []


def get_hpge_E_fixed(func):
    """
    Returns: Sequence list of fixed indexes for fitting and mask for parameters
    """

    if (
        func == pgf.gauss_step_cdf
        or func == pgf.gauss_step_pdf
        or func == pgf.extended_gauss_step_pdf
    ):
        # pars are: n_sig, mu, sigma, n_bkg, hstep, components
        return [5, 6, 7], np.array([True, True, True, True, True, False, False, False])

    if (
        func == pgf.radford_cdf
        or func == pgf.radford_pdf
        or func == pgf.extended_radford_pdf
    ):
        # pars are: n_sig, mu, sigma, htail,tau, n_bkg, hstep, components
        return [7, 8, 9], np.array(
            [True, True, True, True, True, True, True, False, False, False]
        )

    else:
        log.error(f"get_hpge_E_fixed not implemented for {func.__name__}")
        return None
    return None


def get_hpge_E_bounds(func):
    if (
        func == pgf.radford_cdf
        or func == pgf.radford_pdf
        or func == pgf.extended_radford_pdf
    ):
        return [
            (0, None),
            (None, None),
            (None, None),
            (0, 1),
            (0, None),
            (0, None),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
        ]

    elif (
        func == pgf.gauss_step_cdf
        or func == pgf.gauss_step_pdf
        or func == pgf.extended_gauss_step_pdf
    ):
        return [
            (0, None),
            (None, None),
            (None, None),
            (0, None),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
        ]

    else:
        log.error(f"get_hpge_E_bounds not implemented for {func.__name__}")
        return []


def hpge_fit_E_peaks(
    E_uncal,
    mode_guesses,
    wwidths,
    n_bins=50,
    funcs=pgf.gauss_step_cdf,
    method="unbinned",
    gof_funcs=None,
    n_events=15000,
    allowed_p_val=0.05,
    uncal_is_int=False,
    simplex=False,
):
    """Fit the Energy peaks specified using the function given

    Parameters
    ----------
    E_uncal : array
        unbinned energy data to be fit
    mode_guesses : array
        array of guesses for modes of each peak
    wwidths : float or array of float
        array of widths to use for the fit windows (in units of E_uncal),
        typically on the order of 10 sigma where sigma is the peak width
    n_bins : int or array of ints
        array of number of bins to use for the fit window histogramming
    funcs : function or array of functions
        funcs to be used to fit each region
    method : str
        default is unbinned fit can specify to use binned fit method instead
    gof_funcs : function or array of functions
        functions to use for calculation goodness of fit if unspecified will use same func as fit
    uncal_is_int : bool
        if True, attempts will be made to avoid picket-fencing when binning
        E_uncal
    simplex : bool determining whether to do a round of simpson minimisation before gradient minimisation
    n_events : int number of events to use for unbinned fit
    allowed_p_val : lower limit on p_val of fit

    Returns
    -------
    pars : list of array
        a list of best-fit parameters for each peak fit
    covs : list of 2D arrays
        a list of covariance matrices for each pars
    binwidths : list
        a list of bin widths used for each peak fit
    ranges: list of array
        a list of [Euc_min, Euc_max] used for each peak fit
    """
    pars = []
    covs = []
    binws = []
    ranges = []
    errors = []
    p_vals = []

    for i_peak in range(len(mode_guesses)):
        # get args for this peak
        wwidth_i = wwidths if not isinstance(wwidths, list) else wwidths[i_peak]
        n_bins_i = n_bins if np.isscalar(n_bins) else n_bins[i_peak]
        func_i = funcs[i_peak] if hasattr(funcs, "__len__") else funcs
        wleft_i = wwidth_i / 2 if np.isscalar(wwidth_i) else wwidth_i[0]
        wright_i = wwidth_i / 2 if np.isscalar(wwidth_i) else wwidth_i[1]
        if gof_funcs is not None:
            gof_func_i = (
                gof_funcs[i_peak] if hasattr(gof_funcs, "__len__") else gof_funcs
            )
        else:
            gof_func_i = func_i

        try:
            # bin a histogram
            Euc_min = mode_guesses[i_peak] - wleft_i
            Euc_max = mode_guesses[i_peak] + wright_i
            Euc_min, Euc_max, n_bins_i = pgh.better_int_binning(
                x_lo=Euc_min, x_hi=Euc_max, n_bins=n_bins_i
            )

            if method == "unbinned":
                energies = E_uncal[(E_uncal > Euc_min) & (E_uncal < Euc_max)][:n_events]
                hist, bins, var = pgh.get_hist(
                    energies, bins=n_bins_i, range=(Euc_min, Euc_max)
                )
                par_guesses = get_hpge_E_peak_par_guess(hist, bins, var, func_i)
                bounds = get_hpge_E_bounds(func_i)
                fixed, mask = get_hpge_E_fixed(func_i)

                pars_i, errs_i, cov_i = pgf.fit_unbinned(
                    func_i,
                    energies,
                    guess=par_guesses,
                    Extended=True,
                    fixed=fixed,
                    simplex=simplex,
                    bounds=bounds,
                )
            else:
                hist, bins, var = pgh.get_hist(
                    E_uncal, bins=n_bins_i, range=(Euc_min, Euc_max)
                )
                par_guesses = get_hpge_E_peak_par_guess(hist, bins, var, func_i)
                bounds = get_hpge_E_bounds(func_i)
                fixed, mask = get_hpge_E_fixed(func_i)
                pars_i, errs_i, cov_i = pgf.fit_binned(
                    func_i,
                    hist,
                    bins,
                    var=var,
                    guess=par_guesses,
                    cost_func=method,
                    Extended=True,
                    fixed=fixed,
                    simplex=simplex,
                    bounds=bounds,
                )

            csqr = pgf.goodness_of_fit(
                hist, bins, None, gof_func_i, pars_i, method="LR"
            )
            p_val = scipy.stats.chi2.sf(csqr[0], csqr[1])

            pars_i = np.array(pars_i)[mask]
            errs_i = np.array(errs_i)[mask]
            cov_i = np.array(cov_i)[mask, :][:, mask]

            total_events = pgf.get_total_events_func(func_i, pars_i, errors=errs_i)
            if (
                sum(sum(c) if c is not None else 0 for c in cov_i) == np.inf
                or sum(sum(c) if c is not None else 0 for c in cov_i) == 0
                or np.isnan(sum(sum(c) if c is not None else 0 for c in cov_i))
            ):
                log.debug(
                    f"hpge_fit_E_peaks: cov estimation failed for i_peak={i_peak} at loc {mode_guesses[i_peak]:g}"
                )
                pars_i, errs_i, cov_i, p_val = None, None, None, None

            elif (np.abs(errs_i / pars_i) < 1e-7).any():
                log.debug(
                    f"hpge_fit_E_peaks: cov estimation failed for i_peak={i_peak} at loc {mode_guesses[i_peak]:g}, parameter error too low"
                )
                pars_i, errs_i, cov_i, p_val = None, None, None, None

            elif np.abs(total_events[0] - np.sum(hist)) / np.sum(hist) > 0.1:
                log.debug(
                    f"hpge_fit_E_peaks: fit failed for i_peak={i_peak} at loc {mode_guesses[i_peak]:g}, total_events is outside limit"
                )
                pars_i, errs_i, cov_i, p_val = None, None, None, None

            elif p_val < allowed_p_val:
                log.debug(
                    f"hpge_fit_E_peaks: fit failed for i_peak={i_peak}, p-value too low: {p_val}"
                )
                pars_i, errs_i, cov_i, p_val = None, None, None, None

        except:
            pars_i, errs_i, cov_i, p_val = None, None, None, None

        # get binning
        binw_1 = (bins[-1] - bins[0]) / (len(bins) - 1)

        pars.append(pars_i)
        errors.append(errs_i)
        covs.append(cov_i)
        binws.append(binw_1)
        ranges.append([Euc_min, Euc_max])
        p_vals.append(p_val)

    return (
        np.array(pars, dtype=object),
        np.array(errors, dtype=object),
        np.array(covs, dtype=object),
        np.array(binws),
        np.array(ranges),
        np.array(p_vals),
    )


def hpge_fit_E_scale(mus, mu_vars, Es_keV, deg=0):
    """Find best fit of poly(E) = mus +/- sqrt(mu_vars)
    Compare to hpge_fit_E_cal_func which fits for E = poly(mu)

    Parameters
    ----------
    mus : array
        uncalibrated energies
    mu_vars : array
        variances in the mus
    Es_keV : array
        energies to fit to, in keV
    deg : int
        degree for energy scale fit. deg=0 corresponds to a simple scaling
        mu = scale * E. Otherwise deg follows the definition in np.polyfit

    Returns
    -------
    pars : array
        parameters of the best fit. Follows the convention in np.polyfit
    cov : 2D array
        covariance matrix for the best fit parameters.
    """
    if deg == 0:
        scale, scale_cov = pgu.fit_simple_scaling(Es_keV, mus, var=mu_vars)
        pars = np.array([scale, 0])
        cov = np.array([[scale_cov, 0], [0, 0]])
    else:
        pars, cov = np.polyfit(Es_keV, mus, deg=deg, w=1 / np.sqrt(mu_vars), cov=True)
    return pars, cov


def hpge_fit_E_cal_func(mus, mu_vars, Es_keV, E_scale_pars, deg=0):
    """Find best fit of E = poly(mus +/- sqrt(mu_vars))
    This is an inversion of hpge_fit_E_scale.
    E uncertainties are computed from mu_vars / dmu/dE where mu = poly(E) is the
    E_scale function

    Parameters
    ----------
    mus : array
        uncalibrated energies
    mu_vars : array
        variances in the mus
    Es_keV : array
        energies to fit to, in keV
    E_scale_pars : array
        ???
    deg : int
        degree for energy scale fit. deg=0 corresponds to a simple scaling
        mu = scale * E. Otherwise deg follows the definition in np.polyfit

    Returns
    -------
    pars : array
        parameters of the best fit. Follows the convention in np.polyfit
    cov : 2D array
        covariance matrix for the best fit parameters.
    """
    if deg == 0:
        E_vars = mu_vars / E_scale_pars[0] ** 2
        scale, scale_cov = pgu.fit_simple_scaling(mus, Es_keV, var=E_vars)
        pars = np.array([scale, 0])
        cov = np.array([[scale_cov, 0], [0, 0]])
    else:
        mu_ns = np.ones(len(mus))
        dmudEs = np.zeros(len(mus))
        for n in range(len(E_scale_pars) - 1):
            dmudEs += mu_ns * E_scale_pars[-n - 1]
            mu_ns *= mus
        E_weights = dmudEs / np.sqrt(mu_vars)
        pars, cov = np.polyfit(mus, Es_keV, deg=deg, w=E_weights, cov=True)
    return pars, cov


def hpge_E_calibration(
    E_uncal,
    peaks_keV,
    guess_keV,
    deg=0,
    uncal_is_int=False,
    range_keV=None,
    funcs=pgf.gauss_step_cdf,
    gof_funcs=None,
    method="unbinned",
    gof_func=None,
    n_events=15000,
    simplex=False,
    allowed_p_val=0.05,
    verbose=True,
):
    """Calibrate HPGe data to a set of known peaks

    Parameters
    ----------
    E_uncal : array
        unbinned energy data to be calibrated
    peaks_keV : array
        list of peak energies to be fit to. Each must be in the data
    guess_keV : float
        a rough initial guess at the conversion factor from E_uncal to keV. Must
        be positive
    deg : non-negative int
        degree of the polynomial for the E_cal function E_keV = poly(E_uncal).
        deg = 0 corresponds to a simple scaling E_keV = scale * E_uncal.
        Otherwise follows the convention in np.polyfit
    uncal_is_int : bool
        if True, attempts will be made to avoid picket-fencing when binning
        E_uncal
    range_keV : float, tuple, array of floats, or array of tuples of floats
        ranges around which the peak fitting is performed
        if tuple(s) are supplied, they provide the left and right ranges
    funcs
        DOCME
    gof_funcs : function or array of functions
        functions to use for calculation goodness of fit if unspecified will use same func as fit
    method : str
        default is unbinned fit can specify to use binned fit method instead
    gof_func
        DOCME
    n_events : int
        number of events to use for unbinned fit
    simplex : bool
        DOCME
    allowed_p_val
        lower limit on p_val of fit
    verbose : bool
        print debug statements

    Returns
    -------
    pars, cov : array, 2D array
        array of calibration function parameters and their covariances. The form
        of the function is E_keV = poly(E_uncal). Assumes poly() is
        overwhelmingly dominated by the linear term. pars follows convention in
        np.polyfit unless deg=0, in which case it is the (lone) scale factor
    results : dict with the following elements
        'detected_peaks_locs', 'detected_peaks_keV' : array, array
            array of rough uncalibrated/calibrated energies at which the fit peaks were
            found in the initial peak search
        'pt_pars', 'pt_cov' : list of (array), list of (2D array)
            arrays of gaussian parameters / covariances fit to the peak tops in
            the first refinement
        'pt_cal_pars', 'pt_cal_cov' : array, 2D array
            array of calibration parameters E_uncal = poly(E_keV) for fit to
            means of gausses fit to tops of each peak
        'pk_pars', 'pk_cov', 'pk_binws', 'pk_ranges' : list of (array), list of (2D array), list, list of (array)
            the best fit parameters, covariances, bin width and energy range for the local fit to each peak
        'pk_cal_pars', 'pk_cal_cov' : array, 2D array
            array of calibration parameters E_uncal = poly(E_keV) for fit to
            means from full peak fits
        'fwhms', 'dfwhms' : array, array
            the numeric fwhms and their uncertainties for each peak.
    """
    results = {}

    if not isinstance(range_keV, list):
        range_keV = [range_keV for peak in peaks_keV]

    if not hasattr(funcs, "__len__"):
        funcs = [funcs for peak in peaks_keV]

    # sanity checks
    E_uncal = np.asarray(E_uncal)
    peaks_keV = np.asarray(peaks_keV)  # peaks_keV = np.sort(peaks_keV)
    deg = int(deg)
    if guess_keV <= 0:
        log.error(f"hpge_E_cal warning: invalid guess_keV = {guess_keV}")
        return None, None, results
    if deg < 0:
        log.error(f"hpge_E_cal warning: invalid deg = {deg}")
        return None, None, results

    # bin the histogram in ~1 keV bins for the initial rough peak search
    Euc_min = peaks_keV[0] / guess_keV * 0.6
    Euc_max = peaks_keV[-1] / guess_keV * 1.1
    dEuc = 1 / guess_keV
    if uncal_is_int:
        Euc_min, Euc_max, dEuc = pgh.better_int_binning(
            x_lo=Euc_min, x_hi=Euc_max, dx=dEuc
        )
    hist, bins, var = pgh.get_hist(E_uncal, range=(Euc_min, Euc_max), dx=dEuc)

    # Run the initial rough peak search
    detected_peaks_locs, detected_peaks_keV, roughpars = hpge_find_E_peaks(
        hist, bins, var, peaks_keV, n_sigma=5, deg=deg
    )
    log.info(f"{len(detected_peaks_locs)} peaks found:")
    log.info(f"\t   Energy   | Position  ")
    for i, (Li, Ei) in enumerate(zip(detected_peaks_locs, detected_peaks_keV)):
        log.info(f"\t{i}".ljust(4) + str(Ei).ljust(9) + f"| {Li:g}".ljust(5))

    # re-bin the histogram in ~0.2 keV bins with updated E scale par for peak-top fits
    Euc_min, Euc_max = (
        (np.poly1d(roughpars) - i).roots
        for i in (peaks_keV[0] * 0.9, peaks_keV[-1] * 1.1)
    )
    Euc_min = Euc_min[np.logical_and(Euc_min >= 0, Euc_min <= max(Euc_max))][0]
    Euc_max = Euc_max[
        np.logical_and(Euc_max >= Euc_min, Euc_max <= np.nanmax(E_uncal))
    ][0]
    dEuc = 0.2 / roughpars[-2]

    if uncal_is_int:
        Euc_min, Euc_max, dEuc = pgh.better_int_binning(
            x_lo=Euc_min, x_hi=Euc_max, dx=dEuc
        )
    hist, bins, var = pgh.get_hist(E_uncal, range=(Euc_min, Euc_max), dx=dEuc)

    # run peak getter after rebinning
    got_peaks_locs, got_peaks_keV, roughpars = hpge_get_E_peaks(
        hist, bins, var, roughpars, peaks_keV, n_sigma=3
    )
    results["got_peaks_locs"] = got_peaks_locs
    results["got_peaks_keV"] = got_peaks_keV

    log.info(f"{len(got_peaks_locs)} peaks obtained:")
    log.info(f"\t   Energy   | Position  ")
    for i, (Li, Ei) in enumerate(zip(got_peaks_locs, got_peaks_keV)):
        log.info(f"\t{i}".ljust(4) + str(Ei).ljust(9) + f"| {Li:g}".ljust(5))

    # Drop non-gotten peaks
    idx = [i for i, E in enumerate(peaks_keV) if E in got_peaks_keV]
    range_keV = [range_keV[i] for i in idx]
    funcs = [funcs[i] for i in idx]

    # Drop peaks to not be fitted
    tmp = zip(
        *[
            a
            for a in zip(got_peaks_locs, got_peaks_keV, range_keV, funcs)
            if a[2] and a[3]
        ]
    )
    got_peaks_locs, got_peaks_keV, range_keV, funcs = list(map(list, tmp))
    got_peaks_locs = np.asarray(got_peaks_locs)
    got_peaks_keV = np.asarray(got_peaks_keV)

    # Now do a series of full fits to the peak shapes

    # First calculate range around peaks to fit
    if range_keV is None:
        # Need to do initial fit
        pt_pars, pt_covs = hpge_fit_E_peak_tops(
            hist, bins, var, got_peaks_locs, n_to_fit=7
        )
        # Drop failed fits
        fitidx = [i is not None for i in pt_pars]
        results["got_peaks_locs"] = got_peaks_locs = got_peaks_locs[fitidx]
        results["got_peaks_keV"] = got_peaks_keV = got_peaks_keV[fitidx]
        pt_pars = np.asarray(pt_pars)[fitidx]
        pt_covs = np.asarray(pt_covs)[fitidx]
        range_uncal = np.stack(pt_pars)[:, 1].astype(float) * 20
        n_bins = 50
    elif np.isscalar(range_keV):
        derco = np.polyder(np.poly1d(roughpars)).coefficients
        der = [pgf.poly(Ei, derco) for Ei in got_peaks_keV]
        range_uncal = [float(range_keV) / d for d in der]
        n_bins = [range_keV / 0.5 / d for d in der]
    elif isinstance(range_keV, tuple):
        rangeleft_keV, rangeright_keV = range_keV
        derco = np.polyder(np.poly1d(roughpars)).coefficients
        der = [pgf.poly(Ei, derco) for Ei in got_peaks_keV]
        range_uncal = [(rangeleft_keV / d, rangeright_keV / d) for d in der]
        n_bins = [sum(range_keV) / 0.5 / d for d in der]
    elif isinstance(range_keV, list):
        derco = np.polyder(np.poly1d(roughpars)).coefficients
        der = [pgf.poly(Ei, derco) for Ei in got_peaks_keV]
        range_uncal = [
            (r[0] / d, r[1] / d) if isinstance(r, tuple) else r / d
            for r, d in zip(range_keV, der)
        ]
        n_bins = [
            sum(r) / 0.5 / d if isinstance(r, tuple) else r / 0.2 / d
            for r, d in zip(range_keV, der)
        ]

    pk_pars, pk_errors, pk_covs, pk_binws, pk_ranges, pk_pvals = hpge_fit_E_peaks(
        E_uncal,
        got_peaks_locs,
        range_uncal,
        n_bins=n_bins,
        funcs=funcs,
        method=method,
        gof_funcs=gof_funcs,
        n_events=n_events,
        uncal_is_int=False,
        simplex=simplex,
        allowed_p_val=allowed_p_val,
    )
    results["pk_pars"] = pk_pars
    results["pk_errors"] = pk_errors
    results["pk_covs"] = pk_covs
    results["pk_binws"] = pk_binws
    results["pk_ranges"] = pk_ranges
    results["pk_pvals"] = pk_pvals
    # Drop failed fits
    fitidx = [i is not None for i in pk_pars]
    fitted_peaks_keV = results["fitted_keV"] = got_peaks_keV[fitidx]
    funcs = [f for i, f in zip(fitidx, funcs) if i]
    pk_pars = results["pk_pars"] = np.asarray(pk_pars, dtype=object)[fitidx]  # ragged
    pk_errors = results["pk_errors"] = np.asarray(pk_errors, dtype=object)[fitidx]
    pk_covs = results["pk_covs"] = np.asarray(pk_covs, dtype=object)[fitidx]
    pk_binws = results["pk_binws"] = np.asarray(pk_binws)[fitidx]
    pk_ranges = results["pk_ranges"] = np.asarray(pk_ranges)[fitidx]
    pk_pvals = results["pk_pvals"] = np.asarray(pk_pvals)[fitidx]
    log.info(f"{sum(fitidx)} peaks fitted:")
    for i, (Ei, parsi, errorsi, covsi) in enumerate(
        zip(fitted_peaks_keV, pk_pars, pk_errors, pk_covs)
    ):
        func_i = funcs[i] if hasattr(funcs, "__len__") else funcs
        varnames = func_i.__code__.co_varnames[1 : len(pk_pars[-1]) + 1]
        parsi = np.asarray(parsi, dtype=float)
        errorsi = np.asarray(errorsi, dtype=float)
        covsi = np.asarray(covsi, dtype=float)
        # parsigsi = np.sqrt(covsi.diagonal())
        log.info(f"\tEnergy: {str(Ei)}")
        log.info(f"\t\tParameter  |    Value +/- Sigma  ")
        for (vari, pari, errorsi) in zip(varnames, parsi, errorsi):
            log.info(
                f'\t\t{str(vari).ljust(10)} | {("%4.2f" % pari).rjust(8)} +/- {("%4.2f" % errorsi).ljust(8)}'
            )
            # fwhm??

    # Do a second calibration to the results of the full peak fits
    mus = [
        pgf.get_mu_func(func_i, pars_i, errors=errors_i)
        for func_i, pars_i, errors_i in zip(funcs, pk_pars, pk_errors)
    ]
    mus, mu_vars = zip(*mus)
    mus = np.asarray(mus)
    mu_vars = np.asarray(mu_vars) ** 2

    try:
        pars, cov = hpge_fit_E_scale(mus, mu_vars, fitted_peaks_keV, deg=deg)
        results["pk_cal_pars"] = pars
        results["pk_cal_cov"] = cov
    except ValueError:
        log.error("Failed to fit enough peaks to get accurate calibration")
        return None, None, results

    # Invert the E scale fit to get a calibration function
    pars, cov = hpge_fit_E_cal_func(mus, mu_vars, fitted_peaks_keV, pars, deg=deg)

    # Finally, calculate fwhms in keV
    uncal_fwhms = [
        pgf.get_fwhm_func(func_i, pars_i, cov=covs_i)
        for func_i, pars_i, covs_i in zip(funcs, pk_pars, pk_covs)
    ]
    uncal_fwhms, uncal_fwhm_errs = zip(*uncal_fwhms)
    uncal_fwhms = np.asarray(uncal_fwhms)
    uncal_fwhm_errs = np.asarray(uncal_fwhm_errs)
    derco = np.polyder(np.poly1d(pars)).coefficients
    der = [pgf.poly(Ei, derco) for Ei in fitted_peaks_keV]

    cal_fwhms = uncal_fwhms * der
    cal_fwhms_errs = uncal_fwhm_errs * der
    results["pk_fwhms"] = np.asarray(
        [(u * d, e * d) for u, e, d in zip(uncal_fwhms, uncal_fwhm_errs, der)]
    )

    log.info(f"{len(cal_fwhms)} FWHMs found:")
    log.info(f"\t   Energy   | FWHM  ")
    for i, (Ei, fwhm, fwhme) in enumerate(
        zip(fitted_peaks_keV, cal_fwhms, cal_fwhms_errs)
    ):
        log.info(
            f"\t{i}".ljust(4)
            + str(Ei).ljust(9)
            + f"| {fwhm:.2f}+-{fwhme:.2f} keV".ljust(5)
        )

    return pars, cov, results


def poly_match(xx, yy, deg=-1, rtol=1e-5, atol=1e-8):
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
            pars_i = np.array([1, (np.sum(yy_i) - np.sum(xx_i)) / len(yy_i)])
            polxx = xx_i + pars_i[1]

        # simple scaling
        elif deg == 0:
            pars_i = np.array([np.sum(yy_i * xx_i) / np.sum(xx_i * xx_i), 0])
            polxx = pars_i[0] * xx_i

        # generic poly of degree >= 1
        else:
            pars_i = np.polyfit(xx_i, yy_i, deg)
            polxx = np.zeros(len(yy_i))
            xxn = np.ones(len(yy_i))
            polxx = pgf.poly(xx_i, pars_i)

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


def get_i_local_extrema(data, delta):
    """Get lists of indices of the local maxima and minima of data

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
    imaxes, imins : 2-tuple ( array, array )
        A 2-tuple containing arrays of variable length that hold the indices of
        the identified local maxima (first tuple element) and minima (second
        tuple element)
    """

    # prepare output
    imaxes, imins = [], []

    # sanity checks
    data = np.asarray(data)
    if not np.isscalar(delta):
        log.error("get_i_local_extrema: Input argument delta must be a scalar")
        return np.array(imaxes), np.array(imins)
    if delta <= 0:
        log.error(f"get_i_local_extrema: delta ({delta}) must be positive")
        return np.array(imaxes), np.array(imins)

    # now loop over data
    imax, imin = 0, 0
    find_max = True
    for i in range(len(data)):

        if data[i] > data[imax]:
            imax = i
        if data[i] < data[imin]:
            imin = i

        if find_max:
            # if the sample is less than the current max by more than delta,
            # declare the previous one a maximum, then set this as the new "min"
            if data[i] < data[imax] - delta:
                imaxes.append(imax)
                imin = i
                find_max = False
        else:
            # if the sample is more than the current min by more than delta,
            # declare the previous one a minimum, then set this as the new "max"
            if data[i] > data[imin] + delta:
                imins.append(imin)
                imax = i
                find_max = True

    return np.array(imaxes), np.array(imins)


def get_i_local_maxima(data, delta):
    return get_i_local_extrema(data, delta)[0]


def get_i_local_minima(data, delta):
    return get_i_local_extrema(data, delta)[1]


def get_most_prominent_peaks(
    energySeries, xlo, xhi, xpb, max_num_peaks=np.inf, test=False
):
    """find the most prominent peaks in a spectrum by looking for spikes in
    derivative of spectrum energySeries: array of measured energies
    max_num_peaks = maximum number of most prominent peaks to find return a
    histogram around the most prominent peak in a spectrum of a given
    percentage of width
    """
    nb = int((xhi - xlo) / xpb)
    hist, bin_edges = np.histogram(energySeries, range=(xlo, xhi), bins=nb)
    bin_centers = pgh.get_bin_centers(bin_edges)

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
        plt.plot(bin_centers, hist, ls="steps", lw=1, c="b")
        for e in peak_energies:
            plt.axvline(e, color="r", lw=1, alpha=0.6)
        plt.xlabel("Energy [uncal]", ha="right", x=1)
        plt.ylabel("Filtered Spectrum", ha="right", y=1)
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
    for i, cal_set in enumerate(cal_sets):

        cal = cal_pks[list(cal_set)]  # lit energies for this set

        for data_set in data_sets:

            data = data_pks[list(data_set)]  # uncal energies for this set

            m, b, _, _, _ = linregress(data, y=cal)
            err = np.sum((cal - (m * data + b)) ** 2)

            if err < best_err:
                best_err, best_m, best_b = err, m, b

    print(i, best_err)
    print("cal:", cal)
    print("data:", data)
    plt.scatter(data, cal, label=f"min.err:{err:.2e}")
    xs = np.linspace(data[0], data[-1], 10)
    plt.plot(
        xs, best_m * xs + best_b, c="r", label=f"y = {best_m:.2f} x + {best_b:.2f}"
    )
    plt.xlabel("Energy (uncal)", ha="right", x=1)
    plt.ylabel("Energy (keV)", ha="right", y=1)
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
            [238.632, 510.770, 583.191, 727.330, 860.564, 2614.553]
        )  # get_calibration_energies(peak_energies)
    else:
        cal_peaks = np.array(cal_peaks)

    if len(energy_series) < 100:
        return 1, 0

    # get 10 most prominent ~high e peaks
    max_adc = np.amax(energy_series)
    energy_hi = energy_series  # [ (energy_series > np.percentile(energy_series, 20)) & (energy_series < np.percentile(energy_series, 99.9))]

    peak_energies, peak_e_err = get_most_prominent_peaks(
        energy_hi,
    )
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
    # Do a real fit to every peak in peak_energies
    ###############################################
    max_adc = np.amax(energy_series)

    peak_num = len(cal_peaks)
    centers = np.zeros(peak_num)
    fit_result_map = {}
    bin_size = 0.2  # keV

    if plotFigure is not None:
        plot_map = {}

    for i, energy in enumerate(cal_peaks):
        window_width = 10  # keV
        window_width_in_adc = (window_width) / rough_kev_per_adc
        energy_in_adc = (energy - rough_kev_offset) / rough_kev_per_adc
        bin_size_adc = (bin_size) / rough_kev_per_adc

        peak_vals = energy_series[
            (energy_series > energy_in_adc - window_width_in_adc)
            & (energy_series < energy_in_adc + window_width_in_adc)
        ]

        peak_hist, bins = np.histogram(
            peak_vals,
            bins=np.arange(
                energy_in_adc - window_width_in_adc,
                energy_in_adc + window_width_in_adc + bin_size_adc,
                bin_size_adc,
            ),
        )
        bin_centers = pgh.get_bin_centers(bins)
        # plt.ion()
        # plt.figure()
        # plt.plot(bin_centers,peak_hist,  color="k", ls="steps")

        # inp = input("q to quit...")
        # if inp == "q": exit()

        try:
            guess_e, guess_sigma, guess_area = get_gaussian_guess(
                peak_hist, bin_centers
            )
        except IndexError:
            print(f"\n\nIt looks like there may not be a peak at {energy} keV")
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
            bin_centers, gauss(bin_centers, guess_e, guess_sigma, guess_area), color="b"
        )

        # inp = input("q to quit...")
        # if inp == "q": exit()

        bounds = (
            [0.9 * guess_e, 0.5 * guess_sigma, 0, 0, 0, 0, 0],
            [
                1.1 * guess_e,
                2 * guess_sigma,
                0.1,
                0.75,
                window_width_in_adc,
                10,
                5 * guess_area,
            ],
        )
        params = fit_binned(
            radford_peak,
            peak_hist,
            bin_centers,
            [guess_e, guess_sigma, 1e-3, 0.7, 5, 0, guess_area],
        )  # bounds=bounds)

        plt.plot(bin_centers, radford_peak(bin_centers, *params), color="r")

        # inp = input("q to quit...")
        # if inp == "q": exit()

        fit_result_map[energy] = params
        centers[i] = params[0]

        if plotFigure is not None:
            plot_map[energy] = (bin_centers, peak_hist)

    # Do a linear fit to find the calibration
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
                color="k",
            )
            fit = radford_peak(bin_centers, *params)
            ax_peak.plot(
                bin_centers * rough_kev_per_adc + rough_kev_offset, fit, color="b"
            )

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
        ax_spec.semilogy(pgh.get_bin_centers(bins), peak_hist, ls="steps-mid")
        ax_spec.set_xlabel("Energy [keV]")

    return linear_cal


def get_calibration_energies(cal_type):
    if cal_type == "th228":
        return np.array(
            [
                238,
                277,
                300,
                452,
                510.77,
                583.191,
                727,
                763,
                785,
                860.564,
                1620,
                2614.533,
            ],
            dtype="double",
        )

    elif cal_type == "uwmjlab":
        # return np.array([239, 295, 351, 510, 583, 609, 911, 969, 1120,
        #                  1258, 1378, 1401, 1460, 1588, 1764, 2204, 2615],
        #                 dtype="double")
        return np.array([239, 911, 1460, 1764, 2615], dtype="double")
    else:
        raise ValueError
