"""
This module provides routines for calculating and applying quality cuts
"""

from __future__ import annotations

import logging
import re

import awkward as ak
import lgdo.lh5 as lh5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lgdo.types import Table
from scipy import stats
from scipy.stats import chi2, skewnorm

import pygama.math.binned_fitting as pgf
import pygama.math.histogram as pgh
import pygama.pargen.energy_cal as pgc
from pygama.math.binned_fitting import goodness_of_fit
from pygama.math.distributions import exgauss, gaussian
from pygama.math.functions.sum_dists import SumDists
from pygama.math.unbinned_fitting import fit_unbinned

(x_lo, x_hi, n_sig, mu, sigma, n_bkg, tau) = range(7)
par_array = [(gaussian, [mu, sigma]), (exgauss, [mu, sigma, tau])]
gauss_on_exgauss_areas = SumDists(
    par_array,
    [n_sig, n_bkg],
    "areas",
    parameter_names=["x_lo", "x_hi", "n_sig", "mu", "sigma", "n_bkg", "tau"],
    name="gauss_on_exgauss_areas",
)

(x_lo, x_hi, n_sig, mu, sigma, tau1, n_bkg, tau2) = range(8)
par_array = [(exgauss, [mu, sigma, tau1]), (exgauss, [mu, sigma, tau2])]
double_exgauss = SumDists(
    par_array,
    [n_sig, n_bkg],
    "areas",
    parameter_names=["x_lo", "x_hi", "n_sig", "mu", "sigma", "tau1", "n_bkg", "tau2"],
    name="double_exgauss",
)


def skewed_fit(x, n_sig, mu, sigma, alpha):
    return n_sig, n_sig * skewnorm.pdf(x, alpha, mu, sigma)


def skewed_pdf(x, n_sig, mu, sigma, alpha):
    return n_sig * skewnorm.pdf(x, alpha, mu, sigma)


log = logging.getLogger(__name__)
sto = lh5.LH5Store()
mpl.use("agg")


def get_keys(in_data, cut_dict):
    """
    Get the keys of the data that are used in the cut dictionary
    """
    parameters = []
    for _, entry in cut_dict.items():
        if "cut_parameter" in entry:
            parameters.append(entry["cut_parameter"])
        else:
            parameters.append(entry["expression"])

    out_params = []
    if isinstance(in_data, dict):
        possible_keys = in_data.keys()
    elif isinstance(in_data, list):
        possible_keys = in_data
    for param in parameters:
        for key in possible_keys:
            if key in param:
                out_params.append(key)
    return np.unique(out_params).tolist()


def get_mode_stdev(par_array):
    idxs = (par_array > np.nanpercentile(par_array, 1)) & (
        par_array < np.nanpercentile(par_array, 99)
    )
    par_array = par_array[idxs]
    bin_width = np.nanpercentile(par_array, 55) - np.nanpercentile(par_array, 50)

    counts, start_bins, var = pgh.get_hist(
        par_array,
        range=(np.nanmin(par_array), np.nanmax(par_array)),
        dx=bin_width,
    )
    max_idx = np.argmax(counts)
    mu = start_bins[max_idx]
    try:
        fwhm = pgh.get_fwhm(counts, start_bins)[0]
        guess_sig = fwhm / 2.355

        lower_bound = mu - 10 * guess_sig

        upper_bound = mu + 10 * guess_sig

    except Exception:
        lower_bound = np.nanpercentile(par_array, 5)
        upper_bound = np.nanpercentile(par_array, 95)

    if (lower_bound < np.nanmin(par_array)) or (lower_bound > np.nanmax(par_array)):
        lower_bound = np.nanmin(par_array)
    if (upper_bound > np.nanmax(par_array)) or (upper_bound < np.nanmin(par_array)):
        upper_bound = np.nanmax(par_array)

    try:
        counts, bins, var = pgh.get_hist(
            par_array,
            dx=(np.nanpercentile(par_array, 52) - np.nanpercentile(par_array, 50)),
            range=(lower_bound, upper_bound),
        )

        bin_centres = pgh.get_bin_centers(bins)

        fwhm = pgh.get_fwhm(counts, bins)[0]
        mean = float(bin_centres[np.argmax(counts)])
        pars, cov = pgf.gauss_mode_width_max(
            counts,
            bins,
            mode_guess=mean,
            n_bins=20,
            cost_func="Least Squares",
            inflate_errors=False,
            gof_method="var",
        )
        mean = pars[0]
        std = fwhm / 2.355

        if (
            mean < np.nanmin(bins)
            or mean > np.nanmax(bins)
            or (mean + std) < mu
            or (mean - std) > mu
        ):
            raise IndexError
    except IndexError:
        try:
            fwhm = pgh.get_fwhm(counts, bins)[0]
            mean = float(bin_centres[np.argmax(counts)])
            std = fwhm / 2.355
        except Exception:
            lower_bound = np.nanpercentile(par_array, 5)
            upper_bound = np.nanpercentile(par_array, 95)

            counts, bins, var = pgh.get_hist(
                par_array,
                dx=np.nanpercentile(par_array, 52) - np.nanpercentile(par_array, 50),
                range=(lower_bound, upper_bound),
            )

            bin_centres = pgh.get_bin_centers(bins)

            try:
                fwhm = pgh.get_fwhm(counts, bins)[0]
                mean = float(bin_centres[np.argmax(counts)])
                std = fwhm / 2.355
            except Exception:
                mean = float(bin_centres[np.argmax(counts)])
                std = np.nanstd(par_array)
    return mean, std


def fit_distributions(x_lo, x_hi, norm_par_array, display=0):
    peak_par_array = norm_par_array[(norm_par_array > x_lo) & (norm_par_array < x_hi)]

    hist, bins, var = pgh.get_hist(peak_par_array, dx=0.1, range=(x_lo, x_hi))
    var = np.where(var == 0, 1, var)

    exgauss_pars, _, _ = fit_unbinned(
        exgauss.pdf_ext,
        peak_par_array,
        [x_lo, x_hi, len(peak_par_array), 0, 1, -0.1],
        simplex=True,
        bounds=[
            (None, None),
            (None, None),
            (0, None),
            (-0.8, 0.8),
            (0.8, 1.2),
            (None, None),
        ],
        fixed=["x_lo", "x_hi"],
    )

    gauss_pars, _, _ = fit_unbinned(
        gaussian.pdf_ext,
        peak_par_array,
        [x_lo, x_hi, len(peak_par_array), 0, 1],
        simplex=True,
        bounds=[(None, None), (None, None), (0, None), (-0.5, 0.5), (0.5, 1.5)],
        fixed=["x_lo", "x_hi"],
    )

    gauss_on_exgauss_pars, _, _ = fit_unbinned(
        gauss_on_exgauss_areas.pdf_ext,
        peak_par_array,
        [x_lo, x_hi, len(peak_par_array) * 0.9, 0, 1, len(peak_par_array) * 0.1, -0.1],
        simplex=True,
        bounds=[
            (None, None),
            (None, None),
            (0, None),
            (-0.5, 0.5),
            (0, None),
            (0, None),
            (None, None),
        ],
        fixed=["x_lo", "x_hi"],
    )

    skewed_pars, _, _ = fit_unbinned(
        skewed_fit,
        peak_par_array,
        [len(peak_par_array), 0, 1, 0.1],
        simplex=True,
        bounds=[(0, None), (None, None), (0, None), (None, None)],
    )

    double_exgauss_pars, _, _ = fit_unbinned(
        double_exgauss.pdf_ext,
        peak_par_array,
        [
            x_lo,
            x_hi,
            len(peak_par_array) * 0.5,
            0,
            1,
            -0.1,
            len(peak_par_array) * 0.5,
            0.1,
        ],
        simplex=True,
        bounds=[
            (None, None),
            (None, None),
            (0, None),
            (-0.5, 0.5),
            (0, None),
            (None, 0),
            (0, None),
            (0, None),
        ],
        fixed=["x_lo", "x_hi"],
    )

    gauss_csqr = goodness_of_fit(
        hist,
        bins,
        var,
        lambda x, *args: gaussian.pdf_ext(x, *args)[1],
        gauss_pars,
        method="var",
        scale_bins=True,
    )

    exgauss_csqr = goodness_of_fit(
        hist,
        bins,
        var,
        lambda x, *args: exgauss.pdf_ext(x, *args)[1],
        exgauss_pars,
        method="var",
        scale_bins=True,
    )

    skewed_csqr = goodness_of_fit(
        hist,
        bins,
        var,
        lambda x, *args: skewed_fit(x, *args)[1],
        skewed_pars,
        method="var",
        scale_bins=True,
    )

    gauss_on_exgauss_csqr = goodness_of_fit(
        hist,
        bins,
        var,
        gauss_on_exgauss_areas.get_pdf,
        gauss_on_exgauss_pars,
        method="var",
        scale_bins=True,
    )

    double_exgauss_csqr = goodness_of_fit(
        hist,
        bins,
        var,
        double_exgauss.get_pdf,
        double_exgauss_pars,
        method="var",
        scale_bins=True,
    )

    if display > 0:
        bcs = pgh.get_bin_centers(bins)
        plt.figure()
        plt.step(bcs, hist)
        plt.plot(
            bcs, double_exgauss.get_pdf(bcs, *double_exgauss_pars) * np.diff(bins)[0]
        )
        plt.plot(
            bcs,
            gauss_on_exgauss_areas.pdf_ext(bcs, *gauss_on_exgauss_pars)[1]
            * np.diff(bins)[0],
        )
        plt.plot(bcs, skewed_fit(bcs, *skewed_pars)[1] * np.diff(bins)[0])
        plt.plot(bcs, gaussian.pdf_ext(bcs, *gauss_pars)[1] * np.diff(bins)[0])
        plt.plot(bcs, exgauss.pdf_ext(bcs, *exgauss_pars)[1] * np.diff(bins)[0])
        plt.show()

    gauss_p_val = chi2.sf(gauss_csqr[0], gauss_csqr[1] + 2)
    exgauss_p_val = chi2.sf(exgauss_csqr[0], exgauss_csqr[1] + 2)
    skewed_p_val = chi2.sf(skewed_csqr[0], skewed_csqr[1])
    gauss_on_exgauss_p_val = chi2.sf(
        gauss_on_exgauss_csqr[0], gauss_on_exgauss_csqr[1] + 2
    )
    double_exgauss_p_val = chi2.sf(double_exgauss_csqr[0], double_exgauss_csqr[1] + 2)

    funcs = [gaussian, exgauss, skewed_fit, gauss_on_exgauss_areas, double_exgauss]
    pars = [
        gauss_pars,
        exgauss_pars,
        skewed_pars,
        gauss_on_exgauss_pars,
        double_exgauss_pars,
    ]
    pvals = np.array(
        [
            gauss_p_val,
            exgauss_p_val,
            skewed_p_val,
            gauss_on_exgauss_p_val,
            double_exgauss_p_val,
        ]
    )
    csqrs = [
        gauss_csqr[0],
        exgauss_csqr[0],
        skewed_csqr[0],
        gauss_on_exgauss_csqr[0],
        double_exgauss_csqr[0],
    ]

    if (pvals == 0).all():
        idx = np.nanargmin(csqrs)
    else:
        idx = np.nanargmax(pvals)
    func = funcs[idx]
    pars = pars[idx]
    return func, pars


def generate_cuts(
    data: dict[str, np.ndarray],
    cut_dict: dict[str, int],
    rounding: int = 4,
    display: int = 0,
) -> dict:
    """
    Finds double sided cut boundaries for a file for the parameters specified

    Parameters
    ----------
    data : lh5 table, dictionary of arrays or pandas dataframe
        data to calculate cuts on

    parameters : dict
        dictionary of the form:

        .. code-block:: json

            {
                "output_parameter_name": {
                    "cut_parameter": "parameter_to_cut_on",
                    "cut_level": "number_of_sigmas",
                    "mode": "inclusive/exclusive"
                }
            }

        number of sigmas can instead be a dictionary to specify different cut levels for low and high side
        or to only have a one sided cut only specify one of the low or high side
        e.g.

        .. code-block:: json

            {
                "output_parameter_name": {
                    "cut_parameter": "parameter_to_cut_on",
                    "cut_level": {"low_side": "3", "high_side": "2"},
                    "mode": "inclusive/exclusive"
                }
            }

        alternatively can specify hit dict fields to just copy dict into output dict e.g.

        .. code-block:: json

            {
                "is_valid_t0":{
                    "expression":"(tp_0_est>a)&(tp_0_est<b)",
                    "parameters":{"a":"46000", "b":"52000"}
                }
            }

        or

        .. code-block:: json

            {
                "is_valid_cal":{
                    "expression":"(~is_pileup_tail)&(~is_pileup_baseline)"
                }
            }

    rounding : int
        number of decimal places to round to

    display : int
        if 1 will display plots of the cuts
        if 0 will not display plots

    Returns
    -------
    dict
        dictionary of the form (same as hit dicts):

        .. code-block:: python

            {
                "output_parameter_name": {
                    "expression": "cut_expression",
                    "parameters": {"a": "lower_bound", "b": "upper_bound"}
                }
            }

    plot_dict
        dictionary of plots
    """

    output_dict = {}
    plot_dict = {}
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, Table):
        data = {entry: data[entry].nda for entry in get_keys(data, cut_dict)}
        data = pd.DataFrame.from_dict(data)
    elif isinstance(data, dict):
        data = pd.DataFrame.from_dict(data)
    for out_par, cut in cut_dict.items():
        if "expression" in cut:
            output_dict[out_par] = {"expression": cut["expression"]}
            if "parameters" in cut:
                output_dict[out_par].update({"parameters": cut["parameters"]})
        else:
            par = cut["cut_parameter"]
            num_sigmas = cut["cut_level"]
            mode = cut["mode"]
            try:
                all_par_array = data[par].to_numpy()
            except KeyError:
                all_par_array = data.eval(par).to_numpy()

            mean, std = get_mode_stdev(all_par_array)

            if isinstance(num_sigmas, (int, float)):
                num_sigmas_left = num_sigmas
                num_sigmas_right = num_sigmas
            elif isinstance(num_sigmas, dict):
                if "low_side" in num_sigmas:
                    num_sigmas_left = num_sigmas["low_side"]
                else:
                    num_sigmas_left = None
                if "high_side" in num_sigmas:
                    num_sigmas_right = num_sigmas["high_side"]
                else:
                    num_sigmas_right = None
            upper = round(float((num_sigmas_right * std) + mean), rounding)
            lower = round(float((-num_sigmas_left * std) + mean), rounding)
            if mode == "inclusive":
                if upper is not None and lower is not None:
                    cut_string = f"({par}>a) & ({par}<b)"
                    par_dict = {"a": lower, "b": upper}
                elif upper is None:
                    cut_string = f"{par}>a"
                    par_dict = {"a": lower}
                elif lower is None:
                    cut_string = f"{par}<a"
                    par_dict = {"a": upper}
            elif mode == "exclusive":
                if upper is not None and lower is not None:
                    cut_string = f"({par}<a) | ({par}>b)"
                    par_dict = {"a": lower, "b": upper}
                elif upper is None:
                    cut_string = f"{par}<a"
                    par_dict = {"a": lower}
                elif lower is None:
                    cut_string = f"{par}>a"
                    par_dict = {"a": upper}

            output_dict[out_par] = {"expression": cut_string, "parameters": par_dict}

            if display > 0:
                fig = plt.figure()
                low_val = np.nanpercentile(all_par_array, 5)
                up_val = np.nanpercentile(all_par_array, 95)
                if upper is not None:
                    plt.axvline(upper)
                    if up_val < upper:
                        up_val = upper
                if lower is not None:
                    plt.axvline(lower)
                    if low_val > lower:
                        low_val = lower

                plt.hist(
                    all_par_array,
                    bins=np.linspace(
                        low_val,
                        up_val,
                        100,
                    ),
                    histtype="step",
                )

                plt.ylabel("counts")
                plt.xlabel(out_par)
                plot_dict[out_par] = fig
                plt.close()
    if display > 0:
        return output_dict, plot_dict
    else:
        return output_dict


def get_cut_indexes(data, cut_parameters):
    """
    Get the indexes of the data that pass the cuts in
    """
    cut_dict = generate_cuts(data, cut_dict=cut_parameters)
    log.debug(f"Cuts are {cut_dict}")

    if isinstance(data, Table):
        ct_mask = np.full(len(data), True, dtype=bool)
        for outname, info in cut_dict.items():
            outcol = data.eval(info["expression"], info.get("parameters", None))
            data.add_column(outname, outcol)
        log.debug("Applied Cuts")

        for cut in cut_dict:
            ct_mask = data[cut].nda & ct_mask
    elif isinstance(data, pd.DataFrame):
        ct_mask = np.full(len(data), True, dtype=bool)

        for outname, info in cut_dict.items():
            # convert to pandas eval
            exp = info["expression"]
            for key in info.get("parameters", None):
                exp = re.sub(f"(?<![a-zA-Z0-9]){key}(?![a-zA-Z0-9])", f"@{key}", exp)
            data[outname] = data.eval(exp, local_dict=info.get("parameters", None))
            ct_mask = ct_mask & data[outname]
    else:
        raise ValueError("Data must be a Table or DataFrame")

    return ct_mask


def generate_cut_classifiers(
    data: dict[str, np.ndarray],
    cut_dict: dict[str, int],
    rounding: int = 4,
    display: int = 0,
) -> dict:
    """
    Finds double sided cut boundaries for a file for the parameters specified

    Parameters
    ----------
    data : lh5 table, dictionary of arrays or pandas dataframe
        data to calculate cuts on

    parameters : dict
        dictionary of the form:

        .. code-block:: json

            {
                "output_parameter_name": {
                    "cut_parameter": "parameter_to_cut_on",
                    "cut_level": "number_of_sigmas",
                     "mode": "inclusive/exclusive"
                }
            }

        number of sigmas can instead be a dictionary to specify different cut levels for low and high side
        or to only have a one sided cut only specify one of the low or high side
        e.g.

        .. code-block:: json

            {
                "output_parameter_name": {
                    "cut_parameter": "parameter_to_cut_on",
                    "cut_level": {"low_side": "3", "high_side": "2"},
                    "mode": "inclusive/exclusive"
                }
            }

        alternatively can specify hit dict fields to just copy dict into output dict e.g.

        .. code-block:: json

            {
                "is_valid_t0":{
                    "expression":"(tp_0_est>a)&(tp_0_est<b)",
                    "parameters":{"a":"46000", "b":"52000"}
                }
            }

        or

        .. code-block:: json

            {
                "is_valid_cal":{
                    "expression":"(~is_pileup_tail)&(~is_pileup_baseline)"
                }
            }

    rounding : int
        number of decimal places to round to

    display : int
        if 1 will display plots of the cuts
        if 0 will not display plots

    Returns
    -------
    dict
        dictionary of the form (same as hit dicts):

        .. code-block:: python

            {
                "output_parameter_name": {
                    "expression": "cut_expression",
                    "parameters": {"a": lower_bound, "b": upper_bound}
                }
            }

    plot_dict
        dictionary of plots
    """

    output_dict = {}
    plot_dict = {}
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, Table):
        data = {entry: data[entry].nda for entry in get_keys(data, cut_dict)}
        data = pd.DataFrame.from_dict(data)
    elif isinstance(data, dict):
        data = pd.DataFrame.from_dict(data)
    for out_par, cut in cut_dict.items():
        if "expression" in cut:
            output_dict[out_par] = {"expression": cut["expression"]}
            if "parameters" in cut:
                output_dict[out_par].update({"parameters": cut["parameters"]})
        else:
            par = cut["cut_parameter"]
            num_sigmas = cut.get("cut_level", None)
            percentile = cut.get("cut_percentile", None)
            default = cut.get("default", None)
            method = cut.get("method", "fit")
            mode = cut["mode"]
            try:
                all_par_array = data[par].to_numpy()
            except KeyError:
                all_par_array = data.eval(par).to_numpy()

            mean, std = get_mode_stdev(all_par_array)

            norm_par_array = (all_par_array - mean) / std

            if num_sigmas is not None:
                if isinstance(num_sigmas, (int, float)):
                    cut_left = -num_sigmas
                    cut_right = num_sigmas
                elif isinstance(num_sigmas, dict):
                    if "low_side" in num_sigmas:
                        cut_left = num_sigmas["low_side"]
                    else:
                        cut_left = None
                    if "high_side" in num_sigmas:
                        cut_right = num_sigmas["high_side"]
                    else:
                        cut_right = None

            elif percentile is not None:
                if method == "fit":
                    try:
                        x_lo = -10
                        x_hi = 10
                        func, pars = fit_distributions(x_lo, x_hi, norm_par_array)

                    except RuntimeError:
                        x_lo = -20
                        x_hi = 20
                        func, pars = fit_distributions(x_lo, x_hi, norm_par_array)

                    range_low, range_high = (-100, 100)
                    xs = np.arange(range_low, range_high, 0.1)
                    if func == exgauss:
                        cdf = exgauss.cdf_norm(
                            xs,
                            range_low,
                            range_high,
                            pars["mu"],
                            pars["sigma"],
                            pars["tau"],
                        )
                    elif func == gaussian:
                        cdf = gaussian.cdf_norm(
                            xs,
                            range_low,
                            range_high,
                            pars["mu"],
                            pars["sigma"],
                        )
                    elif func == gauss_on_exgauss_areas or func == double_exgauss:
                        cdf = func.cdf_norm(xs, range_low, range_high, *pars[2:])
                    elif func == skewed_fit:
                        cdf = skewnorm.cdf(xs, pars["alpha"], pars["mu"], pars["sigma"])
                    else:
                        raise ValueError("unknown func")

                    if isinstance(percentile, (int, float)):
                        cut_left = xs[np.argmin(np.abs(cdf - (1 - (percentile / 100))))]
                        cut_right = xs[np.argmin(np.abs(cdf - (percentile / 100)))]

                    elif isinstance(percentile, dict):
                        if "low_side" in percentile:
                            cut_left = xs[
                                np.argmin(np.abs(cdf - (1 - (percentile / 100))))
                            ]
                        else:
                            cut_left = None
                        if "high_side" in percentile:
                            cut_right = xs[np.argmin(np.abs(cdf - (percentile / 100)))]
                        else:
                            cut_right = None

                else:
                    if isinstance(percentile, (int, float)):
                        cut_left = np.nanpercentile(norm_par_array, 100 - percentile)
                        cut_right = np.nanpercentile(norm_par_array, percentile)

                    elif isinstance(percentile, dict):
                        if "low_side" in percentile:
                            cut_left = np.nanpercentile(norm_par_array, percentile)
                        else:
                            cut_left = None
                        if "high_side" in percentile:
                            cut_right = np.nanpercentile(norm_par_array, percentile)
                        else:
                            cut_right = None

            if default is not None:
                value = default["value"]
                default_mode = default["mode"]
                if isinstance(value, (int, float)):
                    default_cut_left = -value
                    default_cut_right = value
                else:
                    if "low_side" in default:
                        default_cut_left = value["low_side"]
                    else:
                        default_cut_left = np.nan
                    if "high_side" in default:
                        default_cut_right = value["high_side"]
                    else:
                        default_cut_right = np.nan

                if default_mode == "higher_limit":
                    if cut_left is not None:
                        if cut_left < default_cut_left:
                            cut_left = default_cut_left
                    if cut_right is not None:
                        if cut_right > default_cut_right:
                            cut_right = default_cut_right
                elif default_mode == "lower_limit":
                    if cut_left is not None:
                        if cut_left > default_cut_left:
                            cut_left = default_cut_left
                    if cut_right is not None:
                        if cut_right < default_cut_right:
                            cut_right = default_cut_right
                else:
                    raise ValueError("unknown mode")

            if mode == "inclusive":
                if cut_right is not None and cut_left is not None:
                    cut_string = f"({out_par}_classifier>a) & ({out_par}_classifier<b)"
                    par_dict = {"a": cut_left, "b": cut_right}
                elif cut_right is None:
                    cut_string = f"{out_par}_classifier>a"
                    par_dict = {"a": cut_left}
                elif cut_left is None:
                    cut_string = f"{out_par}_classifier<a"
                    par_dict = {"a": cut_right}
            elif mode == "exclusive":
                if cut_right is not None and cut_left is not None:
                    cut_string = f"({out_par}_classifier<a) | ({out_par}_classifier>b)"
                    par_dict = {"a": cut_left, "b": cut_right}
                elif cut_right is None:
                    cut_string = f"{out_par}_classifier<a"
                    par_dict = {"a": cut_left}
                elif cut_left is None:
                    cut_string = f"{out_par}_classifier>a"
                    par_dict = {"a": cut_right}

            output_dict[f"{out_par}_classifier"] = {
                "expression": f"(({par})-a)/b",
                "parameters": {"a": mean, "b": std},
            }

            output_dict[out_par] = {"expression": cut_string, "parameters": par_dict}
            if display > 0:
                fig = plt.figure()
                low = -10 if cut_left is None or cut_left > -10 else cut_left
                hi = 10 if cut_right is None or cut_right < 10 else cut_right
                hist, _, _ = plt.hist(
                    norm_par_array,
                    bins=np.arange(low, hi, 0.1),
                    histtype="step",
                )
                if percentile is not None and method == "fit":
                    xs = np.arange(low, hi, 0.1)
                    if func == skewed_fit:
                        pdf_values = func(xs, *pars)[1] * 0.1
                    else:
                        pdf_values = func.pdf_ext(xs, *pars)[1] * 0.1
                    plt.plot(xs, pdf_values)
                if cut_left is not None:
                    plt.axvline(cut_left)
                if cut_right is not None:
                    plt.axvline(cut_right)

                plt.ylabel("counts")
                plt.xlabel(f"{out_par}_classifier")
                plot_dict[out_par] = fig
                plt.close()
    if display > 0:
        return output_dict, plot_dict
    else:
        return output_dict


def find_pulser_properties(df, energy="daqenergy"):
    """
    Searches for pulser in the energy spectrum using time between events in peaks
    """
    if np.nanmax(df[energy]) > 8000:
        hist, bins, var = pgh.get_hist(
            df[energy], dx=1, range=(1000, np.nanmax(df[energy]))
        )
        allowed_err = 200
    else:
        hist, bins, var = pgh.get_hist(
            df[energy], dx=0.2, range=(500, np.nanmax(df[energy]))
        )
        allowed_err = 50
    if np.any(var == 0):
        var[np.where(var == 0)] = 1
    imaxes = pgc.get_i_local_maxima(hist / np.sqrt(var), 3)
    peak_energies = pgh.get_bin_centers(bins)[imaxes]
    pt_pars, pt_covs = pgc.hpge_fit_E_peak_tops(
        hist, bins, var, peak_energies, n_to_fit=10
    )
    peak_e_err = pt_pars[:, 1] * 4

    allowed_mask = np.ones(len(peak_energies), dtype=bool)
    for i, e in enumerate(peak_energies[1:-1]):
        i += 1
        if peak_e_err[i] > allowed_err:
            continue
        if i == 1:
            if (
                e - peak_e_err[i] < peak_energies[i - 1] + peak_e_err[i - 1]
                and peak_e_err[i - 1] < allowed_err
            ):
                overlap = (
                    peak_energies[i - 1]
                    + peak_e_err[i - 1]
                    - (peak_energies[i] - peak_e_err[i])
                )
                peak_e_err[i] -= overlap * (
                    peak_e_err[i] / (peak_e_err[i] + peak_e_err[i - 1])
                )
                peak_e_err[i - 1] -= overlap * (
                    peak_e_err[i - 1] / (peak_e_err[i] + peak_e_err[i - 1])
                )

        if (
            e + peak_e_err[i] > peak_energies[i + 1] - peak_e_err[i + 1]
            and peak_e_err[i + 1] < allowed_err
        ):
            overlap = (e + peak_e_err[i]) - (peak_energies[i + 1] - peak_e_err[i + 1])
            total = peak_e_err[i] + peak_e_err[i + 1]
            peak_e_err[i] -= (overlap) * (peak_e_err[i] / total)
            peak_e_err[i + 1] -= (overlap) * (peak_e_err[i + 1] / total)

    out_pulsers = []
    for i, e in enumerate(peak_energies[allowed_mask]):
        if peak_e_err[i] > allowed_err:
            continue

        try:
            e_cut = (df[energy] > e - peak_e_err[i]) & (df[energy] < e + peak_e_err[i])
            df_peak = df[e_cut]

            time_since_last = (
                df_peak.timestamp.values[1:] - df_peak.timestamp.values[:-1]
            )

            tsl = time_since_last[
                (time_since_last >= 0)
                & (time_since_last < np.percentile(time_since_last, 99.9))
            ]

            bins = np.arange(0.1, 5, 0.001)
            bcs = pgh.get_bin_centers(bins)
            hist, bins, var = pgh.get_hist(tsl, bins=bins)

            maxs = pgh.get_i_local_maxima(hist, 45)
            maxs = maxs[maxs > 20]

            super_max = pgh.get_i_local_maxima(hist, 500)
            super_max = super_max[super_max > 20]
            if len(maxs) < 2:
                continue
            else:
                max_locs = np.array([0.0])
                max_locs = np.append(max_locs, bcs[np.array(maxs)])
                if (
                    len(np.where(np.abs(np.diff(np.diff(max_locs))) <= 0.001)[0]) > 1
                    or (np.abs(np.diff(np.diff(max_locs))) <= 0.001).all()
                    or len(super_max) > 0
                ):
                    pulser_e = e
                    period = stats.mode(tsl).mode[0]
                    if period > 0.1:
                        out_pulsers.append((pulser_e, peak_e_err[i], period, energy))

                else:
                    continue
        except Exception:
            continue
    return out_pulsers


def get_tcm_pulser_ids(tcm_file, channel, multiplicity_threshold):
    if isinstance(channel, str):
        if channel[:2] == "ch":
            chan = int(channel[2:])
        else:
            chan = int(channel)
    else:
        chan = channel
    if isinstance(tcm_file, list):
        mask = np.array([], dtype=bool)
        for file in tcm_file:
            _, file_mask = get_tcm_pulser_ids(file, chan, multiplicity_threshold)
            mask = np.append(mask, file_mask)
        ids = np.where(mask)[0]
    else:
        array_ids = lh5.read("hardware_tcm_1/table_key", tcm_file).view_as("ak")
        chan_evts = ak.any(array_ids == chan, axis=-1)
        multiplicity = ak.count(array_ids[chan_evts], axis=-1)
        ids = np.where(multiplicity > multiplicity_threshold)[0]
        mask = np.zeros(len(array_ids[chan_evts]), dtype=bool)
        mask[ids] = True
    return ids, mask


def tag_pulsers(df, chan_info, window=0.01):
    df["isPulser"] = 0

    if isinstance(chan_info, tuple):
        chan_info = [chan_info]
    final_mask = None
    for chan_i in chan_info:
        pulser_energy, peak_e_err, period, energy_name = chan_i

        e_cut = (df[energy_name] < pulser_energy + peak_e_err) & (
            df[energy_name] > pulser_energy - peak_e_err
        )
        df_pulser = df[e_cut]

        time_since_last = np.zeros(len(df_pulser))
        time_since_last[1:] = (
            df_pulser.timestamp.values[1:] - df_pulser.timestamp.values[:-1]
        )

        mode_idxs = (time_since_last > period - window) & (
            time_since_last < period + window
        )

        pulser_events = np.count_nonzero(mode_idxs)
        # print(f"pulser events: {pulser_events}")
        if pulser_events < 3:
            return df
        df_pulser = df_pulser[mode_idxs]

        ts = df_pulser.timestamp.values
        diff_zero = np.zeros(len(ts))
        diff_zero[1:] = np.around(np.divide(np.subtract(ts[1:], ts[:-1]), period))
        diff_cum = np.cumsum(diff_zero)
        z = np.polyfit(diff_cum, ts, 1)

        period = z[0]
        phase = z[1]
        mod = np.abs(df.timestamp - phase) % period

        period_cut = (mod < 0.1) | ((period - mod) < 0.1)  # 0.1)

        if final_mask is None:
            final_mask = e_cut & period_cut
        else:
            final_mask = final_mask | (e_cut & period_cut)

    df.loc[final_mask, "isPulser"] = 1

    return df
