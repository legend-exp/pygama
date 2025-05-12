"""
This module provides functions for calculating survival fractions for a cut.
"""

from __future__ import annotations

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from iminuit.util import ValueView

import pygama.pargen.energy_cal as pgc
from pygama.math.distributions import gauss_on_step, hpge_peak

log = logging.getLogger(__name__)


def energy_guess(energy, func_i, fit_range=None, bin_width=1, peak=None, eres=None):
    """
    Simple guess for peak fitting
    """
    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))
    if func_i == hpge_peak or func_i == gauss_on_step:
        parguess = pgc.get_hpge_energy_peak_par_guess(
            energy, func_i, fit_range=fit_range
        )

        if peak is not None:
            parguess["mu"] = peak

        if eres is not None:
            parguess["sigma"] = eres / 2.355

        for i, guess in enumerate(parguess):
            if np.isnan(guess):
                parguess[i] = 0

    else:
        log.error(f"energy_guess not implemented for {func_i}")
        return None
    return parguess


def fix_all_but_nevents(func):
    """
    Fixes all parameters except the number of signal and background events
    and their efficiencies
    Returns: Sequence list of fixed indexes for fitting and mask for parameters
    """

    if func == gauss_on_step:
        # pars are: n_sig, mu, sigma, n_bkg, hstep, lower, upper, components
        fixed = ["x_lo", "x_hi", "mu", "sigma", "hstep"]

    elif func == hpge_peak:
        # pars are: , components
        fixed = ["x_lo", "x_hi", "mu", "sigma", "htail", "tau", "hstep"]

    else:
        log.error(f"get_hpge_E_fixed not implemented for {func}")
        return None, None
    mask = ~np.in1d(func.required_args(), fixed)
    return fixed, mask


def get_bounds(func, parguess):
    """
    Gets bounds for the fit parameters
    """
    if func == hpge_peak or func == gauss_on_step:
        bounds = pgc.get_hpge_energy_bounds(func, parguess)

        bounds["mu"] = (parguess["mu"] - 1, parguess["mu"] + 1)
        bounds["n_sig"] = (0, 2 * (parguess["n_sig"] + parguess["n_bkg"]))
        bounds["n_bkg"] = (0, 2 * (parguess["n_sig"] + parguess["n_bkg"]))

    else:
        log.error(f"get_bounds not implemented for {func}")
        return None
    return bounds


def pass_pdf_gos(
    x, x_lo, x_hi, n_sig, epsilon_sig, n_bkg, epsilon_bkg, mu, sigma, hstep1, hstep2
):
    """
    pdf for gauss on step fit reparamerised to calculate the efficiency of the cut
    this is the passing pdf
    """
    return gauss_on_step.pdf_ext(
        x, x_lo, x_hi, n_sig * epsilon_sig, mu, sigma, n_bkg * epsilon_bkg, hstep1
    )


def fail_pdf_gos(
    x, x_lo, x_hi, n_sig, epsilon_sig, n_bkg, epsilon_bkg, mu, sigma, hstep1, hstep2
):
    """
    pdf for gauss on step fit reparamerised to calculate the efficiency of the cut
    this is the cut pdf
    """
    return gauss_on_step.pdf_ext(
        x,
        x_lo,
        x_hi,
        n_sig * (1 - epsilon_sig),
        mu,
        sigma,
        n_bkg * (1 - epsilon_bkg),
        hstep2,
    )


def pass_pdf_hpge(
    x,
    x_lo,
    x_hi,
    n_sig,
    epsilon_sig,
    n_bkg,
    epsilon_bkg,
    mu,
    sigma,
    htail,
    tau,
    hstep1,
    hstep2,
):
    """
    pdf for hpge peak fit reparamerised to calculate the efficiency of the cut.
    this is the passing pdf
    """
    return hpge_peak.pdf_ext(
        x,
        x_lo,
        x_hi,
        n_sig * epsilon_sig,
        mu,
        sigma,
        htail,
        tau,
        n_bkg * epsilon_bkg,
        hstep1,
    )


def fail_pdf_hpge(
    x,
    x_lo,
    x_hi,
    n_sig,
    epsilon_sig,
    n_bkg,
    epsilon_bkg,
    mu,
    sigma,
    htail,
    tau,
    hstep1,
    hstep2,
):
    """
    pdf for hpge peak fit reparamerised to calculate the efficiency of the cut
    this is the cut pdf
    """
    return hpge_peak.pdf_ext(
        x,
        x_lo,
        x_hi,
        n_sig * (1 - epsilon_sig),
        mu,
        sigma,
        htail,
        tau,
        n_bkg * (1 - epsilon_bkg),
        hstep2,
    )


def update_guess(func, parguess, energies):
    """
    Updates guess for the number of signal and background events
    """
    if func == gauss_on_step or func == hpge_peak:

        total_events = len(energies)
        parguess["n_sig"] = len(
            energies[
                (energies > parguess["mu"] - 2 * parguess["sigma"])
                & (energies < parguess["mu"] + 2 * parguess["sigma"])
            ]
        )
        parguess["n_sig"] -= len(
            energies[
                (energies > parguess["x_lo"])
                & (energies < parguess["x_lo"] + 2 * parguess["sigma"])
            ]
        )
        parguess["n_sig"] -= len(
            energies[
                (energies > parguess["x_hi"] - 2 * parguess["sigma"])
                & (energies < parguess["x_hi"])
            ]
        )
        parguess["n_bkg"] = total_events - parguess["n_sig"]
        return parguess

    else:
        log.error(f"update_guess not implemented for {func}")
        return parguess


def get_survival_fraction(
    energy: np.ndarray,
    cut_param: np.ndarray,
    cut_val: float,
    peak: float,
    eres_pars: float,
    fit_range: tuple = None,
    high_cut: float = None,
    pars: ValueView = None,
    data_mask: np.ndarray = None,
    mode: str = "greater",
    func=hpge_peak,
    fix_step=True,
    display=0,
):
    """
    Function for calculating the survival fraction of a cut
    using a fit to the surviving and failing energy distributions.

    Parameters
    ----------

    energy: array
        array of energies
    cut_param: array
        array of the cut parameter for the survival fraction calculation, should have the same length as energy
    cut_val: float
        value of the cut parameter to be used for the survival fraction calculation
    peak: float
        energy of the peak to be fitted
    eres_pars: float
        energy resolution parameter for the peak
    fit_range: tuple
        range of the fit in keV
    high_cut: float
        upper value for the cut parameter to have a range in the cut value
    pars: iMinuit ValueView
        initial parameters for the fit
    data_mask: array
        mask for the data to be used in the fit
    mode: str
        mode of the cut, either "greater" or "less"
    func: function
        function to be used in the fit
    fix_step: bool
        option to fix the step parameters in the fit
    display: int
        option to display the fit if greater than 1

    Returns
    -------

    sf: float
        survival fraction
    err: float
        error on the survival fraction
    values: iMinuit ValueView
        values of the parameters of the fit
    errors: iMinuit ValueView
        errors on the parameters of the fit
    """
    if data_mask is None:
        data_mask = np.full(len(cut_param), True, dtype=bool)

    if not isinstance(energy, np.ndarray):
        energy = np.array(energy)
    if not isinstance(cut_param, np.ndarray):
        cut_param = np.array(cut_param)

    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))

    nan_idxs = np.isnan(cut_param)
    if high_cut is not None:
        idxs = (cut_param > cut_val) & (cut_param < high_cut) & data_mask
    else:
        if mode == "greater":
            idxs = (cut_param > cut_val) & data_mask
        elif mode == "less":
            idxs = (cut_param < cut_val) & data_mask
        else:
            raise ValueError("mode not recognised")

    if pars is None:
        (pars, errs, cov, _, func, _, _, _) = pgc.unbinned_staged_energy_fit(
            energy,
            func,
            guess_func=energy_guess,
            bounds_func=get_bounds,
            guess_kwargs={"peak": peak, "eres": eres_pars},
            fit_range=fit_range,
        )

    guess_pars_surv = copy.deepcopy(pars)

    # add update guess here for n_sig and n_bkg
    guess_pars_surv = update_guess(func, guess_pars_surv, energy[(~nan_idxs) & (idxs)])

    parguess = {
        "x_lo": pars["x_lo"],
        "x_hi": pars["x_hi"],
        "mu": pars["mu"],
        "sigma": pars["sigma"],
        "hstep1": pars["hstep"],
        "hstep2": pars["hstep"],
        "n_sig": pars["n_sig"],
        "n_bkg": pars["n_bkg"],
        "epsilon_sig": guess_pars_surv["n_sig"] / pars["n_sig"],
        "epsilon_bkg": guess_pars_surv["n_bkg"] / pars["n_bkg"],
    }

    bounds = {
        "n_sig": (0, pars["n_sig"] + pars["n_bkg"]),
        "epsilon_sig": (0, 1),
        "n_bkg": (0, pars["n_bkg"] + pars["n_sig"]),
        "epsilon_bkg": (0, 1),
        "hstep1": (-1, 1),
        "hstep2": (-1, 1),
    }

    if func == hpge_peak:
        parguess.update({"htail": pars["htail"], "tau": pars["tau"]})

    if func == hpge_peak:
        lh = cost.ExtendedUnbinnedNLL(
            energy[(~nan_idxs) & (idxs)], pass_pdf_hpge
        ) + cost.ExtendedUnbinnedNLL(energy[(~nan_idxs) & (~idxs)], fail_pdf_hpge)
    elif func == gauss_on_step:
        lh = cost.ExtendedUnbinnedNLL(
            energy[(~nan_idxs) & (idxs)], pass_pdf_gos
        ) + cost.ExtendedUnbinnedNLL(energy[(~nan_idxs) & (~idxs)], fail_pdf_gos)

    else:
        raise ValueError("Unknown func")

    m = Minuit(lh, **parguess)
    fixed = ["x_lo", "x_hi", "n_sig", "n_bkg", "mu", "sigma"]  # "hstep"
    if func == hpge_peak:
        fixed += ["tau", "htail"]
    if fix_step is True:
        fixed += ["hstep1", "hstep2"]

    m.fixed[fixed] = True
    for arg, val in bounds.items():
        m.limits[arg] = val

    m.simplex().migrad()
    m.hesse()

    sf = m.values["epsilon_sig"] * 100
    err = m.errors["epsilon_sig"] * 100

    if display > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        bins = np.arange(1552, 1612, 1)
        ax1.hist(energy[(~nan_idxs) & (idxs)], bins=bins, histtype="step")

        ax2.hist(energy[(~nan_idxs) & (~idxs)], bins=bins, histtype="step")

        if func == hpge_peak:
            ax1.plot(bins, pass_pdf_hpge(bins, **m.values.to_dict())[1])
            ax2.plot(bins, fail_pdf_hpge(bins, **m.values.to_dict())[1])
        elif func == gauss_on_step:
            ax1.plot(bins, pass_pdf_gos(bins, **m.values.to_dict())[1])
            ax2.plot(bins, fail_pdf_gos(bins, **m.values.to_dict())[1])

        plt.show()

    return sf, err, m.values, m.errors


def get_sf_sweep(
    energy: np.array,
    cut_param: np.array,
    final_cut_value: float = None,
    peak: float = 1592.5,
    eres_pars: list = None,
    data_mask=None,
    cut_range=(-5, 5),
    n_samples=26,
    mode="greater",
    fit_range=None,
    debug_mode=False,
) -> tuple(pd.DataFrame, float, float):
    """
    Function sweeping through cut values and calculating the survival fraction for each value
    using a fit to the surviving and failing enegry distributions.

    Parameters
    ----------

    energy: array
        array of energies
    cut_param: array
        array of the cut parameter for the survival fraction calculation, should have the same length as energy
    final_cut_val: float
        value of the final cut parameter to be used for the survival fraction calculation
    peak: float
        energy of the peak to be fitted
    eres_pars: float
        energy resolution parameter for the peak
    data_mask: array
        mask for the data to be used in the fit
    cut_range: tuple
        range of the cut values to be swept through
    n_samples: int
        number of samples to be taken in the cut range
    mode: str
        mode of the cut, either "greater" or "less"
    fit_range: tuple
        range of the fit in keV
    debug_mode: bool
        option to raise an error if there is an issue with

    Returns
    -------

    out_df: Dataframe
        Dataframe of cut values with the survival fraction and error
    sf: float
        survival fraction
    err: float
        error on the survival fraction

    """

    if data_mask is None:
        data_mask = np.full(len(cut_param), True, dtype=bool)

    if not isinstance(energy, np.ndarray):
        energy = np.array(energy)
    if not isinstance(cut_param, np.ndarray):
        cut_param = np.array(cut_param)

    cut_vals = np.linspace(cut_range[0], cut_range[1], n_samples)
    out_df = pd.DataFrame()

    (pars, errs, _, _, func, _, _, _) = pgc.unbinned_staged_energy_fit(
        energy,
        hpge_peak,
        guess_func=energy_guess,
        bounds_func=get_bounds,
        guess_kwargs={"peak": peak, "eres": eres_pars},
        fit_range=fit_range,
    )

    for cut_val in cut_vals:
        try:
            sf, err, _, _ = get_survival_fraction(
                energy,
                cut_param,
                cut_val,
                peak,
                eres_pars,
                fit_range=fit_range,
                data_mask=data_mask,
                mode=mode,
                pars=pars,
                func=func,
            )
            out_df = pd.concat(
                [out_df, pd.DataFrame([{"cut_val": cut_val, "sf": sf, "sf_err": err}])]
            )
        except BaseException as e:
            if e == KeyboardInterrupt:
                raise (e)
            elif debug_mode:
                raise (e)
    out_df.set_index("cut_val", inplace=True)
    if final_cut_value is not None:
        sf, sf_err, _, _ = get_survival_fraction(
            energy,
            cut_param,
            final_cut_value,
            peak,
            eres_pars,
            fit_range=fit_range,
            data_mask=data_mask,
            mode=mode,
            pars=pars,
            func=func,
        )
    else:
        sf = None
        sf_err = None
    return (
        out_df,
        sf,
        sf_err,
    )


def compton_sf(
    cut_param, low_cut_val, high_cut_val=None, mode="greater", data_mask=None
) -> dict:
    """
    Function for calculating the survival fraction of a cut in a compton region with
    a simple counting analysis.

    Parameters
    ----------

    cut_param: array
        array of the cut parameter for the survival fraction calculation, should have the same length as energy
    low_cut_val: float
        value of the cut parameter to be used for the survival fraction calculation
    high_cut_val: float
        upper value for the cut parameter to have a range in the cut value
    mode: str
        mode of the cut, either "greater" or "less"
    data_mask: array
        mask for the data to be used in the fit

    Returns
    -------

    sf : dict
        dictionary containing the low cut value, survival fraction and error on the survival fraction
    """
    if data_mask is None:
        data_mask = np.full(len(cut_param), True, dtype=bool)

    if not isinstance(cut_param, np.ndarray):
        cut_param = np.array(cut_param)

    if high_cut_val is not None:
        mask = (cut_param > low_cut_val) & (cut_param < high_cut_val) & data_mask
    else:
        if mode == "greater":
            mask = (cut_param > low_cut_val) & data_mask
        elif mode == "less":
            mask = (cut_param < low_cut_val) & data_mask
        else:
            raise ValueError("mode not recognised")

    sf = len(cut_param[mask]) / len(cut_param)
    err = 100 * np.sqrt((sf * (1 - sf)) / len(cut_param))
    sf *= 100

    return {
        "low_cut": low_cut_val,
        "sf": sf,
        "sf_err": err,
        "high_cut": high_cut_val,
    }


def compton_sf_sweep(
    energy: np.array,
    cut_param: np.array,
    final_cut_value: float,
    data_mask: np.array = None,
    cut_range=(-5, 5),
    n_samples=51,
    mode="greater",
) -> tuple(pd.DataFrame, float, float):
    """
    Function sweeping through cut values and calculating the survival fraction for each value
    using a simple counting analysis.

    Parameters
    ----------

    energy: array
        array of energies
    cut_param: array
        array of the cut parameter for the survival fraction calculation, should have the same length as energy
    final_cut_val: float
        value of the final cut parameter to be used for the survival fraction calculation
    data_mask: array
        mask for the data to be used in the fit
    cut_range: tuple
        range of the cut values to be swept through
    n_samples: int
        number of samples to be taken in the cut range
    mode: str
        mode of the cut, either "greater" or "less"

    Returns
    -------

    out_df: Dataframe
        Dataframe of cut values with the survival fraction and error
    sf: float
        survival fraction
    err: float
        error on the survival fraction

    """
    if not isinstance(energy, np.ndarray):
        energy = np.array(energy)
    if not isinstance(cut_param, np.ndarray):
        cut_param = np.array(cut_param)

    cut_vals = np.linspace(cut_range[0], cut_range[1], n_samples)
    out_df = pd.DataFrame()

    for cut_val in cut_vals:
        ct_dict = compton_sf(cut_param, cut_val, mode=mode, data_mask=data_mask)
        df = pd.DataFrame(
            [
                {
                    "cut_val": ct_dict["low_cut"],
                    "sf": ct_dict["sf"],
                    "sf_err": ct_dict["sf_err"],
                }
            ]
        )
        out_df = pd.concat([out_df, df])
    out_df.set_index("cut_val", inplace=True)

    sf_dict = compton_sf(cut_param, final_cut_value, mode=mode, data_mask=data_mask)

    return out_df, sf_dict["sf"], sf_dict["sf_err"]
