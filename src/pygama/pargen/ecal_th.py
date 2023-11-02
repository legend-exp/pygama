"""
This module provides a routine for running the energy calibration on Th data
"""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
from datetime import datetime

import matplotlib as mpl
from scipy.stats import binned_statistic

mpl.use("agg")
import lgdo.lh5_store as lh5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from iminuit import Minuit, cost
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.cuts as cts
import pygama.pargen.energy_cal as cal
from pygama.pargen.utils import load_data, return_nans

log = logging.getLogger(__name__)


def fwhm_slope(x: np.array, m0: float, m1: float, m2: float = None) -> np.array:
    """
    Fit the energy resolution curve
    """
    if m2 is None:
        return np.sqrt(m0 + m1 * x)
    else:
        return np.sqrt(m0 + m1 * x + m2 * x**2)


def apply_cuts(
    data: pd.DataFrame,
    hit_dict,
    cut_parameters=None,
    final_cut_field: str = "is_valid_cal",
    pulser_field="is_pulser",
):
    if cut_parameters is not None:
        cut_dict = cts.generate_cuts(data.query(f"(~{pulser_field})"), cut_parameters)
        hit_dict.update(
            cts.cut_dict_to_hit_dict(cut_dict, final_cut_field=final_cut_field)
        )
        mask = cts.get_cut_indexes(data, cut_dict)

        data[final_cut_field] = mask

    else:
        data[final_cut_field] = np.ones(len(data), dtype=bool)

    events_pqc = len(data.query(f"{final_cut_field}&(~{pulser_field})"))
    log.debug(f"{events_pqc} events valid for calibration")

    return data, hit_dict


def gen_pars_dict(pars, deg, energy_param):
    if deg == 1:
        out_dict = {
            "expression": f"a*{energy_param}+b",
            "parameters": {"a": pars[0], "b": pars[1]},
        }
    elif deg == 0:
        out_dict = {
            "expression": f"a*{energy_param}",
            "parameters": {"a": pars[0]},
        }
    elif deg == 2:
        out_dict = {
            "expression": f"a*{energy_param}**2 +b*{energy_param}+c",
            "parameters": {"a": pars[0], "b": pars[1], "c": pars[2]},
        }
    else:
        out_dict = {}
        log.error(f"hit_dict not implemented for deg = {deg}")

    return out_dict


class fwhm_linear:
    def func(x, a, b):
        return np.sqrt(a + b * x)

    def string_func(input_param):
        return f"(a+b*{input_param})**(0.5)"

    def guess(xs, ys, y_errs):
        return [np.nanmin(ys), 10**-3]

    def bounds():
        return [(0, None), (0, None)]


class fwhm_quadratic:
    def func(x, a, b, c):
        return np.sqrt(a + b * x + c * x**2)

    def string_func(input_param):
        return f"(a+b*{input_param}+c*{input_param}**2)**(0.5)"

    def guess(xs, ys, y_errs):
        return [np.nanmin(ys), 10**-3, 10**-5]

    def bounds():
        return [(0, None), (0, None), (0, None)]


class calibrate_parameter:
    glines = [
        # 238.632,
        583.191,
        727.330,
        860.564,
        1592.53,
        1620.50,
        2103.53,
        2614.50,
    ]  # gamma lines used for calibration
    range_keV = [
        # (8, 8),
        (20, 20),
        (30, 30),
        (30, 30),
        (40, 20),
        (20, 40),
        (40, 40),
        (60, 60),
    ]  # side bands width
    funcs = [
        # pgf.extended_gauss_step_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_gauss_step_pdf,
        pgf.extended_gauss_step_pdf,
        pgf.extended_gauss_step_pdf,
        pgf.extended_radford_pdf,
    ]
    gof_funcs = [
        # pgf.gauss_step_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.gauss_step_pdf,
        pgf.gauss_step_pdf,
        pgf.gauss_step_pdf,
        pgf.radford_pdf,
    ]

    def __init__(
        self,
        energy_param,
        selection_string="",
        plot_options: dict = None,
        guess_keV: float | None = None,
        threshold: int = 0,
        p_val: float = 0,
        n_events: int = None,
        simplex: bool = True,
        deg: int = 1,
        cal_energy_param: str = None,
        tail_weight=100,
    ):
        self.energy_param = energy_param
        if cal_energy_param is None:
            self.cal_energy_param = f"{self.energy_param}_cal"
        else:
            self.cal_energy_param = cal_energy_param
        self.selection_string = selection_string
        self.guess_keV = guess_keV
        self.threshold = threshold
        self.p_val = p_val
        self.n_events = n_events
        self.deg = deg
        self.plot_options = plot_options
        self.simplex = simplex
        self.tail_weight = tail_weight

        self.output_dict = {}
        self.hit_dict = {}

    def fit_energy_res(self):
        fitted_peaks = self.results["fitted_keV"]
        fwhms = self.results["pk_fwhms"][:, 0]
        dfwhms = self.results["pk_fwhms"][:, 1]

        #####
        # Remove the Tl SEP and DEP from calibration if found
        fwhm_peaks = np.array([], dtype=np.float32)
        all_peaks = np.array([], dtype=np.float32)
        indexes = []
        for i, peak in enumerate(fitted_peaks):
            all_peaks = np.append(all_peaks, peak)
            if peak == 2103.53:
                log.info(f"Tl SEP found at index {i}")
                indexes.append(i)
                continue
            elif peak == 1592.53:
                log.info(f"Tl DEP found at index {i}")
                indexes.append(i)
                continue
            elif peak == 511.0:
                log.info(f"e annhilation found at index {i}")
                indexes.append(i)
                continue
            elif np.isnan(dfwhms[i]):
                log.info(f"{peak} failed")
                indexes.append(i)
                continue
            else:
                fwhm_peaks = np.append(fwhm_peaks, peak)
        fit_fwhms = np.delete(fwhms, [indexes])
        fit_dfwhms = np.delete(dfwhms, [indexes])
        #####
        for i, peak in enumerate(fwhm_peaks):
            log.info(
                f"FWHM of {peak} keV peak is: {fit_fwhms[i]:1.2f} +- {fit_dfwhms[i]:1.2f} keV"
            )
        try:
            if 2614.50 not in fwhm_peaks:
                raise RuntimeError

            c_lin = cost.LeastSquares(
                fwhm_peaks, fit_fwhms, fit_dfwhms, fwhm_linear.func
            )
            c_lin.loss = "soft_l1"
            m_lin = Minuit(c_lin, *fwhm_linear.guess(fwhm_peaks, fit_fwhms, fit_dfwhms))
            m_lin.limits = fwhm_linear.bounds()
            m_lin.simplex()
            m_lin.migrad()
            m_lin.hesse()

            rng = np.random.default_rng(1)
            pars_b = rng.multivariate_normal(m_lin.values, m_lin.covariance, size=1000)
            fits = np.array([fwhm_linear.func(fwhm_peaks, *par_b) for par_b in pars_b])
            qbb_vals = np.array([fwhm_linear.func(2039.0, *par_b) for par_b in pars_b])
            qbb_err = np.nanstd(qbb_vals)
            predicted_fwhms = fwhm_linear.func(fwhm_peaks, *m_lin.values)
            fit_qbb = fwhm_linear.func(2039.0, *m_lin.values)

            p_val = scipy.stats.chi2.sf(m_lin.fval, len(fwhm_peaks) - len(m_lin.values))

            self.fwhm_fit_linear = {
                "function": fwhm_linear.__name__,
                "module": fwhm_linear.__module__,
                "expression": fwhm_linear.string_func("x"),
                "Qbb_fwhm_in_keV": fit_qbb,
                "Qbb_fwhm_err_in_keV": qbb_err,
                "parameters": m_lin.values,
                "uncertainties": m_lin.errors,
                "cov": m_lin.covariance,
                "csqr": (m_lin.fval, len(fwhm_peaks) - len(m_lin.values)),
                "p_val": p_val,
            }

            log.info(f'FWHM linear fit: {self.fwhm_fit_linear["parameters"].to_dict()}')
            log.info(f"FWHM fit values:")
            log.info(f"\t   Energy   | FWHM (keV)  | Predicted (keV)")
            for i, (peak, fwhm, fwhme) in enumerate(
                zip(fwhm_peaks, fit_fwhms, fit_dfwhms)
            ):
                log.info(
                    f"\t{i}".ljust(4)
                    + str(peak).ljust(9)
                    + f"| {fwhm:.2f}+-{fwhme:.2f}  ".ljust(5)
                    + f"| {fwhm_linear.func(peak, *self.fwhm_fit_linear['parameters']):.2f}".ljust(
                        5
                    )
                )

            log.info(
                f"FWHM energy resolution at Qbb (linear fit): {fit_qbb:1.2f} +- {qbb_err:1.2f} keV"
            )
        except RuntimeError:
            log.error(f"FWHM linear fit failed for {self.energy_param}")
            pars, errs, cov = return_nans(fwhm_linear.func)
            self.fwhm_fit_linear = {
                "function": fwhm_linear.__name__,
                "module": fwhm_linear.__module__,
                "expression": fwhm_linear.string_func("x"),
                "Qbb_fwhm_in_keV": np.nan,
                "Qbb_fwhm_err_in_keV": np.nan,
                "parameters": pars,
                "uncertainties": errs,
                "cov": cov,
                "csqr": (np.nan, np.nan),
                "p_val": 0,
            }
            log.error("FWHM linear fit failed to converge")
        try:
            if 2614.50 not in fwhm_peaks:
                raise RuntimeError
            c_quad = cost.LeastSquares(
                fwhm_peaks, fit_fwhms, fit_dfwhms, fwhm_quadratic.func
            )
            c_quad.loss = "soft_l1"
            m_quad = Minuit(
                c_quad, *fwhm_quadratic.guess(fwhm_peaks, fit_fwhms, fit_dfwhms)
            )
            m_quad.limits = fwhm_quadratic.bounds()
            m_quad.simplex()
            m_quad.migrad()
            m_quad.hesse()

            rng = np.random.default_rng(1)
            pars_b = rng.multivariate_normal(
                m_quad.values, m_quad.covariance, size=1000
            )
            fits = np.array(
                [fwhm_quadratic.func(fwhm_peaks, *par_b) for par_b in pars_b]
            )
            qbb_vals = np.array(
                [fwhm_quadratic.func(2039.0, *par_b) for par_b in pars_b]
            )
            qbb_err = np.nanstd(qbb_vals)
            predicted_fwhms = fwhm_quadratic.func(fwhm_peaks, *m_quad.values)
            fit_qbb = fwhm_quadratic.func(2039.0, *m_quad.values)

            p_val = scipy.stats.chi2.sf(
                m_quad.fval, len(fwhm_peaks) - len(m_quad.values)
            )

            self.fwhm_fit_quadratic = {
                "function": fwhm_quadratic.__name__,
                "module": fwhm_quadratic.__module__,
                "expression": fwhm_quadratic.string_func("x"),
                "Qbb_fwhm_in_keV": fit_qbb,
                "Qbb_fwhm_err_in_keV": qbb_err,
                "parameters": m_quad.values,
                "uncertainties": m_quad.errors,
                "cov": m_quad.covariance,
                "csqr": (m_quad.fval, len(fwhm_peaks) - len(m_quad.values)),
                "p_val": p_val,
            }
            log.info(
                f'FWHM quadratic fit: {self.fwhm_fit_quadratic["parameters"].to_dict()}'
            )
            log.info(
                f"FWHM energy resolution at Qbb (quadratic fit): {fit_qbb:1.2f} +- {qbb_err:1.2f} keV"
            )
        except RuntimeError:
            log.error(f"FWHM quadratic fit failed for {self.energy_param}")
            pars, errs, cov = return_nans(fwhm_quadratic.func)
            self.fwhm_fit_quadratic = {
                "function": fwhm_quadratic.__name__,
                "module": fwhm_quadratic.__module__,
                "expression": fwhm_quadratic.string_func("x"),
                "Qbb_fwhm_in_keV": np.nan,
                "Qbb_fwhm_err_in_keV": np.nan,
                "parameters": pars,
                "uncertainties": errs,
                "cov": cov,
                "csqr": (np.nan, np.nan),
                "p_val": 0,
            }
            log.error("FWHM quadratic fit failed to converge")

    def gen_pars_dict(self):
        if self.deg == 1:
            out_dict = {
                "expression": f"a*{self.energy_param}+b",
                "parameters": {"a": self.pars[0], "b": self.pars[1]},
            }
        elif self.deg == 0:
            out_dict = {
                "expression": f"a*{self.energy_param}",
                "parameters": {"a": self.pars[0]},
            }
        elif self.deg == 2:
            out_dict = {
                "expression": f"a*{self.energy_param}**2 +b*{self.energy_param}+c",
                "parameters": {"a": self.pars[0], "b": self.pars[1], "c": self.pars[2]},
            }
        else:
            out_dict = {}
            log.warning(f"hit_dict not implemented for deg = {self.deg}")

        return out_dict

    def get_results_dict(self, data):
        if np.isnan(self.pars).all():
            return {}
        else:
            fwhm_linear = self.fwhm_fit_linear.copy()
            fwhm_linear["parameters"] = fwhm_linear["parameters"].to_dict()
            fwhm_linear["uncertainties"] = fwhm_linear["uncertainties"].to_dict()
            fwhm_linear["cov"] = fwhm_linear["cov"].tolist()
            fwhm_quad = self.fwhm_fit_quadratic.copy()
            fwhm_quad["parameters"] = fwhm_quad["parameters"].to_dict()
            fwhm_quad["uncertainties"] = fwhm_quad["uncertainties"].to_dict()
            fwhm_quad["cov"] = fwhm_quad["cov"].tolist()

            pk_dict = {
                Ei: {
                    "function": func_i.__name__,
                    "module": func_i.__module__,
                    "parameters_in_ADC": parsi.to_dict(),
                    "uncertainties_in_ADC": errorsi.to_dict(),
                    "p_val": pvali,
                    "fwhm_in_keV": list(fwhmi),
                }
                for i, (Ei, parsi, errorsi, pvali, fwhmi, func_i) in enumerate(
                    zip(
                        self.results["fitted_keV"],
                        self.results["pk_pars"][self.results["pk_validities"]],
                        self.results["pk_errors"][self.results["pk_validities"]],
                        self.results["pk_pvals"][self.results["pk_validities"]],
                        self.results["pk_fwhms"],
                        self.funcs,
                    )
                )
            }

            return {
                "total_fep": len(
                    data.query(
                        f"{self.cal_energy_param}>2604&{self.cal_energy_param}<2624"
                    )
                ),
                "total_dep": len(
                    data.query(
                        f"{self.cal_energy_param}>1587&{self.cal_energy_param}<1597"
                    )
                ),
                "pass_fep": len(
                    data.query(
                        f"{self.cal_energy_param}>2604&{self.cal_energy_param}<2624&{self.selection_string}"
                    )
                ),
                "pass_dep": len(
                    data.query(
                        f"{self.cal_energy_param}>1587&{self.cal_energy_param}<1597&{self.selection_string}"
                    )
                ),
                "eres_linear": fwhm_linear,
                "eres_quadratic": fwhm_quad,
                "fitted_peaks": self.results["fitted_keV"].tolist(),
                "pk_fits": pk_dict,
            }

    def calibrate_parameter(self, data):
        kev_ranges = self.range_keV.copy()
        if self.guess_keV is None:
            self.guess_keV = 2620 / np.nanpercentile(
                data.query(
                    f"{self.selection_string} & {self.energy_param}>{self.threshold}"
                )[self.energy_param],
                99,
            )

        log.debug(f"Find peaks and compute calibration curve for {self.energy_param}")
        log.debug(f"Guess is {self.guess_keV:.3f}")

        try:
            self.pars, self.cov, self.results = cal.hpge_E_calibration(
                data.query(self.selection_string)[self.energy_param],
                self.glines,
                self.guess_keV,
                deg=self.deg,
                range_keV=kev_ranges,
                funcs=self.funcs,
                gof_funcs=self.gof_funcs,
                n_events=self.n_events,
                allowed_p_val=self.p_val,
                simplex=self.simplex,
                tail_weight=self.tail_weight,
                verbose=False,
            )
            pk_pars = self.results["pk_pars"]
            found_peaks = self.results["got_peaks_locs"]
            fitted_peaks = self.results["fitted_keV"]
            fitted_funcs = self.results["pk_funcs"]
            if self.pars is None:
                raise ValueError

            for i, peak in enumerate(self.results["got_peaks_keV"]):
                idx = np.where(peak == self.glines)[0][0]
                self.funcs[idx] = fitted_funcs[i]
                if fitted_funcs[i] == pgf.extended_radford_pdf:
                    self.gof_funcs[idx] = pgf.radford_pdf
                else:
                    self.gof_funcs[idx] = pgf.gauss_step_pdf
        except:
            found_peaks = np.array([])
            fitted_peaks = np.array([])
            fitted_funcs = np.array([])
        if len(fitted_peaks) != len(self.glines):
            for i, peak in enumerate(self.glines):
                if peak not in fitted_peaks:
                    kev_ranges[i] = (kev_ranges[i][0] - 5, kev_ranges[i][1] - 5)
            for i, peak in enumerate(fitted_peaks):
                try:
                    if (
                        self.results["pk_fwhms"][:, 1][i]
                        / self.results["pk_fwhms"][:, 0][i]
                        > 0.05
                    ):
                        index = np.where(self.glines == peak)[0][0]
                        kev_ranges[index] = (
                            kev_ranges[index][0] - 5,
                            kev_ranges[index][1] - 5,
                        )
                except:
                    pass
            try:
                self.pars, self.cov, self.results = cal.hpge_E_calibration(
                    data.query(self.selection_string)[self.energy_param],
                    self.glines,
                    self.guess_keV,
                    deg=self.deg,
                    range_keV=kev_ranges,
                    funcs=self.funcs,
                    gof_funcs=self.gof_funcs,
                    n_events=self.n_events,
                    allowed_p_val=self.p_val,
                    simplex=self.simplex,
                    tail_weight=self.tail_weight,
                    verbose=False,
                )
                fitted_peaks = self.results["fitted_keV"]
                fitted_funcs = self.results["pk_funcs"]

                log.debug("Calibrated found")
                log.info(f"Calibration pars are {self.pars}")

                for i, peak in enumerate(self.results["got_peaks_keV"]):
                    idx = np.where(peak == self.glines)[0][0]
                    self.funcs[idx] = fitted_funcs[i]
                    if fitted_funcs[i] == pgf.extended_radford_pdf:
                        self.gof_funcs[idx] = pgf.radford_pdf
                    else:
                        self.gof_funcs[idx] = pgf.gauss_step_pdf
                if self.pars is None:
                    raise ValueError

            except:
                self.pars = np.full(self.deg + 1, np.nan)
                self.results = None

                log.error(f"Calibration failed completely for {self.energy_param}")
        else:
            log.debug("Calibrated found")
            log.info(f"Calibration pars are {self.pars}")
        if ~np.isnan(self.pars).all():
            self.fit_energy_res()
        self.hit_dict[self.cal_energy_param] = self.gen_pars_dict()
        data[f"{self.energy_param}_cal"] = pgf.poly(data[self.energy_param], self.pars)

    def fill_plot_dict(self, data, plot_dict={}):
        for key, item in self.plot_options.items():
            if item["options"] is not None:
                plot_dict[key] = item["function"](self, data, **item["options"])
            else:
                plot_dict[key] = item["function"](self, data)
        return plot_dict


class high_stats_fitting(calibrate_parameter):
    glines = [
        238.632,
        511,
        583.191,
        727.330,
        763,
        785,
        860.564,
        893,
        1079,
        1513,
        1592.53,
        1620.50,
        2103.53,
        2614.50,
        3125,
        3198,
        3474,
    ]  # gamma lines used for calibration
    range_keV = [
        (10, 10),
        (30, 30),
        (30, 30),
        (30, 30),
        (30, 15),
        (15, 30),
        (30, 25),
        (25, 30),
        (30, 30),
        (30, 30),
        (30, 20),
        (20, 30),
        (30, 30),
        (30, 30),
        (30, 30),
        (30, 30),
        (30, 30),
    ]  # side bands width
    binning = [
        0.02,
        0.02,
        0.02,
        0.02,
        0.2,
        0.2,
        0.02,
        0.2,
        0.2,
        0.2,
        0.1,
        0.1,
        0.1,
        0.02,
        0.2,
        0.2,
        0.2,
    ]
    funcs = [
        pgf.extended_gauss_step_pdf,  # probably should be gauss on exp
        pgf.extended_gauss_step_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_gauss_step_pdf,
        pgf.extended_gauss_step_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_gauss_step_pdf,
        pgf.extended_gauss_step_pdf,
        pgf.extended_gauss_step_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_gauss_step_pdf,
        pgf.extended_gauss_step_pdf,
        pgf.extended_gauss_step_pdf,
    ]
    gof_funcs = [
        pgf.gauss_step_pdf,
        pgf.gauss_step_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.gauss_step_pdf,
        pgf.gauss_step_pdf,
        pgf.gauss_step_pdf,
        pgf.radford_pdf,
        pgf.gauss_step_pdf,
        pgf.gauss_step_pdf,
        pgf.gauss_step_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.gauss_step_pdf,
        pgf.gauss_step_pdf,
        pgf.gauss_step_pdf,
    ]

    def __init__(
        self,
        energy_param,
        selection_string,
        threshold,
        p_val,
        plot_options={},
        simplex=False,
        tail_weight=20,
    ):
        self.energy_param = energy_param
        self.cal_energy_param = energy_param
        self.selection_string = selection_string
        self.threshold = threshold
        self.p_val = p_val
        self.plot_options = plot_options
        self.simplex = simplex
        self.results = {}
        self.plot_dict = {}
        self.n_events = None
        self.output_dict = {}
        self.pars = [1, 0]
        self.tail_weight = tail_weight

    def get_results_dict(self, data):
        if self.results:
            fwhm_linear = self.fwhm_fit_linear.copy()
            fwhm_linear["parameters"] = fwhm_linear["parameters"].to_dict()
            fwhm_linear["uncertainties"] = fwhm_linear["uncertainties"].to_dict()
            fwhm_linear["cov"] = fwhm_linear["cov"].tolist()
            fwhm_quad = self.fwhm_fit_quadratic.copy()
            fwhm_quad["parameters"] = fwhm_quad["parameters"].to_dict()
            fwhm_quad["uncertainties"] = fwhm_quad["uncertainties"].to_dict()
            fwhm_quad["cov"] = fwhm_quad["cov"].tolist()

            pk_dict = {
                Ei: {
                    "function": func_i.__name__,
                    "module": func_i.__module__,
                    "parameters_in_keV": parsi.to_dict(),
                    "uncertainties_in_keV": errorsi.to_dict(),
                    "p_val": pvali,
                    "fwhm_in_keV": list(fwhmi),
                }
                for i, (Ei, parsi, errorsi, pvali, fwhmi, func_i) in enumerate(
                    zip(
                        self.results["fitted_keV"],
                        self.results["pk_pars"][self.results["pk_validities"]],
                        self.results["pk_errors"][self.results["pk_validities"]],
                        self.results["pk_pvals"][self.results["pk_validities"]],
                        self.results["pk_fwhms"],
                        self.funcs,
                    )
                )
            }

            return {
                "eres_linear": fwhm_linear,
                "eres_quadratic": fwhm_quad,
                "fitted_peaks": self.results["fitted_keV"].tolist(),
                "pk_fits": pk_dict,
            }
        else:
            return {}

    def fit_peaks(self, data):
        log.debug(f"Fitting {self.energy_param}")
        try:
            n_bins = [
                int((self.range_keV[i][1] + self.range_keV[i][0]) / self.binning[i])
                for i in range(len(self.glines))
            ]
            (
                pk_pars,
                pk_errors,
                pk_covs,
                pk_binws,
                pk_ranges,
                pk_pvals,
                valid_pks,
                pk_funcs,
            ) = cal.hpge_fit_E_peaks(
                data.query(self.selection_string)[self.energy_param],
                self.glines,
                self.range_keV,
                n_bins=n_bins,
                funcs=self.funcs,
                method="unbinned",
                gof_funcs=self.gof_funcs,
                n_events=None,
                allowed_p_val=self.p_val,
                tail_weight=20,
            )
            for idx, peak in enumerate(self.glines):
                self.funcs[idx] = pk_funcs[idx]
                if pk_funcs[idx] == pgf.extended_radford_pdf:
                    self.gof_funcs[idx] = pgf.radford_pdf
                else:
                    self.gof_funcs[idx] = pgf.gauss_step_pdf

            self.results["got_peaks_keV"] = self.glines
            self.results["pk_pars"] = pk_pars
            self.results["pk_errors"] = pk_errors
            self.results["pk_covs"] = pk_covs
            self.results["pk_binws"] = pk_binws
            self.results["pk_ranges"] = pk_ranges
            self.results["pk_pvals"] = pk_pvals

            for i, pk in enumerate(self.results["got_peaks_keV"]):
                try:
                    if self.results["pk_pars"][i]["n_sig"] < 10:
                        valid_pks[i] = False
                    elif (
                        2 * self.results["pk_errors"][i]["n_sig"]
                        > self.results["pk_pars"][i]["n_sig"]
                    ):
                        valid_pks[i] = False
                except:
                    pass

            self.results["pk_validities"] = valid_pks

            # Drop failed fits
            fitted_peaks_keV = self.results["fitted_keV"] = np.asarray(self.glines)[
                valid_pks
            ]
            pk_pars = np.asarray(pk_pars, dtype=object)[valid_pks]  # ragged
            pk_errors = np.asarray(pk_errors, dtype=object)[valid_pks]
            pk_covs = np.asarray(pk_covs, dtype=object)[valid_pks]
            pk_binws = np.asarray(pk_binws)[valid_pks]
            pk_ranges = np.asarray(pk_ranges)[valid_pks]
            pk_pvals = np.asarray(pk_pvals)[valid_pks]
            pk_funcs = np.asarray(pk_funcs)[valid_pks]

            log.info(f"{len(np.where(valid_pks)[0])} peaks fitted:")
            for i, (Ei, parsi, errorsi, covsi, func_i) in enumerate(
                zip(fitted_peaks_keV, pk_pars, pk_errors, pk_covs, pk_funcs)
            ):
                varnames = func_i.__code__.co_varnames[1 : len(pk_pars[-1]) + 1]
                parsi = np.asarray(parsi, dtype=float)
                errorsi = np.asarray(errorsi, dtype=float)
                covsi = np.asarray(covsi, dtype=float)
                # parsigsi = np.sqrt(covsi.diagonal())
                log.info(f"\tEnergy: {str(Ei)}")
                log.info(f"\t\tParameter  |    Value +/- Sigma  ")
                for vari, pari, errorsi in zip(varnames, parsi, errorsi):
                    log.info(
                        f'\t\t{str(vari).ljust(10)} | {("%4.2f" % pari).rjust(8)} +/- {("%4.2f" % errorsi).ljust(8)}'
                    )

            cal_fwhms = [
                pgf.get_fwhm_func(func_i, pars_i, cov=covs_i)
                for func_i, pars_i, covs_i in zip(pk_funcs, pk_pars, pk_covs)
            ]

            cal_fwhms, cal_fwhms_errs = zip(*cal_fwhms)
            cal_fwhms = np.asarray(cal_fwhms)
            cal_fwhms_errs = np.asarray(cal_fwhms_errs)
            self.results["pk_fwhms"] = np.asarray(
                [(u, e) for u, e in zip(cal_fwhms, cal_fwhms_errs)]
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
            self.fit_energy_res()
            log.debug(f"high stats fitting successful")
        except:
            self.results = {}
            log.debug(f"high stats fitting failed")


def get_peak_labels(
    labels: list[str], pars: list[float]
) -> tuple(list[float], list[float]):
    out = []
    out_labels = []
    for i, label in enumerate(labels):
        if i % 2 == 1:
            continue
        else:
            out.append(f"{pgf.poly(label, pars):.1f}")
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
    ecal_class, data, figsize=[12, 8], fontsize=12, ncols=3, nrows=3, binning_keV=5
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
    derco = np.polyder(np.poly1d(ecal_class.pars)).coefficients
    der = [pgf.poly(5, derco) for Ei in fitted_peaks]
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
        except:
            pass

    plt.tight_layout()
    plt.close()
    return fig


def plot_2614_timemap(
    ecal_class,
    data,
    figsize=[12, 8],
    fontsize=12,
    erange=[2580, 2630],
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
    figsize=[12, 8],
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


def bin_stability(ecal_class, data, time_slice=180, energy_range=[2585, 2660]):
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


def plot_cal_fit(ecal_class, data, figsize=[12, 8], fontsize=12, erange=[200, 2700]):
    pk_pars = ecal_class.results["pk_pars"]
    fitted_peaks = ecal_class.results["got_peaks_keV"]
    pk_errs = ecal_class.results["pk_errors"]

    fitted_gof_funcs = []
    for i, peak in enumerate(ecal_class.glines):
        if peak in fitted_peaks:
            fitted_gof_funcs.append(ecal_class.gof_funcs[i])

    mus = [
        pgf.get_mu_func(func_i, pars_i) if pars_i is not None else np.nan
        for func_i, pars_i in zip(fitted_gof_funcs, pk_pars)
    ]

    mu_errs = [
        pgf.get_mu_func(func_i, pars_i) if pars_i is not None else np.nan
        for func_i, pars_i in zip(fitted_gof_funcs, pk_errs)
    ]

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    cal_bins = np.linspace(0, np.nanmax(mus) * 1.1, 20)

    ax1.scatter(fitted_peaks, mus, marker="x", c="b")

    ax1.plot(pgf.poly(cal_bins, ecal_class.pars), cal_bins, lw=1, c="g")

    ax1.grid()
    ax1.set_xlim([erange[0], erange[1]])
    ax1.set_ylabel("Energy (ADC)")
    ax2.errorbar(
        fitted_peaks,
        pgf.poly(np.array(mus), ecal_class.pars) - fitted_peaks,
        yerr=pgf.poly(np.array(mus) + np.array(mu_errs), ecal_class.pars)
        - pgf.poly(np.array(mus), ecal_class.pars),
        linestyle=" ",
        marker="x",
        c="b",
    )
    ax2.grid()
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Residuals (keV)")
    plt.close()
    return fig


def plot_eres_fit(ecal_class, data, erange=[200, 2700], figsize=[12, 8], fontsize=12):
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
            log.info(f"e annhilation found at index {i}")
            indexes.append(i)
            continue
        else:
            fwhm_peaks = np.append(fwhm_peaks, peak)
    fit_fwhms = np.delete(fwhms, [indexes])
    fit_dfwhms = np.delete(dfwhms, [indexes])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.errorbar(fwhm_peaks, fit_fwhms, yerr=fit_dfwhms, marker="x", lw=0, c="black")

    fwhm_slope_bins = np.arange(erange[0], erange[1], 10)

    qbb_line_vx = [2039.0, 2039.0]
    qbb_line_vy = [
        0.9
        * np.nanmin(
            fwhm_linear.func(fwhm_slope_bins, *ecal_class.fwhm_fit_linear["parameters"])
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
        fwhm_linear.func(fwhm_slope_bins, *ecal_class.fwhm_fit_linear["parameters"]),
        lw=1,
        c="g",
        label=f'linear, Qbb fwhm: {ecal_class.fwhm_fit_linear["Qbb_fwhm_in_keV"]:1.2f} +- {ecal_class.fwhm_fit_linear["Qbb_fwhm_err_in_keV"]:1.2f} keV',
    )
    ax1.plot(
        fwhm_slope_bins,
        fwhm_quadratic.func(
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
        [
            0.9 * np.nanmin(fit_fwhms),
            1.1 * np.nanmax(fit_fwhms),
        ]
    else:
        ax1.set_ylim(
            [
                0.9
                * np.nanmin(
                    fwhm_linear.func(
                        fwhm_slope_bins, *ecal_class.fwhm_fit_linear["parameters"]
                    )
                ),
                1.1
                * np.nanmax(
                    fwhm_linear.func(
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
            - fwhm_linear.func(fwhm_peaks, *ecal_class.fwhm_fit_linear["parameters"])
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
            - fwhm_quadratic.func(
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
    erange=[0, 3000],
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
    erange=[0, 3000],
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
