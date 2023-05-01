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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

import pygama.lgdo.lh5_store as lh5
import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.cuts as cts
import pygama.pargen.energy_cal as cal

log = logging.getLogger(__name__)


def fwhm_slope(x: np.array, m0: float, m1: float, m2: float = None) -> np.array:
    """
    Fit the energy resolution curve
    """
    if m2 is None:
        return np.sqrt(m0 + m1 * x)
    else:
        return np.sqrt(m0 + m1 * x + m2 * x**2)


def load_data(
    files: list[str],
    lh5_path: str,
    energy_params: list[str],
    hit_dict: dict = {},
    cut_parameters: list[str] = ["bl_mean", "bl_std", "pz_std"],
) -> pd.DataFrame:
    df = lh5.load_dfs(files, ["timestamp", "trapTmax"], lh5_path)
    pulser_props = cts.find_pulser_properties(df, energy="trapTmax")
    if len(pulser_props) > 0:
        final_mask = None
        for entry in pulser_props:
            e_cut = (df.trapTmax.values < entry[0] + entry[1]) & (
                df.trapTmax.values > entry[0] - entry[1]
            )
            if final_mask is None:
                final_mask = e_cut
            else:
                final_mask = final_mask | e_cut
        ids = ~(final_mask)
        log.debug(f"pulser found: {pulser_props}")

    else:
        ids = np.ones(len(df), dtype=bool)
        log.debug(f"no pulser found")

    sto = lh5.LH5Store()
    table = sto.read_object(lh5_path, files)[0]

    if len(hit_dict.keys()) == 0:
        out_df = df.copy()
        for param in energy_params:
            try:
                out_df[param] = table[param].nda

            except RuntimeError:
                param = param.split("_")[0]
                out_df[param] = table[param].nda

    else:
        out_df = table.eval(hit_dict).get_dataframe()
        out_df = pd.concat([df, out_df], axis=1)
        out_df["is_not_pulser"] = ids

        cut_parameters = cts.get_keys(table, cut_parameters)

        for param in energy_params:
            if param not in out_df:
                out_df[param] = table[param].nda
        if cut_parameters is not None:
            for param in cut_parameters:
                if param not in df:
                    out_df[param] = table[param].nda
    log.debug("Data Loaded")
    return out_df


def apply_cuts(
    data: pd.DataFrame,
    hit_dict,
    cut_parameters=None,
    final_cut_field: str = "is_valid_cal",
):
    if cut_parameters is not None:
        cut_dict = cts.generate_cuts(data.query("is_not_pulser"), cut_parameters)
        hit_dict.update(
            cts.cut_dict_to_hit_dict(cut_dict, final_cut_field=final_cut_field)
        )
        mask = cts.get_cut_indexes(data, cut_dict)

        data["is_valid_cal"] = mask

    else:
        data["is_valid_cal"] = np.ones(len(data), dtype=bool)
    data["is_usable"] = data["is_valid_cal"] & data["is_not_pulser"]

    events_pqc = len(data.query("is_usable"))
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
        (40, 25),
        (25, 40),
        (40, 40),
        (60, 60),
    ]  # side bands width
    funcs = [
        # pgf.extended_gauss_step_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
    ]
    gof_funcs = [
        # pgf.gauss_step_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
    ]

    def __init__(
        self,
        data,
        energy_param,
        plot_options: dict = None,
        guess_keV: float | None = None,
        threshold: int = 0,
        p_val: float = 0,
        n_events: int = 15000,
        deg: int = 1,
    ):
        self.data = data
        self.energy_param = energy_param
        self.guess_keV = guess_keV
        self.threshold = threshold
        self.p_val = p_val
        self.n_events = n_events
        self.deg = deg
        self.plot_options = plot_options

        self.output_dict = {}
        self.hit_dict = {}
        self.plot_dict = {}

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
            elif np.isnan(dfwhms[i]):
                log.info(f"{peak} failed")
                indexes.append(i)
                continue
            else:
                fwhm_peaks = np.append(fwhm_peaks, peak)
        fit_fwhms = np.delete(fwhms, [indexes])
        fit_dfwhms = np.delete(dfwhms, [indexes])
        #####
        param_guess = [2, 0.001]
        param_bounds = (0, np.inf)
        for i, peak in enumerate(fwhm_peaks):
            log.info(
                f"FWHM of {peak} keV peak is: {fit_fwhms[i]:1.2f} +- {fit_dfwhms[i]:1.2f} keV"
            )
        try:
            self.fit_pars, self.fit_covs = curve_fit(
                fwhm_slope,
                fwhm_peaks,
                fit_fwhms,
                sigma=fit_dfwhms,
                p0=param_guess,
                bounds=param_bounds,
                absolute_sigma=True,
            )
            rng = np.random.default_rng(1)
            pars_b = rng.multivariate_normal(self.fit_pars, self.fit_covs, size=1000)
            fits = np.array([fwhm_slope(fwhm_peaks, *par_b) for par_b in pars_b])
            qbb_vals = np.array([fwhm_slope(2039.0, *par_b) for par_b in pars_b])
            self.qbb_err = np.nanstd(qbb_vals)
            predicted_fwhms = fwhm_slope(fwhm_peaks, *self.fit_pars)
            self.fit_qbb = fwhm_slope(2039.0, *self.fit_pars)

            if 2614.50 not in fwhm_peaks:
                self.fit_qbb = np.nan
                self.qbb_err = np.nan
            log.info(f"FWHM curve fit: {self.fit_pars}")
            log.info(f"FWHM fit values:")
            for peak in fwhm_peaks:
                log.info(
                    f"Predicted FWHM of {peak} keV peak is: {fwhm_slope(peak, *self.fit_pars):.2f} keV"
                )
            log.info(
                f"FWHM energy resolution at Qbb: {self.fit_qbb:1.2f} +- {self.qbb_err:1.2f} keV"
            )
        except RuntimeError:
            log.error(f"FWHM fit failed for {energy_param}")
            self.fit_pars = np.array([np.nan, np.nan])
            self.fit_covs = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            self.fit_qbb = np.nan
            self.qbb_err = np.nan
            log.error("FWHM fit failed to converge")

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

    def calibrate_parameter(self):
        kev_ranges = self.range_keV.copy()
        if self.guess_keV is None:
            self.guess_keV = 2620 / np.nanpercentile(
                self.data.query(f"is_usable & {self.energy_param}>{self.threshold}")[
                    self.energy_param
                ],
                99,
            )

        log.debug(f"Find peaks and compute calibration curve for {self.energy_param}")
        log.debug(f"Guess is {self.guess_keV:.3f}")

        try:
            self.pars, self.cov, self.results = cal.hpge_E_calibration(
                self.data.query("is_usable")[self.energy_param],
                self.glines,
                self.guess_keV,
                deg=self.deg,
                range_keV=kev_ranges,
                funcs=self.funcs,
                gof_funcs=self.gof_funcs,
                n_events=self.n_events,
                allowed_p_val=self.p_val,
                simplex=True,
                verbose=False,
            )
            pk_pars = self.results["pk_pars"]
            found_peaks = self.results["got_peaks_locs"]
            fitted_peaks = self.results["fitted_keV"]
        except:
            found_peaks = np.array([])
            fitted_peaks = np.array([])

        for i, peak in enumerate(self.glines):
            if peak not in fitted_peaks:
                kev_ranges[i] = (kev_ranges[i][0] - 5, kev_ranges[i][1] - 5)
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
                    kev_ranges[i] = (kev_ranges[index][0] - 5, kev_ranges[index][1] - 5)
            except:
                pass

        try:
            self.pars, self.cov, self.results = cal.hpge_E_calibration(
                self.data.query("is_usable")[self.energy_param],
                self.glines,
                self.guess_keV,
                deg=self.deg,
                range_keV=kev_ranges,
                funcs=self.funcs,
                gof_funcs=self.gof_funcs,
                n_events=self.n_events,
                allowed_p_val=self.p_val,
                simplex=True,
                verbose=False,
            )
        except:
            self.pars = None
        if self.pars is None:
            log.error(
                f"Calibration failed for {self.energy_param}, trying with 0 p_val"
            )
            try:
                self.pars, self.cov, self.results = cal.hpge_E_calibration(
                    self.data.query("is_usable")[self.energy_param],
                    self.glines,
                    self.guess_keV,
                    deg=self.deg,
                    range_keV=kev_ranges,
                    funcs=self.funcs,
                    gof_funcs=self.gof_funcs,
                    n_events=self.n_events,
                    allowed_p_val=0,
                    simplex=True,
                    verbose=False,
                )
                if self.pars is None:
                    raise ValueError

                self.fit_energy_res()
                self.data[f"{self.energy_param}_cal"] = pgf.poly(
                    self.data[self.energy_param], self.pars
                )
                self.hit_dict[f"{self.energy_param}_cal"] = self.gen_pars_dict()
                self.output_dict[f"{self.energy_param}_cal"] = {
                    "Qbb_fwhm": np.nan,
                    "Qbb_fwhm_err": np.nan,
                    "2.6_fwhm": np.nan,
                    "2.6_fwhm_err": np.nan,
                    "eres_pars": self.fit_pars.tolist(),
                    "fitted_peaks": np.nan,
                    "fwhms": np.nan,
                    "peak_fit_pars": np.nan,
                    "total_fep": len(
                        self.data.query(
                            f"{self.energy_param}_cal>2604&{self.energy_param}_cal<2624"
                        )
                    ),
                    "total_dep": len(
                        self.data.query(
                            f"{self.energy_param}_cal>1587&{self.energy_param}_cal<1597"
                        )
                    ),
                    "pass_fep": len(
                        self.data.query(
                            f"{self.energy_param}_cal>2604&{self.energy_param}_cal<2624&is_usable"
                        )
                    ),
                    "pass_dep": len(
                        self.data.query(
                            f"{self.energy_param}_cal>1587&{self.energy_param}_cal<1597&is_usable"
                        )
                    ),
                }
            except:
                log.error(
                    f"Calibration failed completely for {self.energy_param} even with 0 p_val"
                )
                self.pars = np.full(self.deg + 1, np.nan)

                self.hit_dict[f"{self.energy_param}_cal"] = self.gen_pars_dict()

                self.output_dict[f"{self.energy_param}_cal"] = {
                    "Qbb_fwhm": np.nan,
                    "Qbb_fwhm_err": np.nan,
                    "2.6_fwhm": np.nan,
                    "2.6_fwhm_err": np.nan,
                    "eres_pars": [np.nan, np.nan],
                    "fitted_peaks": np.nan,
                    "fwhms": np.nan,
                    "peak_fit_pars": np.nan,
                    "total_fep": np.nan,
                    "total_dep": np.nan,
                    "pass_fep": np.nan,
                    "pass_dep": np.nan,
                }

        else:
            log.debug("done")
            log.info(f"Calibration pars are {self.pars}")

            self.data[f"{self.energy_param}_cal"] = pgf.poly(
                self.data[self.energy_param], self.pars
            )

            pk_rs_dict = {
                peak: self.results["pk_pars"][i].tolist()
                for i, peak in enumerate(self.results["fitted_keV"])
            }

            self.fit_energy_res()
            self.hit_dict[f"{self.energy_param}_cal"] = self.gen_pars_dict()

            if self.results["fitted_keV"][-1] == 2614.50:
                fep_fwhm = round(self.results["pk_fwhms"][-1, 0], 2)
                fep_dwhm = round(self.results["pk_fwhms"][-1, 1], 2)
            else:
                fep_fwhm = np.nan
                fep_dwhm = np.nan

            self.output_dict[f"{self.energy_param}_cal"] = {
                "Qbb_fwhm": round(self.fit_qbb, 2),
                "Qbb_fwhm_err": round(self.qbb_err, 2),
                "2.6_fwhm": fep_fwhm,
                "2.6_fwhm_err": fep_dwhm,
                "eres_pars": self.fit_pars.tolist(),
                "fitted_peaks": self.results["fitted_keV"].tolist(),
                "fwhms": self.results["pk_fwhms"].tolist(),
                "peak_fit_pars": pk_rs_dict,
                "total_fep": len(
                    self.data.query(
                        f"{self.energy_param}_cal>2604&{self.energy_param}_cal<2624"
                    )
                ),
                "total_dep": len(
                    self.data.query(
                        f"{self.energy_param}_cal>1587&{self.energy_param}_cal<1597"
                    )
                ),
                "pass_fep": len(
                    self.data.query(
                        f"{self.energy_param}_cal>2604&{self.energy_param}_cal<2624&is_usable"
                    )
                ),
                "pass_dep": len(
                    self.data.query(
                        f"{self.energy_param}_cal>1587&{self.energy_param}_cal<1597&is_usable"
                    )
                ),
            }
        log.info(
            f"Results {self.energy_param}: {json.dumps(self.output_dict[f'{self.energy_param}_cal'], indent=2)}"
        )

    def fill_plot_dict(self):
        for key, item in self.plot_options.items():
            if item["options"] is not None:
                self.plot_dict[key] = item["function"](self, **item["options"])
            else:
                self.plot_dict[key] = item["function"](self)


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


def plot_fits(ecal_class, figsize=[12, 8], fontsize=12, ncols=3, n_rows=3):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fitted_peaks = ecal_class.results["fitted_keV"]
    pk_pars = ecal_class.results["pk_pars"]
    pk_ranges = ecal_class.results["pk_ranges"]
    p_vals = ecal_class.results["pk_pvals"]

    fitted_gof_funcs = []
    for i, peak in enumerate(ecal_class.glines):
        if peak in fitted_peaks:
            fitted_gof_funcs.append(ecal_class.gof_funcs[i])

    mus = [
        pgf.get_mu_func(func_i, pars_i)
        for func_i, pars_i in zip(fitted_gof_funcs, pk_pars)
    ]
    fwhms = ecal_class.results["pk_fwhms"][:, 0]
    dfwhms = ecal_class.results["pk_fwhms"][:, 1]

    fig = plt.figure()
    range_adu = 5 / ecal_class.pars[0]  # 10keV window around peak in adu
    for i, peak in enumerate(mus):
        # plt.subplot(math.ceil((len(mus)) / 2), 2, i + 1)
        plt.subplot(n_rows, ncols, i + 1)
        binning = np.arange(pk_ranges[i][0], pk_ranges[i][1], 1)
        bin_cs = (binning[1:] + binning[:-1]) / 2
        energies = ecal_class.data.query(
            f"{ecal_class.energy_param}>{pk_ranges[i][0]}&{ecal_class.energy_param}<{pk_ranges[i][1]}&is_usable"
        )[ecal_class.energy_param]
        energies = energies.iloc[: ecal_class.n_events]

        counts, bs, bars = plt.hist(energies, bins=binning, histtype="step")
        fit_vals = fitted_gof_funcs[i](bin_cs, *pk_pars[i]) * np.diff(bs)
        plt.plot(bin_cs, fit_vals)
        plt.step(
            bin_cs,
            [
                (fval - count) / count if count != 0 else (fval - count)
                for count, fval in zip(counts, fit_vals)
            ],
        )
        plt.plot(
            [bin_cs[10]],
            [0],
            label=get_peak_label(fitted_peaks[i]),
            linestyle="None",
        )
        plt.plot(
            [bin_cs[10]],
            [0],
            label=f"{fitted_peaks[i]:.1f} keV",
            linestyle="None",
        )
        plt.plot(
            [bin_cs[10]],
            [0],
            label=f"{fwhms[i]:.2f} +- {dfwhms[i]:.2f} keV",
            linestyle="None",
        )
        plt.plot(
            [bin_cs[10]],
            [0],
            label=f"p-value : {p_vals[i]:.2f}",
            linestyle="None",
        )

        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        plt.legend(loc="upper left", frameon=False)
        plt.xlim([peak - range_adu, peak + range_adu])
        locs, labels = plt.xticks()
        new_locs, new_labels = get_peak_labels(locs, ecal_class.pars)
        plt.xticks(ticks=new_locs, labels=new_labels)

    plt.tight_layout()
    plt.close()
    return fig


def plot_2614_timemap(
    ecal_class, figsize=[12, 8], fontsize=12, erange=[2580, 2630], dx=1, time_dx=180
):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    selection = ecal_class.data.query(
        f"{ecal_class.energy_param}_cal>2560&{ecal_class.energy_param}_cal<2660&is_usable"
    )

    if len(selection) == 0:
        pass
    else:
        time_bins = np.arange(
            (np.amin(ecal_class.data["timestamp"]) // time_dx) * time_dx,
            ((np.amax(ecal_class.data["timestamp"]) // time_dx) + 2) * time_dx,
            time_dx,
        )

        fig = plt.figure()
        plt.hist2d(
            selection["timestamp"],
            selection[f"{ecal_class.energy_param}_cal"],
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
    ecal_class, figsize=[12, 8], fontsize=12, dx=0.2, time_dx=180, n_spread=3
):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    time_bins = np.arange(
        (np.amin(ecal_class.data["timestamp"]) // time_dx) * time_dx,
        ((np.amax(ecal_class.data["timestamp"]) // time_dx) + 2) * time_dx,
        time_dx,
    )

    selection = ecal_class.data.query(f"~is_not_pulser")
    fig = plt.figure()
    if len(selection) == 0:
        pass

    else:
        mean = np.nanpercentile(selection[f"{ecal_class.energy_param}_cal"], 50)
        spread = mean - np.nanpercentile(
            selection[f"{ecal_class.energy_param}_cal"], 10
        )

        plt.hist2d(
            selection["timestamp"],
            selection[f"{ecal_class.energy_param}_cal"],
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


def bin_pulser_stability(ecal_class, time_slice=180):
    selection = ecal_class.data.query(f"~is_not_pulser")

    utime_array = ecal_class.data["timestamp"]
    select_energies = selection[f"{ecal_class.energy_param}_cal"].to_numpy()

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


def bin_stability(ecal_class, time_slice=180, energy_range=[2585, 2660]):
    selection = ecal_class.data.query(
        f"{ecal_class.energy_param}_cal>{energy_range[0]}&{ecal_class.energy_param}_cal<{energy_range[1]}&is_usable"
    )

    utime_array = ecal_class.data["timestamp"]
    select_energies = selection[f"{ecal_class.energy_param}_cal"].to_numpy()

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


def plot_cal_fit(ecal_class, figsize=[12, 8], fontsize=12, erange=[200, 2700]):
    pk_pars = ecal_class.results["pk_pars"]
    fitted_peaks = ecal_class.results["fitted_keV"]

    fitted_gof_funcs = []
    for i, peak in enumerate(ecal_class.glines):
        if peak in fitted_peaks:
            fitted_gof_funcs.append(ecal_class.gof_funcs[i])

    mus = [
        pgf.get_mu_func(func_i, pars_i)
        for func_i, pars_i in zip(fitted_gof_funcs, pk_pars)
    ]

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    cal_bins = np.arange(0, np.nanmax(mus) * 1.1, 10)

    ax1.scatter(fitted_peaks, mus, marker="x", c="b")

    ax1.plot(pgf.poly(cal_bins, ecal_class.pars), cal_bins, lw=1, c="g")

    ax1.grid()
    ax1.set_xlim([erange[0], erange[1]])
    ax1.set_ylabel("Energy (ADC)")
    ax2.plot(
        fitted_peaks,
        pgf.poly(np.array(mus), ecal_class.pars) - fitted_peaks,
        lw=0,
        marker="x",
        c="b",
    )
    ax2.grid()
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Residuals (keV)")
    plt.close()
    return fig


def plot_eres_fit(ecal_class, figsize=[12, 8], fontsize=12, erange=[200, 2700]):
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
        else:
            fwhm_peaks = np.append(fwhm_peaks, peak)
    fit_fwhms = np.delete(fwhms, [indexes])
    fit_dfwhms = np.delete(dfwhms, [indexes])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.errorbar(fwhm_peaks, fit_fwhms, yerr=fit_dfwhms, marker="x", lw=0, c="b")

    fwhm_slope_bins = np.arange(erange[0], erange[1], 10)

    qbb_line_vx = [2039.0, 2039.0]
    qbb_line_vy = [
        0.9 * np.nanmin(fwhm_slope(fwhm_slope_bins, *ecal_class.fit_pars)),
        ecal_class.fit_qbb,
    ]
    qbb_line_hx = [erange[0], 2039.0]
    qbb_line_hy = [ecal_class.fit_qbb, ecal_class.fit_qbb]

    ax1.plot(
        fwhm_slope_bins, fwhm_slope(fwhm_slope_bins, *ecal_class.fit_pars), lw=1, c="g"
    )
    ax1.plot(qbb_line_hx, qbb_line_hy, lw=1, c="r")
    ax1.plot(qbb_line_vx, qbb_line_vy, lw=1, c="r")
    ax1.plot(
        np.nan,
        np.nan,
        "-",
        color="none",
        label=f"Qbb fwhm: {ecal_class.fit_qbb:1.2f} +- {ecal_class.qbb_err:1.2f} keV",
    )
    ax1.legend(loc="upper left", frameon=False)
    if np.isnan(ecal_class.fit_pars).all():
        [
            0.9 * np.nanmin(fit_fwhms),
            1.1 * np.nanmax(fit_fwhms),
        ]
    else:
        ax1.set_ylim(
            [
                0.9 * np.nanmin(fwhm_slope(fwhm_slope_bins, *ecal_class.fit_pars)),
                1.1 * np.nanmax(fwhm_slope(fwhm_slope_bins, *ecal_class.fit_pars)),
            ]
        )
    ax1.set_xlim([200, 2700])
    ax1.grid()
    ax1.set_ylabel("FWHM energy resolution (keV)")
    ax2.plot(
        fwhm_peaks,
        (fit_fwhms - fwhm_slope(fwhm_peaks, *ecal_class.fit_pars)) / fit_dfwhms,
        lw=0,
        marker="x",
        c="b",
    )
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Normalised Residuals")
    ax2.grid()
    plt.tight_layout()
    plt.close()
    return fig


def bin_spectrum(ecal_class, erange=[0, 3000], dx=2):
    bins = np.arange(erange[0], erange[1] + dx, dx)
    return {
        "bins": pgh.get_bin_centers(bins),
        "counts": np.histogram(
            ecal_class.data.query("is_usable")[f"{ecal_class.energy_param}_cal"], bins
        )[0],
        "cut_counts": np.histogram(
            ecal_class.data.query("~is_valid_cal&is_not_pulser")[
                f"{ecal_class.energy_param}_cal"
            ],
            bins,
        )[0],
        "pulser_counts": np.histogram(
            ecal_class.data.query("~is_not_pulser")[f"{ecal_class.energy_param}_cal"],
            bins,
        )[0],
    }


def bin_survival_fraction(ecal_class, erange=[0, 3000], dx=6):
    counts_pass, bins_pass, _ = pgh.get_hist(
        ecal_class.data.query("is_usable")[f"{ecal_class.energy_param}_cal"],
        bins=np.arange(erange[0], erange[1] + dx, dx),
    )
    counts_fail, bins_fail, _ = pgh.get_hist(
        ecal_class.data.query("~is_valid_cal&is_not_pulser")[
            f"{ecal_class.energy_param}_cal"
        ],
        bins=np.arange(erange[0], erange[1] + dx, dx),
    )
    sf = 100 * (counts_pass + 10 ** (-6)) / (counts_pass + counts_fail + 10 ** (-6))
    return {"bins": pgh.get_bin_centers(bins_pass), "sf": sf}


def energy_cal_th(
    files: list[str],
    energy_params: list[str],
    hit_dict: dict = {},
    cut_parameters: dict[str, int] = {"bl_mean": 4, "bl_std": 4, "pz_std": 4},
    lh5_path: str = "dsp",
    plot_options: dict = None,
    guess_keV: float | None = None,
    threshold: int = 0,
    p_val: float = 0,
    n_events: int = 15000,
    final_cut_field: str = "is_valid_cal",
    deg: int = 1,
) -> tuple(dict, dict):
    data = load_data(
        files,
        lh5_path,
        energy_params,
        hit_dict,
        cut_parameters=list(cut_parameters) if cut_parameters is not None else None,
    )

    data, hit_dict = apply_cuts(data, hit_dict, cut_parameters, final_cut_field)

    output_dict = {}
    plot_dict = {}
    for energy_param in energy_params:
        ecal = calibrate_parameter(
            data, energy_param, plot_options, guess_keV, threshold, p_val, n_events, deg
        )
        ecal.calibrate_parameter()
        output_dict.update(ecal.output_dict)
        hit_dict.update(ecal.hit_dict)
        if ~np.isnan(ecal.pars).all():
            ecal.fill_plot_dict()
        plot_dict[energy_param] = ecal.plot_dict

    log.info(f"Finished all calibrations")
    return hit_dict, output_dict, plot_dict
