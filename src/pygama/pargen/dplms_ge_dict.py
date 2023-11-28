"""
This module is for creating dplms dictionary for ge processing
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import pathlib
import pickle
import time
from collections import OrderedDict

import lgdo
import lgdo.lh5_store as lh5
import matplotlib.pyplot as plt
import numpy as np
from lgdo import Array
from scipy.signal import convolve, convolve2d

from pygama.math.histogram import get_hist
from pygama.math.peak_fitting import (
    extended_gauss_step_pdf,
    extended_radford_pdf,
    gauss_step_pdf,
    radford_pdf,
)
from pygama.pargen.cuts import find_pulser_properties, generate_cuts, get_cut_indexes
from pygama.pargen.dsp_optimize import run_one_dsp
from pygama.pargen.energy_cal import hpge_find_E_peaks
from pygama.pargen.energy_optimisation import (
    event_selection,
    fom_FWHM,
    fom_FWHM_with_dt_corr_fit,
    index_data,
)
from pygama.pargen.noise_optimization import calculate_spread

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


def dplms_ge_dict(
    lh5_path: str,
    fft_files: list[str],
    cal_files: list[str],
    dsp_config: dict,
    par_dsp: dict,
    par_dsp_lh5: str,
    dplms_dict: dict,
    decay_const: float = 0,
    ene_par: str = "dplmsEmax",
    display: int = 0,
) -> dict:
    """
    This function calculates the dplms dictionary for HPGe detectors.

    Parameters
    ----------
    lh5_path: str
        Name of channel to process, should be name of lh5 group in raw files
    fft_files : list[str]
        raw files with fft data
    cal_files : list[str]
        raw files with cal data
    dsp_config: dict
        dsp config file
    par_dsp: dict
        Dictionary with db parameters for dsp processing
    par_dsp_lh5: str
        Path for saving dplms coefficients
    dplms_dict: dict
        Dictionary with various parameters

    Returns
    -------
    out_dict : dict
    """

    t0 = time.time()
    log.info(f"\nSelecting baselines")
    raw_bls = load_data(
        fft_files,
        lh5_path,
        "bls",
        n_events=dplms_dict["n_baselines"],
        raw_wf_field=dplms_dict["raw_wf_field"],
    )

    dsp_bls = run_one_dsp(raw_bls, dsp_config, db_dict=par_dsp[lh5_path])
    cut_dict = generate_cuts(dsp_bls, parameters=dplms_dict["bls_cut_pars"])
    idxs = get_cut_indexes(dsp_bls, cut_dict)
    bl_field = dplms_dict["bl_field"]
    log.info(f"... {len(dsp_bls[bl_field].values.nda[idxs,:])} baselines after cuts")

    bls = dsp_bls[bl_field].values.nda[idxs, : dplms_dict["bsize"]]
    bls_par = {}
    bls_cut_pars = [par for par in dplms_dict["bls_cut_pars"].keys()]
    for par in bls_cut_pars:
        bls_par[par] = dsp_bls[par].nda
    t1 = time.time()
    log.info(
        f"total events {len(raw_bls)}, {len(bls)} baseline selected in {(t1-t0):.2f} s"
    )

    log.info(
        "\nCalculating noise matrix of length",
        dplms_dict["length"],
        "n. events",
        bls.shape[0],
        "size",
        bls.shape[1],
    )
    nmat = noise_matrix(bls, dplms_dict["length"])
    t2 = time.time()
    log.info(f"Time to calculate noise matrix {(t2-t1):.2f} s")

    log.info("\nSelecting signals")
    peaks_keV = np.array(dplms_dict["peaks_keV"])
    wsize = dplms_dict["wsize"]
    wf_field = dplms_dict["wf_field"]
    kev_widths = [tuple(kev_width) for kev_width in dplms_dict["kev_widths"]]

    raw_cal, idx_list = event_selection(
        cal_files,
        f"{lh5_path}/raw",
        dsp_config,
        par_dsp[lh5_path],
        peaks_keV,
        np.arange(0, len(peaks_keV), 1).tolist(),
        kev_widths,
        cut_parameters=dplms_dict["wfs_cut_pars"],
        n_events=dplms_dict["n_signals"],
    )
    t3 = time.time()
    log.info(
        f"Time to run event selection {(t3-t2):.2f} s, total events {len(raw_cal)}"
    )

    raw_cal = index_data(raw_cal, idx_list[-1])
    log.info(f"Produce dsp data for {len(raw_cal)} events")
    dsp_cal = run_one_dsp(raw_cal, dsp_config, db_dict=par_dsp[lh5_path])
    t4 = time.time()
    log.info(f"Time to run dsp production {(t4-t3):.2f} s")

    # minimal processing chain
    with open(dsp_config) as r:
        dsp_config = json.load(r)
    dsp_config["outputs"] = [ene_par, "dt_eff"]

    # dictionary for peak fitting
    peak_dict = {
        "peak": peaks_keV[-1],
        "kev_width": kev_widths[-1],
        "parameter": ene_par,
        "func": extended_gauss_step_pdf,
        "gof_func": gauss_step_pdf,
    }

    if display > 0:
        plot_dict = {}
        plot_dict["dplms"] = {}
        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")

    # penalized coefficients
    dp_coeffs = dplms_dict["dp_coeffs"]
    if lh5_path in dplms_dict["noisy_bl"]:
        log.info("Setting explicit zero area condition")
        za_coeff = dp_coeffs["za"]
    else:
        za_coeff = dplms_dict["dp_def"]["za"]
    dp_coeffs.pop("za")
    coeff_keys = [key for key in dp_coeffs.keys()]
    lists = [dp_coeffs[key] for key in dp_coeffs.keys()]

    prod = list(itertools.product(*lists))
    grid_dict = {}
    min_fom = float("inf")
    min_idx = None

    for i, values in enumerate(prod):
        coeff_values = dict(zip(coeff_keys, values))

        log.info(
            "\nCase",
            i,
            "->",
            ", ".join(f"{key} = {value}" for key, value in coeff_values.items()),
        )
        grid_dict[i] = coeff_values

        sel_dict = signal_selection(dsp_cal, dplms_dict, coeff_values)
        wfs = dsp_cal[wf_field].nda[sel_dict["idxs"], :]
        log.info(f"... {len(wfs)} signals after signal selection")

        ref, rmat, pmat, fmat = signal_matrices(wfs, dplms_dict["length"], decay_const)

        t_tmp = time.time()
        nm_coeff = coeff_values["nm"]
        ft_coeff = coeff_values["ft"]
        x, y, refy = filter_synthesis(
            ref,
            nm_coeff * nmat,
            rmat,
            za_coeff,
            pmat,
            ft_coeff * fmat,
            dplms_dict["length"],
            wsize,
        )
        par_dsp[lh5_path]["dplms"] = {}
        par_dsp[lh5_path]["dplms"]["length"] = dplms_dict["length"]
        par_dsp[lh5_path]["dplms"]["coefficients"] = x.tolist()
        log.info(
            f"Filter synthesis in {time.time()-t_tmp:.1f} s, filter area", np.sum(x)
        )

        t_tmp = time.time()
        dsp_opt = run_one_dsp(raw_bls, dsp_config, db_dict=par_dsp[lh5_path])
        energies = dsp_opt[ene_par].nda
        enc_results = calculate_spread(energies, 10, 90, 1000)
        enc, enc_err = enc_results["fom"], enc_results["fom_err"]
        log.info(
            f"ENC: mean = {energies.mean():.2f} ADC, FOM = {enc:.2f} ± {enc_err:.2f} ADC, evaluated in {time.time()-t_tmp:.1f} s"
        )
        grid_dict[i]["enc"] = enc
        grid_dict[i]["enc_err"] = enc_err

        if display > 0:
            hist, bins, var = get_hist(energies, range=(-20, 20), dx=0.1)
            bc = (bins[:-1] + bins[1:]) / 2.0
            ax.plot(
                bc,
                hist,
                ds="steps",
                label=f"{ene_par} - ENC = {enc:.3f} ± {enc_err:.3f} ADC",
            )
            ax.set_xlabel("energy (ADC)")
            ax.set_ylabel("counts")
            ax.legend(loc="upper right")

        t_tmp = time.time()
        dsp_opt = run_one_dsp(raw_cal, dsp_config, db_dict=par_dsp[lh5_path])

        try:
            res = fom_FWHM_with_dt_corr_fit(
                dsp_opt,
                peak_dict,
                "QDrift",
                idxs=np.where(~np.isnan(dsp_opt["dt_eff"].nda))[0],
            )
        except:
            log.debug("FWHM not calculated")
            continue

        fwhm, fwhm_err, alpha, chisquare = (
            res["fwhm"],
            res["fwhm_err"],
            res["alpha"],
            res["chisquare"],
        )
        log.info(
            f"FWHM = {fwhm:.2f} ± {fwhm_err:.2f} keV, evaluated in {time.time()-t_tmp:.1f} s"
        )

        grid_dict[i]["fwhm"] = fwhm
        grid_dict[i]["fwhm_err"] = fwhm_err
        grid_dict[i]["alpha"] = alpha

        if (
            fwhm < dplms_dict["fwhm_limit"]
            and fwhm_err < dplms_dict["err_limit"]
            and chisquare < dplms_dict["chi_limit"]
        ):
            if fwhm < min_fom:
                min_idx, min_fom = i, fwhm

    if min_idx is not None:
        min_result = grid_dict[min_idx]
        best_case_values = {key: min_result[key] for key in min_result.keys()}

        enc = best_case_values.get("enc", None)
        enc_err = best_case_values.get("enc_err", 0)
        fwhm = best_case_values.get("fwhm", None)
        fwhm_err = best_case_values.get("fwhm_err", 0)
        alpha = best_case_values.get("alpha", 0)
        nm_coeff = best_case_values.get("nm", dplms_dict["dp_def"]["nm"])
        ft_coeff = best_case_values.get("ft", dplms_dict["dp_def"]["nm"])
        rt_coeff = best_case_values.get("rt", dplms_dict["dp_def"]["rt"])
        pt_coeff = best_case_values.get("pt", dplms_dict["dp_def"]["pt"])

        if all(
            v is not None
            for v in [
                enc,
                enc_err,
                fwhm,
                fwhm_err,
                alpha,
                nm_coeff,
                ft_coeff,
                rt_coeff,
                pt_coeff,
            ]
        ):
            log.info(
                f"\nBest case: FWHM = {fwhm:.2f} ± {fwhm_err:.2f} keV, ctc {alpha}"
            )
        else:
            log.error("Some values are missing in the best case results")
    else:
        log.error("Filter synthesis failed")
        nm_coeff = dplms_dict["dp_def"]["nm"]
        ft_coeff = dplms_dict["dp_def"]["ft"]
        rt_coeff = dplms_dict["dp_def"]["rt"]
        pt_coeff = dplms_dict["dp_def"]["pt"]

    # filter synthesis
    sel_dict = signal_selection(dsp_cal, dplms_dict, best_case_values)
    idxs = sel_dict["idxs"]
    wfs = dsp_cal[wf_field].nda[idxs, :]
    ref, rmat, pmat, fmat = signal_matrices(wfs, dplms_dict["length"], decay_const)

    x, y, refy = filter_synthesis(
        ref,
        nm_coeff * nmat,
        rmat,
        za_coeff,
        pmat,
        ft_coeff * fmat,
        dplms_dict["length"],
        wsize,
    )

    sto.write_object(
        Array(x),
        name="dplms",
        lh5_file=par_dsp_lh5,
        wo_mode="overwrite",
        group=lh5_path,
    )

    out_dict = {
        "dplms": {
            "length": dplms_dict["length"],
            "coefficients": f"loadlh5('{par_dsp_lh5}', '{lh5_path}/dplms')",
            "dp_coeffs": {
                "nm": nm_coeff,
                "za": za_coeff,
                "ft": ft_coeff,
                "rt": rt_coeff,
                "pt": pt_coeff,
            },
        }
    }
    out_alpha_dict = {
        f"{ene_par}_ctc": {
            "expression": f"{ene_par}*(1+dt_eff*a)",
            "parameters": {"a": round(alpha, 9)},
        }
    }
    out_dict.update({"ctc_params": out_alpha_dict})

    log.info(f"Time to complete DPLMS filter synthesis {time.time()-t0:.1f}")

    if display > 0:
        plot_dict["dplms"]["enc_hist"] = fig
        plot_dict["dplms"]["enc"] = enc
        plot_dict["dplms"]["enc_err"] = enc_err
        plot_dict["dplms"]["ref"] = ref
        plot_dict["dplms"]["coefficients"] = x

        bl_idxs = np.random.choice(len(bls), dplms_dict["n_plot"])
        bls = bls[bl_idxs]
        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
        for ii, wf in enumerate(bls):
            if ii < 10:
                ax.plot(wf, label=f"mean = {wf.mean():.1f}")
            else:
                ax.plot(wf)
        ax.legend(title=f"{lh5_path}", loc="upper right")
        plot_dict["dplms"]["bls"] = fig
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), facecolor="white")
        for ii, par in enumerate(bls_cut_pars):
            mean = cut_dict[par]["Mean Value"]
            llo, lup = cut_dict[par]["Lower Boundary"], cut_dict[par]["Upper Boundary"]
            plo, pup = mean - 2 * (mean - llo), mean + 2 * (lup - mean)
            hh, bb = np.histogram(bls_par[par], bins=np.linspace(plo, pup, 200))
            ax.flat[ii].plot(bb[1:], hh, ds="steps", label=f"cut on {par}")
            ax.flat[ii].axvline(lup, color="k", linestyle=":", label="selection")
            ax.flat[ii].axvline(llo, color="k", linestyle=":")
            ax.flat[ii].set_xlabel(par)
            ax.flat[ii].set_yscale("log")
            ax.flat[ii].legend(title=f"{lh5_path}", loc="upper right")
        plot_dict["dplms"]["bl_sel"] = fig

        wf_idxs = np.random.choice(len(wfs), dplms_dict["n_plot"])
        wfs = wfs[wf_idxs]
        peak_pos = dsp_cal["peak_pos"].nda
        peak_pos_neg = dsp_cal["peak_pos_neg"].nda
        centroid = dsp_cal["centroid"].nda
        risetime = dsp_cal["tp_90"].nda - dsp_cal["tp_10"].nda
        rt_low = dplms_dict["rt_low"]
        rt_high = dplms_dict["rt_high"]
        peak_lim = dplms_dict["peak_lim"]
        cal_par = {}
        wfs_cut_pars = [par for par in dplms_dict["wfs_cut_pars"].keys()]
        for par in wfs_cut_pars:
            cal_par[par] = dsp_cal[par].nda
        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
        for ii, wf in enumerate(wfs):
            if ii < 10:
                ax.plot(wf, label=f"centr = {centroid[ii]}")
            else:
                ax.plot(wf)
        ax.legend(title=f"{lh5_path}", loc="upper right")
        axin = ax.inset_axes([0.1, 0.15, 0.35, 0.5])
        for wf in wfs:
            axin.plot(wf)
        axin.set_xlim(wsize / 2 - dplms_dict["zoom"], wsize / 2 + dplms_dict["zoom"])
        axin.set_yticklabels("")
        plot_dict["dplms"]["wfs"] = fig
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), facecolor="white")
        wfs_cut_pars.append("centroid")
        wfs_cut_pars.append("peak_pos")
        wfs_cut_pars.append("risetime")
        for ii, par in enumerate(wfs_cut_pars):
            pspace = np.linspace(
                wsize / 2 - peak_lim, wsize / 2 + peak_lim, 2 * peak_lim
            )
            if par == "centroid":
                llo, lup = sel_dict["ct_ll"], sel_dict["ct_hh"]
                hh, bb = np.histogram(centroid, bins=pspace)
            elif par == "peak_pos":
                llo, lup = sel_dict["pp_ll"], sel_dict["pp_hh"]
                hh, bb = np.histogram(peak_pos, bins=pspace)
            elif par == "risetime":
                llo, lup = sel_dict["rt_ll"], sel_dict["rt_hh"]
                rt_bins = int((rt_high - rt_low) / dplms_dict["period"])
                rt_space = np.linspace(rt_low, rt_high, rt_bins)
                hh, bb = np.histogram(risetime, bins=rt_space)
            else:
                llo, lup = np.min(cal_par[par]), np.max(cal_par[par])
                hh, bb = np.histogram(cal_par[par], bins=np.linspace(llo, lup, 200))
            ax.flat[ii + 1].plot(bb[1:], hh, ds="steps", label=f"cut on {par}")
            ax.flat[ii + 1].axvline(
                llo, color="k", linestyle=":", label=f"sel. {llo:.1f} {lup:.1f}"
            )
            if par != "centroid":
                ax.flat[ii + 1].axvline(lup, color="k", linestyle=":")
            ax.flat[ii + 1].set_xlabel(par)
            ax.flat[ii + 1].set_yscale("log")
            ax.flat[ii + 1].legend(title=f"{lh5_path}", loc="upper right")
        roughenergy = dsp_cal["trapTmax"].nda
        roughenergy_sel = roughenergy[idxs]
        ell, ehh = roughenergy.min(), roughenergy.max()
        he, be = np.histogram(roughenergy, bins=np.linspace(ell, ehh, 1000))
        hs, be = np.histogram(roughenergy_sel, bins=np.linspace(ell, ehh, 1000))
        ax.flat[0].plot(be[1:], he, c="b", ds="steps", label="initial")
        ax.flat[0].plot(be[1:], hs, c="r", ds="steps", label="selected")
        ax.flat[0].set_xlabel("rough energy (ADC)")
        ax.flat[0].set_yscale("log")
        ax.flat[0].legend(loc="upper right", title=f"{lh5_path}")
        plot_dict["dplms"]["wf_sel"] = fig

        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
        ax.plot(np.flip(x), "r-", label=f"filter")
        ax.axhline(0, color="black", linestyle=":")
        ax.legend(loc="upper right", title=f"{lh5_path}")
        axin = ax.inset_axes([0.6, 0.1, 0.35, 0.33])
        axin.plot(np.flip(x), "r-")
        axin.set_xlim(
            dplms_dict["length"] / 2 - dplms_dict["zoom"],
            dplms_dict["length"] / 2 + dplms_dict["zoom"],
        )
        axin.set_yticklabels("")
        ax.indicate_inset_zoom(axin)

        return out_dict, plot_dict
    else:
        return out_dict


def load_data(
    raw_file: list[str],
    lh5_path: str,
    sel_type: str,
    peaks: np.array = [],
    n_events: int = 5000,
    e_lower_lim: float = 1200,
    e_upper_lim: float = 2700,
    raw_wf_field: str = "waveform",
) -> lgdo.Table:
    sto = lh5.LH5Store()
    df = lh5.load_dfs(raw_file, ["daqenergy", "timestamp"], f"{lh5_path}/raw")

    if sel_type == "bls":
        cuts = np.where(df.daqenergy.values == 0)[0]
        idx_list = []
        waveforms = sto.read_object(
            f"{lh5_path}/raw/{raw_wf_field}", raw_file, n_rows=n_events, idx=cuts
        )[0]
        daqenergy = sto.read_object(
            f"{lh5_path}/raw/daqenergy", raw_file, n_rows=n_events, idx=cuts
        )[0]
        tb_data = lh5.Table(col_dict={"waveform": waveforms, "daqenergy": daqenergy})
        return tb_data
    else:
        pulser_props = find_pulser_properties(df, energy="daqenergy")
        if len(pulser_props) > 0:
            final_mask = None
            for entry in pulser_props:
                pulser_e, pulser_err = entry[0], entry[1]
                if pulser_err < 10:
                    pulser_err = 10
                e_cut = (df.daqenergy.values < pulser_e + pulser_err) & (
                    df.daqenergy.values > pulser_e - pulser_err
                )
                if final_mask is None:
                    final_mask = e_cut
                else:
                    final_mask = final_mask | e_cut
            ids = final_mask
            log.debug(f"pulser found: {pulser_props}")
        else:
            log.debug("no pulser")
            ids = np.zeros(len(df.daqenergy.values), dtype=bool)
        if sel_type == "pul":
            cuts = np.where(ids == True)[0]
            log.debug(f"{len(cuts)} events found for pulser")
            waveforms = sto.read_object(
                f"{lh5_path}/raw/waveform", raw_file, n_rows=n_events, idx=cuts
            )[0]
            daqenergy = sto.read_object(
                f"{lh5_path}/raw/daqenergy", raw_file, n_rows=n_events, idx=cuts
            )[0]
            tb_data = lh5.Table(
                col_dict={"waveform": waveforms, "daqenergy": daqenergy}
            )
            return tb_data
        else:
            # Get events around peak using raw file values
            initial_mask = (df.daqenergy.values > 0) & (~ids)
            rough_energy = df.daqenergy.values[initial_mask]
            initial_idxs = np.where(initial_mask)[0]

            guess_keV = 2620 / np.nanpercentile(rough_energy, 99)
            Euc_min = 0  # threshold / guess_keV * 0.6
            Euc_max = 2620 / guess_keV * 1.1
            dEuc = 1  # / guess_keV
            hist, bins, var = get_hist(rough_energy, range=(Euc_min, Euc_max), dx=dEuc)
            detected_peaks_locs, detected_peaks_keV, roughpars = hpge_find_E_peaks(
                hist, bins, var, peaks
            )
            log.debug(
                f"detected {detected_peaks_keV} keV peaks at {detected_peaks_locs}"
            )
            e_lower_lim = (e_lower_lim - roughpars[1]) / roughpars[0]
            e_upper_lim = (e_upper_lim - roughpars[1]) / roughpars[0]
            log.debug(f"lower_lim: {e_lower_lim}, upper_lim: {e_upper_lim}")
            mask = (rough_energy > e_lower_lim) & (rough_energy < e_upper_lim)
            cuts = initial_idxs[mask][:]
            log.debug(f"{len(cuts)} events found in energy range")
            rough_energy = rough_energy[mask]
            rough_energy = rough_energy[:n_events]
            rough_energy = rough_energy * roughpars[0] + roughpars[1]
            waveforms = sto.read_object(
                f"{lh5_path}/raw/waveform", raw_file, n_rows=n_events, idx=cuts
            )[0]
            daqenergy = sto.read_object(
                f"{lh5_path}/raw/daqenergy", raw_file, n_rows=n_events, idx=cuts
            )[0]
            tb_data = lh5.Table(
                col_dict={"waveform": waveforms, "daqenergy": daqenergy}
            )
            return tb_data, rough_energy


def is_valid_centroid(
    centroid: np.array, lim: int, size: int, full_size: int
) -> list[bool]:
    llim = size / 2 - lim
    hlim = full_size - size / 2
    idxs = (centroid > llim) & (centroid < hlim)
    return idxs, llim, hlim


def is_not_pile_up(
    peak_pos: np.array, peak_pos_neg: np.array, thr: int, lim: int, size: int
) -> list[bool]:
    bin_edges = np.linspace(size / 2 - lim, size / 2 + lim, 2 * lim)
    hist, bin_edges = np.histogram(peak_pos, bins=bin_edges)

    thr = thr * hist.max() / 100
    low_thr_idxs = np.where(hist[: hist.argmax()] < thr)[0]
    upp_thr_idxs = np.where(hist[hist.argmax() :] < thr)[0]

    idx_low = low_thr_idxs[-1] if low_thr_idxs.size > 0 else 0
    idx_upp = (
        upp_thr_idxs[0] + hist.argmax() if upp_thr_idxs.size > 0 else len(hist) - 1
    )

    llow, lupp = bin_edges[idx_low], bin_edges[idx_upp]

    idxs = []
    for n, nn in zip(peak_pos, peak_pos_neg):
        condition1 = np.count_nonzero(n > 0) == 1
        condition2 = (
            np.count_nonzero((n > 0) & ((n < llow) | (n > lupp) & (n < size))) == 0
        )
        condition3 = np.count_nonzero(nn > 0) == 0
        idxs.append(condition1 and condition2 and condition3)
    return idxs, llow, lupp


def is_valid_risetime(risetime: np.array, llim: int, perc: float):
    hlim = np.percentile(risetime[~np.isnan(risetime)], perc)
    idxs = (risetime >= llim) & (risetime <= hlim)
    return idxs, llim, hlim


def signal_selection(dsp_cal, dplms_dict, coeff_values):
    peak_pos = dsp_cal["peak_pos"].nda
    peak_pos_neg = dsp_cal["peak_pos_neg"].nda
    centroid = dsp_cal["centroid"].nda
    risetime = dsp_cal["tp_90"].nda - dsp_cal["tp_10"].nda

    rt_low = dplms_dict["rt_low"]
    rt_high = dplms_dict["rt_high"]
    peak_lim = dplms_dict["peak_lim"]
    wsize = dplms_dict["wsize"]
    bsize = dplms_dict["bsize"]

    centroid_lim = dplms_dict["centroid_lim"]
    if "rt" in coeff_values:
        perc = coeff_values["rt"]
    else:
        perc = dplms_dict["dp_def"]["rt"]
    if "pt" in coeff_values:
        thr = coeff_values["pt"]
    else:
        thr = dplms_dict["dp_def"]["rt"]

    idxs_ct, ct_ll, ct_hh = is_valid_centroid(centroid, centroid_lim, wsize, bsize)
    log.info(f"... {len(peak_pos[idxs_ct,:])} signals after alignment")

    idxs_pp, pp_ll, pp_hh = is_not_pile_up(peak_pos, peak_pos_neg, thr, peak_lim, wsize)
    log.info(f"... {len(peak_pos[idxs_pp,:])} signals after pile-up cut")

    idxs_rt, rt_ll, rt_hh = is_valid_risetime(risetime, rt_low, perc)
    log.info(f"... {len(peak_pos[idxs_rt,:])} signals after risetime cut")

    idxs = idxs_ct & idxs_pp & idxs_rt
    sel_dict = {
        "idxs": idxs,
        "ct_ll": ct_ll,
        "ct_hh": ct_hh,
        "pp_ll": pp_ll,
        "pp_hh": pp_hh,
        "rt_ll": rt_ll,
        "rt_hh": rt_hh,
    }
    return sel_dict


def noise_matrix(bls: np.array, length: int) -> np.array:
    nev, size = bls.shape
    ref = np.mean(bls, axis=0)
    offset = np.mean(ref)
    bls = bls - offset
    nmat = np.matmul(bls.T, bls, dtype=float) / nev
    kernel = np.identity(size - length + 1)
    nmat = convolve2d(nmat, kernel, boundary="symm", mode="valid") / (size - length + 1)
    return nmat


def signal_matrices(
    wfs: np.array, length: int, decay_const: float, ff: int = 2
) -> np.array:
    nev, size = wfs.shape
    lo = size // 2 - 100
    flo = size // 2 - length // 2
    fhi = size // 2 + length // 2
    offsets = np.mean(wfs[:, :lo], axis=1)
    wfs = wfs - offsets[:, np.newaxis]

    # Reference signal
    ref = np.sum(wfs, axis=0)
    ref /= np.max(ref)
    rmat = np.outer(ref[flo:fhi], ref[flo:fhi])

    # Pile-up matrix
    if decay_const > 0:
        decay = np.exp(-np.arange(length) / decay_const)
    else:
        decay = np.zeros(length)
    pmat = np.outer(decay, decay)

    # Flat top matrix
    flo -= ff // 2
    fhi += ff // 2
    wfs = wfs[:, flo:fhi]
    fmat = np.matmul(wfs.T, wfs, dtype=float) / nev
    m1 = ((1, -1), (-1, 1))
    fmat = convolve2d(fmat, m1, boundary="symm", mode="valid")
    if ff > 0:
        fmat = convolve2d(fmat, np.identity(ff), boundary="symm", mode="valid") / ff
    return ref, rmat, pmat, fmat


def filter_synthesis(
    ref: np.array,
    nmat: np.array,
    rmat: np.array,
    za: int,
    pmat: np.array,
    fmat: np.array,
    length: int,
    size: int,
) -> np.array:
    mat = nmat + rmat + za * np.ones([length, length]) + pmat + fmat
    flo = (size // 2) - (length // 2)
    fhi = (size // 2) + (length // 2)
    x = np.linalg.solve(mat, ref[flo:fhi])
    y = convolve(ref, np.flip(x), mode="valid")
    maxy = np.max(y)
    x /= maxy
    y /= maxy
    refy = ref[(size // 2) - (len(y) // 2) : (size // 2) + (len(y) // 2)]
    return x, y, refy
