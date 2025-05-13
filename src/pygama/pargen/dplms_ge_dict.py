"""
This module is for creating dplms dictionary for ge processing
"""

from __future__ import annotations

import itertools
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from lgdo import Table
from scipy.signal import convolve, convolve2d
from scipy.stats import chi2

import pygama.math.distributions as pmd  # noqa: pycln, F401
from pygama.pargen.data_cleaning import generate_cuts
from pygama.pargen.dsp_optimize import run_one_dsp
from pygama.pargen.energy_optimisation import fom_fwhm_with_alpha_fit

log = logging.getLogger(__name__)


def dplms_ge_dict(
    raw_fft: Table,
    raw_cal: Table,
    dsp_config: dict,
    par_dsp: dict,
    dplms_dict: dict,
    fom_func,
    decay_const: float = 0,
    ene_par: str = "dplmsEmax",
    display: int = 0,
) -> dict:
    """
    This function calculates the dplms dictionary for HPGe detectors.

    Parameters
    ----------
    raw_fft
        table with fft raw data
    raw_cal
        table with cal raw data
    dsp_config
        dsp config file
    par_dsp
        Dictionary with db parameters for dsp processing
    dplms_dict
        Dictionary with various parameters
    fom_func
        Function for peak fits

    Returns
    -------
    out_dict
    """

    t0 = time.time()
    log.info("Selecting baselines")

    dsp_fft = run_one_dsp(raw_fft, dsp_config, db_dict=par_dsp)

    cut_dict = generate_cuts(dsp_fft, cut_dict=dplms_dict["bls_cut_pars"])
    log.debug(f"Cuts are {cut_dict}")
    idxs = np.full(len(dsp_fft), True, dtype=bool)
    for outname, info in cut_dict.items():
        outcol = dsp_fft.eval(info["expression"], info.get("parameters", None))
        dsp_fft.add_column(outname, outcol)
    for cut in cut_dict:
        idxs = dsp_fft[cut].nda & idxs
    log.debug("Applied Cuts")

    bl_field = dplms_dict["bl_field"]
    log.info(f"... {len(dsp_fft[bl_field].values.nda[idxs, :])} baselines after cuts")

    bls = dsp_fft[bl_field].values.nda[idxs, : dplms_dict["bsize"]]
    bls_par = {}
    bls_cut_pars = [par for par in dplms_dict["bls_cut_pars"]]
    for par in bls_cut_pars:
        bls_par[par] = dsp_fft[dplms_dict["bls_cut_pars"][par]["cut_parameter"]].nda
    t1 = time.time()
    log.info(
        f"total events {len(raw_fft)}, {len(bls)} baseline selected in {(t1-t0):.2f} s"
    )

    log.info(
        f'Calculating noise matrix of length {dplms_dict["length"]} n. events: {bls.shape[0]}, size: {bls.shape[1]}'
    )
    nmat = noise_matrix(bls, dplms_dict["length"])
    t2 = time.time()
    log.info(f"Time to calculate noise matrix {(t2-t1):.2f} s")

    log.info("Selecting signals")
    wsize = dplms_dict["wsize"]
    wf_field = dplms_dict["wf_field"]
    peaks_kev = np.array(dplms_dict["peaks_kev"])
    kev_widths = [tuple(kev_width) for kev_width in dplms_dict["kev_widths"]]

    log.info(f"Produce dsp data for {len(raw_cal)} events")
    dsp_cal = run_one_dsp(raw_cal, dsp_config, db_dict=par_dsp)
    t3 = time.time()
    log.info(f"Time to run dsp production {(t3-t2):.2f} s")

    dsp_config["outputs"] = [ene_par, "dt_eff"]

    # dictionary for peak fitting
    peak_dict = {
        "peak": peaks_kev[-1],
        "kev_width": kev_widths[-1],
        "parameter": ene_par,
        "func": fom_func,
    }

    if display > 0:
        plot_dict = {}
        plot_dict["dplms"] = {}

    # penalized coefficients
    dp_coeffs = dplms_dict["dp_coeffs"]
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
        log_msg = f"Case {i} ->"
        for key, value in coeff_values.items():
            log_msg += f" {key} = {value}"
        log.info(log_msg)

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
        par_dsp["dplms"] = {"length": dplms_dict["length"], "coefficients": x}
        log.info(
            f"Filter synthesis in {time.time()-t_tmp:.1f} s, filter area {np.sum(x)}"
        )

        t_tmp = time.time()
        dsp_opt = run_one_dsp(raw_cal, dsp_config, db_dict=par_dsp)

        try:
            res = fom_fwhm_with_alpha_fit(
                dsp_opt,
                peak_dict,
                "dt_eff",
                idxs=np.where(~np.isnan(dsp_opt["dt_eff"].nda))[0],
                frac_max=0.5,
            )
        except Exception:
            log.debug("FWHM not calculated")
            continue

        fwhm, fwhm_err, alpha, chisquare = (
            res["fwhm"],
            res["fwhm_err"],
            res["alpha"],
            res["chisquare"],
        )
        p_val = chi2.sf(chisquare[0], chisquare[1])
        log.info(
            f"FWHM = {fwhm:.2f} ± {fwhm_err:.2f} keV, p_val={p_val} evaluated in {time.time()-t_tmp:.1f} s"
        )
        grid_dict[i]["fwhm"] = fwhm
        grid_dict[i]["fwhm_err"] = fwhm_err
        grid_dict[i]["alpha"] = alpha
        if (
            fwhm < dplms_dict["fwhm_limit"]
            and fwhm_err < dplms_dict["err_limit"]
            and p_val > dplms_dict["p_val_lim"]
            and ~np.isnan(fwhm)
        ):
            if fwhm < min_fom:
                min_idx, min_fom = i, fwhm

    if min_idx is not None:
        min_result = grid_dict[min_idx]
        best_case_values = {key: min_result[key] for key in min_result.keys()}

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
            log.debug("Some values are missing in the best case results")
    else:
        log.debug("Filter synthesis failed")
        nm_coeff = dplms_dict["dp_def"]["nm"]
        ft_coeff = dplms_dict["dp_def"]["ft"]
        rt_coeff = dplms_dict["dp_def"]["rt"]
        pt_coeff = dplms_dict["dp_def"]["pt"]
        best_case_values = {}
        alpha = 0

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

    out_dict = {
        "dplms": {
            "length": dplms_dict["length"],
            "coefficients": x,
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
        plot_dict = {"ref": ref, "coefficients": x}

        bl_idxs = np.random.choice(len(bls), dplms_dict["n_plot"])
        bls = bls[bl_idxs]
        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
        for ii, wf in enumerate(bls):
            if ii < 10:
                ax.plot(wf, label=f"mean = {wf.mean():.1f}")
            else:
                ax.plot(wf)
        ax.legend(loc="upper right")
        plot_dict["bls"] = fig
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), facecolor="white")
        for ii, par in enumerate(bls_cut_pars):
            if "parameters" in cut_dict[par]:
                if "a" in cut_dict[par]["parameters"]:
                    llo = cut_dict[par]["parameters"]["a"]
                else:
                    llo = np.nan
                if "b" in cut_dict[par]["parameters"]:
                    lup = cut_dict[par]["parameters"]["b"]
                else:
                    lup = np.nan
            mean = (lup + llo) / 2
            plo, pup = mean - 2 * (mean - llo), mean + 2 * (lup - mean)
            hh, bb = np.histogram(bls_par[par], bins=np.linspace(plo, pup, 200))
            ax.flat[ii].plot(bb[1:], hh, ds="steps", label=f"cut on {par}")
            ax.flat[ii].axvline(lup, color="k", linestyle=":", label="selection")
            ax.flat[ii].axvline(llo, color="k", linestyle=":")
            ax.flat[ii].set_xlabel(par)
            ax.flat[ii].set_yscale("log")
            ax.flat[ii].legend(loc="upper right")
        plot_dict["bl_sel"] = fig

        wf_idxs = np.random.choice(len(wfs), dplms_dict["n_plot"])
        wfs = wfs[wf_idxs]
        centroid = dsp_cal["centroid"].nda

        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
        for ii, wf in enumerate(wfs):
            if ii < 10:
                ax.plot(wf, label=f"centr = {centroid[ii]}")
            else:
                ax.plot(wf)
        ax.legend(loc="upper right")
        axin = ax.inset_axes([0.1, 0.15, 0.35, 0.5])
        for wf in wfs:
            axin.plot(wf)
        axin.set_xlim(wsize / 2 - dplms_dict["zoom"], wsize / 2 + dplms_dict["zoom"])
        axin.set_yticklabels("")
        plot_dict["wfs"] = fig

        peak_pos = dsp_cal["peak_pos"].nda
        risetime = dsp_cal["tp_90"].nda - dsp_cal["tp_10"].nda
        rt_low = dplms_dict["rt_low"]
        rt_high = dplms_dict["rt_high"]
        peak_lim = dplms_dict["peak_lim"]
        cal_par = {}
        wfs_cut_pars = ["centroid", "peak_pos", "risetime"]

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), facecolor="white")

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
            ax.flat[ii + 1].legend(loc="upper right")
        roughenergy = dsp_cal["trapTmax"].nda
        roughenergy_sel = roughenergy[idxs]
        ell, ehh = roughenergy.min(), roughenergy.max()
        he, be = np.histogram(roughenergy, bins=np.linspace(ell, ehh, 1000))
        hs, be = np.histogram(roughenergy_sel, bins=np.linspace(ell, ehh, 1000))
        ax.flat[0].plot(be[1:], he, c="b", ds="steps", label="initial")
        ax.flat[0].plot(be[1:], hs, c="r", ds="steps", label="selected")
        ax.flat[0].set_xlabel("rough energy (ADC)")
        ax.flat[0].set_yscale("log")
        ax.flat[0].legend(loc="upper right")
        plot_dict["wf_sel"] = fig

        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
        ax.plot(x, "r-", label="filter")
        ax.axhline(0, color="black", linestyle=":")
        ax.legend(loc="upper right")
        axin = ax.inset_axes([0.6, 0.1, 0.35, 0.33])
        axin.plot(x, "r-")
        axin.set_xlim(
            dplms_dict["length"] / 2 - dplms_dict["zoom"],
            dplms_dict["length"] / 2 + dplms_dict["zoom"],
        )
        axin.set_yticklabels("")
        ax.indicate_inset_zoom(axin)

        return out_dict, plot_dict
    else:
        return out_dict


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
    log.info(f"... {len(peak_pos[idxs_ct, :])} signals after alignment")

    idxs_pp, pp_ll, pp_hh = is_not_pile_up(peak_pos, peak_pos_neg, thr, peak_lim, wsize)
    log.info(f"... {len(peak_pos[idxs_pp, :])} signals after pile-up cut")

    idxs_rt, rt_ll, rt_hh = is_valid_risetime(risetime, rt_low, perc)
    log.info(f"... {len(peak_pos[idxs_rt, :])} signals after risetime cut")

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
    flip: bool = True,
) -> np.array:
    mat = nmat + rmat + za * np.ones([length, length]) + pmat + fmat
    flo = (size // 2) - (length // 2)
    fhi = (size // 2) + (length // 2)
    x = np.linalg.solve(mat, ref[flo:fhi]).astype(np.float32)
    y = convolve(ref, np.flip(x), mode="valid")
    maxy = np.max(y)
    x /= maxy
    y /= maxy
    refy = ref[(size // 2) - (len(y) // 2) : (size // 2) + (len(y) // 2)]
    if flip:
        return np.flip(x), y, refy
    else:
        return x, y, refy
