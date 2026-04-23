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
    ctc_par: str = "dt_eff",
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

    bl_field = dplms_dict["bl_field"]
    log.info("... %s baselines", len(dsp_fft[bl_field].values.nda))

    bls = dsp_fft[bl_field].values.nda[:, : dplms_dict["bsize"]]
    bls_par = {}
    bls_cut_pars = list(dplms_dict["bls_cut_pars"])
    for par in bls_cut_pars:
        bls_par[par] = dsp_fft[dplms_dict["bls_cut_pars"][par]["cut_parameter"]].nda
    t1 = time.time()
    log.info(
        "total events %s, %s baseline selected in %.2f s",
        len(raw_fft),
        len(bls),
        t1 - t0,
    )
    log.info(
        "Calculating noise matrix of length %s n. events: %s, size: %s",
        dplms_dict["length"],
        bls.shape[0],
        bls.shape[1],
    )

    nmat = noise_matrix(bls, dplms_dict["length"])
    t2 = time.time()
    log.info("Time to calculate noise matrix %.2f s", t2 - t1)

    log.info("Selecting signals")
    wsize = dplms_dict["wsize"]
    wf_field = dplms_dict["wf_field"]
    peaks_kev = np.array(dplms_dict["peaks_kev"])
    kev_widths = [tuple(kev_width) for kev_width in dplms_dict["kev_widths"]]

    log.info("Produce dsp data for %s events", len(raw_cal))
    dsp_cal = run_one_dsp(raw_cal, dsp_config, db_dict=par_dsp)
    t3 = time.time()
    log.info("Time to run dsp production %.2f s", t3 - t2)

    dsp_config["outputs"] = [ene_par, ctc_par]

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
    coeff_keys = list(dp_coeffs.keys())
    lists = [dp_coeffs[key] for key in dp_coeffs]

    prod = list(itertools.product(*lists))
    grid_dict = {}
    min_fom = float("inf")
    min_idx = None

    for i, values in enumerate(prod):
        coeff_values = dict(zip(coeff_keys, values, strict=False))
        log_msg = f"Case {i} ->"
        for key, value in coeff_values.items():
            log_msg += f" {key} = {value}"
        log.info(log_msg)

        grid_dict[i] = coeff_values

        sel_dict = signal_selection(dsp_cal, dplms_dict, coeff_values)
        wfs = dsp_cal[wf_field].nda[sel_dict["idxs"], :]
        log.info("... %s signals after signal selection", len(wfs))

        ref, rmat, pmat, fmat = signal_matrices(wfs, dplms_dict["length"], decay_const)

        t_tmp = time.time()
        nm_coeff = coeff_values["nm"]
        za_coeff = coeff_values["za"]
        pl_coeff = coeff_values["pl"]
        ft_coeff = coeff_values["ft"]
        x, _y, _refy = filter_synthesis(
            ref,
            nm_coeff * nmat,
            rmat,
            za_coeff,
            pmat * pl_coeff,
            ft_coeff * fmat,
            dplms_dict["length"],
            wsize,
        )
        par_dsp["dplms"] = {"length": dplms_dict["length"], "coefficients": x}
        log.info(
            "Filter synthesis in %.1f s, filter area %s",
            time.time() - t_tmp,
            np.sum(x),
        )

        t_tmp = time.time()
        dsp_opt = run_one_dsp(raw_cal, dsp_config, db_dict=par_dsp)

        try:
            res = fom_fwhm_with_alpha_fit(
                dsp_opt,
                peak_dict,
                ctc_par,
                idxs=np.where(~np.isnan(dsp_opt[ctc_par].nda))[0],
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
            "FWHM = %.2f ± %.2f keV, p_val=%s evaluated in %.1f s",
            fwhm,
            fwhm_err,
            p_val,
            time.time() - t_tmp,
        )
        grid_dict[i]["fwhm"] = fwhm
        grid_dict[i]["fwhm_err"] = fwhm_err
        grid_dict[i]["alpha"] = alpha
        if (
            fwhm < dplms_dict["fwhm_limit"]
            and fwhm_err < dplms_dict["err_limit"]
            and p_val > dplms_dict["p_val_lim"]
            and ~np.isnan(fwhm)
        ) and fwhm < min_fom:
            min_idx, min_fom = i, fwhm

    if min_idx is not None:
        min_result = grid_dict[min_idx]
        best_case_values = {key: min_result[key] for key in min_result}

        fwhm = best_case_values.get("fwhm")
        fwhm_err = best_case_values.get("fwhm_err", 0)
        alpha = best_case_values.get("alpha", 0)
        nm_coeff = best_case_values.get("nm", dplms_dict["dp_def"]["nm"])
        za_coeff = best_case_values.get("za", dplms_dict["dp_def"]["za"])
        pl_coeff = best_case_values.get("pl", dplms_dict["dp_def"]["pl"])
        ft_coeff = best_case_values.get("ft", dplms_dict["dp_def"]["ft"])
        rt_coeff = best_case_values.get("rt", dplms_dict["dp_def"]["rt"])
        pt_coeff = best_case_values.get("pt", dplms_dict["dp_def"]["pt"])
        if all(
            v is not None
            for v in [
                fwhm,
                fwhm_err,
                alpha,
                nm_coeff,
                za_coeff,
                pl_coeff,
                ft_coeff,
                rt_coeff,
                pt_coeff,
            ]
        ):
            log.info("\nBest case: %s", best_case_values)
        else:
            log.debug("Some values are missing in the best case results")
    else:
        log.debug("Filter synthesis failed")
        nm_coeff = dplms_dict["dp_def"]["nm"]
        za_coeff = dplms_dict["dp_def"]["za"]
        pl_coeff = dplms_dict["dp_def"]["pl"]
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

    x, _y, _refy = filter_synthesis(
        ref,
        nm_coeff * nmat,
        rmat,
        za_coeff,
        pmat * pl_coeff,
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
                "pl": pl_coeff,
                "ft": ft_coeff,
                "rt": rt_coeff,
                "pt": pt_coeff,
            },
        }
    }
    out_alpha_dict = {
        f"{ene_par}_ctc": {
            "expression": f"{ene_par}*(1+{ctc_par}*a)",
            "parameters": {"a": round(alpha, 9)},
        }
    }
    out_dict.update({"ctc_params": out_alpha_dict})

    log.info("Time to complete DPLMS filter synthesis %.1f", time.time() - t0)

    if display > 0:
        plot_dict = {"ref": ref, "coefficients": x}

        bl_idxs = np.random.choice(len(bls), dplms_dict["n_plot"])  # noqa: NPY002
        bls = bls[bl_idxs]
        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
        for ii, wf in enumerate(bls):
            if ii < 10:
                ax.plot(wf, label=f"mean = {wf.mean():.1f}")
            else:
                ax.plot(wf)
        ax.legend(loc="upper right")
        plot_dict["bls"] = fig
        plt.close()

        wf_idxs = np.random.choice(len(wfs), dplms_dict["n_plot"])  # noqa: NPY002
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
        plt.close()

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
        plt.close()

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
        plot_dict["filter"] = fig
        plt.close()

        return out_dict, plot_dict
    return out_dict


def is_valid_centroid(centroid: np.array, lim: int, size: int, full_size: int) -> tuple:
    """
    Select waveforms whose centroid lies within a valid alignment window.

    Parameters
    ----------
    centroid
        Per-event centroid positions (samples).
    lim
        Half-width of the allowed centroid displacement from the nominal
        centre of the waveform window (samples).
    size
        Length of the waveform window used for filter synthesis (samples).
    full_size
        Total waveform buffer length (samples).

    Returns
    -------
    idxs
        Boolean mask that is True for valid events.
    llim
        Lower centroid boundary used (samples).
    hlim
        Upper centroid boundary used (samples).
    """
    llim = size / 2 - lim
    hlim = full_size - size / 2
    idxs = (centroid > llim) & (centroid < hlim)
    return idxs, llim, hlim


def is_not_pile_up(
    peak_pos: np.array, peak_pos_neg: np.array, thr: int, lim: int, size: int
) -> tuple:
    """
    Reject pile-up events based on the presence of secondary peaks.

    A histogram of positive-peak positions is used to define a single-peak
    band; events with additional peaks outside that band, or any negative
    peaks, are flagged as pile-up.

    Parameters
    ----------
    peak_pos
        2-D array of positive peak positions per event (samples).
    peak_pos_neg
        2-D array of negative peak positions per event (samples).
    thr
        Amplitude threshold (percent of histogram maximum) used to define
        the edges of the primary peak band.
    lim
        Half-width of the search window around the nominal waveform centre
        (samples).
    size
        Length of the waveform window used for filter synthesis (samples).

    Returns
    -------
    idxs
        Per-event mask; True means the event is pile-up free.
    llow
        Lower boundary of the accepted peak band (samples).
    lupp
        Upper boundary of the accepted peak band (samples).
    """
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
    for n, nn in zip(peak_pos, peak_pos_neg, strict=False):
        condition1 = np.count_nonzero(n > 0) == 1
        condition2 = (
            np.count_nonzero((n > 0) & ((n < llow) | (n > lupp) & (n < size))) == 0
        )
        condition3 = np.count_nonzero(nn > 0) == 0
        idxs.append(condition1 and condition2 and condition3)
    return idxs, llow, lupp


def is_valid_risetime(risetime: np.array, llim: int, perc: float) -> tuple:
    """
    Select waveforms within an acceptable risetime range.

    Parameters
    ----------
    risetime
        Per-event risetime values (samples), e.g. ``tp_90 - tp_10``.
    llim
        Minimum allowed risetime (samples).
    perc
        Upper percentile cutoff applied to the non-NaN risetime distribution.

    Returns
    -------
    idxs
        Boolean mask; True for events within the accepted risetime range.
    llim
        Lower risetime boundary used (samples).
    hlim
        Upper risetime boundary used (samples, computed from *perc*).
    """
    hlim = np.percentile(risetime[~np.isnan(risetime)], perc)
    idxs = (risetime >= llim) & (risetime <= hlim)
    return idxs, llim, hlim


def signal_selection(dsp_cal, dplms_dict, coeff_values) -> dict:
    """
    Apply centroid, pile-up, and risetime cuts to select clean calibration signals.

    Parameters
    ----------
    dsp_cal
        DSP output table (lgdo.Table) containing at minimum ``peak_pos``,
        ``peak_pos_neg``, ``centroid``, ``tp_90``, and ``tp_10`` fields.
    dplms_dict
        Dictionary with DPLMS configuration parameters including
        ``rt_low``, ``peak_lim``, ``wsize``, ``bsize``, ``centroid_lim``, and
        ``dp_def`` sub-dictionary with ``rt`` and ``pt`` defaults.
    coeff_values
        Dictionary of per-optimisation-point overrides; may contain ``rt``
        (risetime percentile) and ``pt`` (pile-up threshold) keys.

    Returns
    -------
    sel_dict
        Selection dictionary with keys:

        * ``idxs`` - combined boolean mask of passing events
        * ``ct_ll``, ``ct_hh`` - centroid window boundaries
        * ``pp_ll``, ``pp_hh`` - pile-up peak band boundaries
        * ``rt_ll``, ``rt_hh`` - risetime window boundaries
    """
    peak_pos = dsp_cal["peak_pos"].nda
    peak_pos_neg = dsp_cal["peak_pos_neg"].nda
    centroid = dsp_cal["centroid"].nda
    risetime = dsp_cal["tp_90"].nda - dsp_cal["tp_10"].nda

    rt_low = dplms_dict["rt_low"]
    peak_lim = dplms_dict["peak_lim"]
    wsize = dplms_dict["wsize"]
    bsize = dplms_dict["bsize"]

    centroid_lim = dplms_dict["centroid_lim"]
    perc = coeff_values["rt"] if "rt" in coeff_values else dplms_dict["dp_def"]["rt"]
    thr = coeff_values["pt"] if "pt" in coeff_values else dplms_dict["dp_def"]["pt"]

    idxs_ct, ct_ll, ct_hh = is_valid_centroid(centroid, centroid_lim, wsize, bsize)
    log.info("... %s signals after alignment", len(peak_pos[idxs_ct, :]))

    idxs_pp, pp_ll, pp_hh = is_not_pile_up(peak_pos, peak_pos_neg, thr, peak_lim, wsize)
    log.info("... %s signals after pile-up cut", len(peak_pos[idxs_pp, :]))

    idxs_rt, rt_ll, rt_hh = is_valid_risetime(risetime, rt_low, perc)
    log.info("... %s signals after risetime cut", len(peak_pos[idxs_rt, :]))

    idxs = idxs_ct & idxs_pp & idxs_rt
    return {
        "idxs": idxs,
        "ct_ll": ct_ll,
        "ct_hh": ct_hh,
        "pp_ll": pp_ll,
        "pp_hh": pp_hh,
        "rt_ll": rt_ll,
        "rt_hh": rt_hh,
    }


def noise_matrix(bls: np.array, length: int) -> np.ndarray:
    """
    Compute the noise covariance matrix from baseline waveforms.

    The baselines are mean-subtracted, the raw covariance is estimated, and
    then compressed to the filter length via a sliding-window average
    (convolution with an identity kernel).

    Parameters
    ----------
    bls
        2-D array of baseline waveform segments, shape ``(n_events, n_samples)``.
    length
        Target filter length (samples).  The output matrix has shape
        ``(length, length)``.

    Returns
    -------
    nmat
        Noise covariance matrix of shape ``(length, length)``.
    """
    nev, size = bls.shape
    ref = np.mean(bls, axis=0)
    offset = np.mean(ref)
    bls = bls - offset
    nmat = np.matmul(bls.T, bls, dtype=float) / nev
    kernel = np.identity(size - length + 1)
    return convolve2d(nmat, kernel, boundary="symm", mode="valid") / (size - length + 1)


def noise_matrix_corr(
    bls: np.ndarray, bls_corr: list[np.ndarray], length: int
) -> np.ndarray:
    """
    Compute a block noise covariance matrix including cross-channel correlations.

    Extends :func:`noise_matrix` to the multi-channel case by building an
    ``(n_channels * length, n_channels * length)`` block matrix where each
    block is the cross-channel noise covariance between two channels,
    compressed to *length* via the same sliding-window average used in
    :func:`noise_matrix`.

    Parameters
    ----------
    bls
        Baseline waveforms for the primary channel, shape
        ``(n_events, n_samples)``.
    bls_corr
        List of baseline waveform arrays for correlated channels, each of
        shape ``(n_events, n_samples)``.
    length
        Target filter length (samples).

    Returns
    -------
    nmat
        Symmetrised block noise covariance matrix of shape
        ``(n_channels * length, n_channels * length)``.
    """
    all_bls = [bls, *bls_corr]
    n = len(all_bls)

    processed = []
    for arr in all_bls:
        ref = np.mean(arr, axis=0)
        processed.append(arr - np.mean(ref))

    size = processed[0].shape[1]
    kernel = np.identity(size - length + 1)

    block_mat = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            nev = processed[i].shape[0]
            nij = np.matmul(processed[i].T, processed[j], dtype=float) / nev
            nij = convolve2d(nij, kernel, boundary="symm", mode="valid") / (
                size - length + 1
            )
            block_mat[i][j] = nij
            if i != j:
                block_mat[j][i] = nij.T

    big_size = n * length
    nmat = np.full((big_size, big_size), np.nan)

    for i in range(n):
        for j in range(n):
            if block_mat[i][j] is not None:
                nmat[i * length : (i + 1) * length, j * length : (j + 1) * length] = (
                    block_mat[i][j]
                )

    return 0.5 * (nmat + nmat.T)


def signal_matrices(
    wfs: np.array, length: int, decay_const: float, ff: int = 2
) -> tuple:
    """
    Compute the signal-shape matrices needed for DPLMS filter synthesis.

    Three matrices are constructed from the waveform ensemble:

    * **rmat** - outer product of the mean reference pulse (shape constraint).
    * **pmat** - pile-up penalty matrix built from an exponential decay model.
    * **fmat** - flat-top matrix encoding the derivative of the flat-top region.

    Parameters
    ----------
    wfs
        2-D array of aligned signal waveforms, shape ``(n_events, n_samples)``.
    length
        Filter length (samples); determines the output matrix dimensions.
    decay_const
        Exponential decay constant used to build the pile-up penalty matrix.
        Set to 0 to disable the pile-up term.
    ff
        Flat-top length (samples) used to construct the flat-top derivative
        matrix.

    Returns
    -------
    ref
        Mean reference pulse of length *n_samples*.
    rmat
        Reference signal matrix, shape ``(length, length)``.
    pmat
        Pile-up penalty matrix, shape ``(length, length)``.
    fmat
        Flat-top derivative matrix, shape ``(length, length)``.
    """
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
    decay = -np.arange(length) * decay_const if decay_const > 0 else np.zeros(length)
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
) -> tuple:
    """
    Synthesise a DPLMS optimal filter for a single channel.

    Solves the linear system ``(N + R + za*I + P + F) x = r`` for the filter
    coefficients *x*, where *N*, *R*, *P*, and *F* are the noise, reference,
    pile-up, and flat-top matrices respectively, *za* is a zero-area
    regularisation term, and *r* is the reference pulse window.

    Parameters
    ----------
    ref
        Mean reference pulse of length *size*.
    nmat
        Noise covariance matrix, shape ``(length, length)``.
    rmat
        Reference signal matrix, shape ``(length, length)``.
    za
        Zero-area regularisation coefficient.
    pmat
        Pile-up penalty matrix, shape ``(length, length)``.
    fmat
        Flat-top derivative matrix, shape ``(length, length)``.
    length
        Filter length (samples).
    size
        Full waveform buffer length (samples); used to centre the reference
        window.
    flip
        If True (default), return the time-reversed filter coefficients so
        that the filter can be applied via convolution.

    Returns
    -------
    x
        Filter coefficients of length *length*.
    y
        Response of the filter convolved with the reference pulse.
    refy
        Reference pulse window aligned to *y* for comparison.
    """
    # Reference slice
    flo = (size // 2) - (length // 2)
    fhi = (size // 2) + (length // 2)
    ref_window = ref[flo:fhi]

    # Construct full correlated matrix
    mat = nmat + rmat + za * np.ones([length, length]) + pmat + fmat

    # Solve system for filter coefficients
    x = np.linalg.solve(mat, ref_window).astype(np.float32)

    # Normalize via convolution with reference
    y = convolve(ref, np.flip(x), mode="valid")
    maxy = np.max(y)
    x /= maxy
    y /= maxy
    refy = ref[(size // 2) - (len(y) // 2) : (size // 2) + (len(y) // 2)]

    if flip:
        return np.flip(x), y, refy
    return x, y, refy


def filter_synthesis_corr(
    ref: np.array,
    nmat_corr: np.array,
    rmat: np.array,
    za: int,
    pmat: np.array,
    fmat: np.array,
    length: int,
    size: int,
    flip: bool = True,
) -> tuple:
    """
    Synthesise a DPLMS optimal filter exploiting cross-channel noise correlations.

    Extends :func:`filter_synthesis` to the multi-channel case by building
    block-extended versions of the signal-shape matrices and solving the
    combined linear system.  Only the primary-channel slice of the solution
    is returned.

    Parameters
    ----------
    ref
        Mean reference pulse for the primary channel, length *size*.
    nmat_corr
        Block noise covariance matrix from :func:`noise_matrix_corr`, shape
        ``(n_channels * length, n_channels * length)``.
    rmat
        Reference signal matrix for the primary channel, shape
        ``(length, length)``.
    za
        Zero-area regularisation coefficient.
    pmat
        Pile-up penalty matrix for the primary channel, shape
        ``(length, length)``.
    fmat
        Flat-top derivative matrix for the primary channel, shape
        ``(length, length)``.
    length
        Filter length per channel (samples).
    size
        Full waveform buffer length (samples).
    flip
        If True (default), return the time-reversed filter coefficients.

    Returns
    -------
    x
        Primary-channel filter coefficients of length *length*.
    y
        Response of the filter convolved with the (extended) reference pulse.
    refy
        Reference pulse window aligned to *y* for comparison.
    """
    # Extract number of correlated geds
    n_geds = nmat_corr.shape[0] // length

    # Reference slice
    flo = (size // 2) - (length // 2)
    fhi = (size // 2) + (length // 2)
    ref_window = ref[flo:fhi]

    # Extend reference
    ref_corr = np.zeros(n_geds * length, dtype=np.float32)
    ref_corr[:length] = ref_window

    # Build extended correlation matrices
    rmat_corr = np.zeros((n_geds * length, n_geds * length), dtype=np.float32)
    pmat_corr = np.zeros((n_geds * length, n_geds * length), dtype=np.float32)
    fmat_corr = np.zeros((n_geds * length, n_geds * length), dtype=np.float32)

    rmat_corr[:length, :length] = rmat
    pmat_corr[:length, :length] = pmat
    fmat_corr[:length, :length] = fmat

    # Construct full correlated matrix
    mat_corr = (
        nmat_corr
        + rmat_corr
        + pmat_corr
        + fmat_corr
        + za * np.ones((n_geds * length, n_geds * length), dtype=np.float32)
    )

    # Solve system for filter coefficients
    x = np.linalg.solve(mat_corr, ref_corr).astype(np.float32)

    # Normalize via convolution with reference
    y = convolve(ref_corr, np.flip(x), mode="valid")
    x /= np.max(y)

    if flip:
        return [np.flip(x[n * length : (n + 1) * length]) for n in range(n_geds)]
    return [x[n * length : (n + 1) * length] for n in range(n_geds)]
