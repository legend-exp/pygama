"""
Module for special event level routines for SiPMs

functions must take as the first 3 args in order:
- path to the hit file
- path to the dsp file
- path to the tcm file
- list of channels processed
additional parameters are free to the user and need to be defined in the JSON
"""

import warnings

import numpy as np
from lgdo import Array, VectorOfVectors
from lgdo.lh5 import LH5Store


# get LAr energy per event over all channels
def get_energy(f_hit, f_dsp, f_tcm, chs, lim, trgr, tdefault, tmin, tmax) -> Array:
    trig = trgr
    if isinstance(trgr, VectorOfVectors):
        trig = trig.to_aoesa().nda
    elif isinstance(trgr, Array):
        trig = trig.nda
    if isinstance(trig, np.ndarray) and trig.ndim == 2:
        trig = np.where(np.isnan(trig).all(axis=1)[:, None], tdefault, trig)
        trig = np.nanmin(trig, axis=1)

    elif isinstance(trig, np.ndarray) and trig.ndim == 1:
        trig = np.where(np.isnan(trig), tdefault, trig)
    else:
        raise ValueError(f"Can't deal with t0 of type {type(trgr)}")
    tmi = trig - tmin
    tma = trig + tmax
    sum = np.zeros(len(trig))
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")

    for ch in chs:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]
        energy_in_pe = store.read(f"{ch}/hit/energy_in_pe", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        trigger_pos = store.read(f"{ch}/hit/trigger_pos", f_hit, idx=idx_ch)[0].view_as(
            "np"
        )
        mask = (
            (trigger_pos < tma[:, None] / 16)
            & (trigger_pos > tmi[:, None] / 16)
            & (energy_in_pe > lim)
        )
        pes = energy_in_pe
        pes = np.where(np.isnan(pes), 0, pes)
        pes = np.where(mask, pes, 0)
        chsum = np.nansum(pes, axis=1)
        sum[idx_ch] = sum[idx_ch] + chsum
    return Array(nda=sum)


# get LAr majority per event over all channels
def get_majority(f_hit, f_dsp, f_tcm, chs, lim, trgr, tdefault, tmin, tmax) -> Array:
    trig = trgr
    if isinstance(trgr, VectorOfVectors):
        trig = trig.to_aoesa().nda
    elif isinstance(trgr, Array):
        trig = trig.nda
    if isinstance(trig, np.ndarray) and trig.ndim == 2:
        trig = np.where(np.isnan(trig).all(axis=1)[:, None], tdefault, trig)
        trig = np.nanmin(trig, axis=1)

    elif isinstance(trig, np.ndarray) and trig.ndim == 1:
        trig = np.where(np.isnan(trig), tdefault, trig)
    else:
        raise ValueError(f"Can't deal with t0 of type {type(trgr)}")
    tmi = trig - tmin
    tma = trig + tmax
    maj = np.zeros(len(trig))
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    for ch in chs:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]
        energy_in_pe = store.read(f"{ch}/hit/energy_in_pe", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        trigger_pos = store.read(f"{ch}/hit/trigger_pos", f_hit, idx=idx_ch)[0].view_as(
            "np"
        )
        mask = (
            (trigger_pos < tma[:, None] / 16)
            & (trigger_pos > tmi[:, None] / 16)
            & (energy_in_pe > lim)
        )
        pes = energy_in_pe
        pes = np.where(np.isnan(pes), 0, pes)
        pes = np.where(mask, pes, 0)
        chsum = np.nansum(pes, axis=1)
        chmaj = np.where(chsum > lim, 1, 0)
        maj[idx_ch] = maj[idx_ch] + chmaj
    return Array(nda=maj)


# get LAr energy per event over all channels
def get_energy_dplms(
    f_hit, f_dsp, f_tcm, chs, lim, trgr, tdefault, tmin, tmax
) -> Array:
    trig = trgr
    if isinstance(trgr, VectorOfVectors):
        trig = trig.to_aoesa().nda
    elif isinstance(trgr, Array):
        trig = trig.nda
    if isinstance(trig, np.ndarray) and trig.ndim == 2:
        trig = np.where(np.isnan(trig).all(axis=1)[:, None], tdefault, trig)
        trig = np.nanmin(trig, axis=1)

    elif isinstance(trig, np.ndarray) and trig.ndim == 1:
        trig = np.where(np.isnan(trig), tdefault, trig)
    else:
        raise ValueError(f"Can't deal with t0 of type {type(trgr)}")
    tmi = trig - tmin
    tma = trig + tmax
    sum = np.zeros(len(trig))
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    for ch in chs:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]
        energy_in_pe_dplms = store.read(
            f"{ch}/hit/energy_in_pe_dplms", f_hit, idx=idx_ch
        )[0].view_as("np")
        trigger_pos_dplms = store.read(
            f"{ch}/hit/trigger_pos_dplms", f_hit, idx=idx_ch
        )[0].view_as("np")
        mask = (
            (trigger_pos_dplms < tma[:, None] / 16)
            & (trigger_pos_dplms > tmi[:, None] / 16)
            & (energy_in_pe_dplms > lim)
        )
        pes = energy_in_pe_dplms
        pes = np.where(np.isnan(pes), 0, pes)
        pes = np.where(mask, pes, 0)
        chsum = np.nansum(pes, axis=1)
        sum[idx_ch] = sum[idx_ch] + chsum
    return Array(nda=sum)


# get LAr majority per event over all channels
def get_majority_dplms(
    f_hit, f_dsp, f_tcm, chs, lim, trgr, tdefault, tmin, tmax
) -> Array:
    trig = trgr
    if isinstance(trgr, VectorOfVectors):
        trig = trig.to_aoesa().nda
    elif isinstance(trgr, Array):
        trig = trig.nda
    if isinstance(trig, np.ndarray) and trig.ndim == 2:
        trig = np.where(np.isnan(trig).all(axis=1)[:, None], tdefault, trig)
        trig = np.nanmin(trig, axis=1)

    elif isinstance(trig, np.ndarray) and trig.ndim == 1:
        trig = np.where(np.isnan(trig), tdefault, trig)
    else:
        raise ValueError(f"Can't deal with t0 of type {type(trgr)}")
    tmi = trig - tmin
    tma = trig + tmax
    maj = np.zeros(len(trig))
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    for ch in chs:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]
        energy_in_pe_dplms = store.read(
            f"{ch}/hit/energy_in_pe_dplms", f_hit, idx=idx_ch
        )[0].view_as("np")
        trigger_pos_dplms = store.read(
            f"{ch}/hit/trigger_pos_dplms", f_hit, idx=idx_ch
        )[0].view_as("np")
        mask = (
            (trigger_pos_dplms < tma[:, None] / 16)
            & (trigger_pos_dplms > tmi[:, None] / 16)
            & (energy_in_pe_dplms > lim)
        )
        pes = energy_in_pe_dplms
        pes = np.where(np.isnan(pes), 0, pes)
        pes = np.where(mask, pes, 0)
        chsum = np.nansum(pes, axis=1)
        chmaj = np.where(chsum > lim, 1, 0)
        maj[idx_ch] = maj[idx_ch] + chmaj
    return Array(nda=maj)


def get_etc(
    f_hit, f_dsp, f_tcm, chs, lim, trgr, tdefault, tmin, tmax, swin, trail
) -> Array:
    # ignore stupid numpy warnings
    warnings.filterwarnings("ignore", r"All-NaN slice encountered")
    warnings.filterwarnings("ignore", r"invalid value encountered in true_divide")
    warnings.filterwarnings("ignore", r"invalid value encountered in divide")

    store = LH5Store()
    energy_in_pe, _ = store.read(f"{chs[0]}/hit/energy_in_pe", f_hit)

    peshape = energy_in_pe.view_as("np").shape
    # 1D = channel, 2D = event num, 3D = array per event
    pes = np.zeros([len(chs), peshape[0], peshape[1]])
    times = np.zeros([len(chs), peshape[0], peshape[1]])

    tge = trgr
    if isinstance(trgr, VectorOfVectors):
        tge = tge.to_aoesa().nda
    elif isinstance(trgr, Array):
        tge = tge.nda
    if isinstance(tge, np.ndarray) and tge.ndim == 2:
        tge = np.where(np.isnan(tge).all(axis=1)[:, None], tdefault, tge)
        tge = np.nanmin(tge, axis=1)

    elif isinstance(tge, np.ndarray) and tge.ndim == 1:
        tge = np.where(np.isnan(tge), tdefault, tge)
    else:
        raise ValueError(f"Can't deal with t0 of type {type(trgr)}")

    tmi = tge - tmin
    tma = tge + tmax

    # load TCM data to define an event
    store = LH5Store()
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    for i in range(len(chs)):
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(chs[i][2:])]
        energy_in_pe = store.read(f"{chs[i]}/hit/energy_in_pe", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        trigger_pos = store.read(f"{chs[i]}/hit/trigger_pos", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        mask = (
            (trigger_pos < tma[:, None] / 16)
            & (trigger_pos > tmi[:, None] / 16)
            & (energy_in_pe > lim)
        )
        pe = energy_in_pe
        time = trigger_pos * 16

        pe = np.where(mask, pe, np.nan)
        time = np.where(mask, time, np.nan)

        pes[i][idx_ch] = pe
        times[i][idx_ch] = time

    outi = None
    if trail > 0:
        t1d = np.nanmin(times, axis=(0, 2))
        if trail == 2:
            t1d[t1d > tge] = tge[t1d > tge]
        tt = t1d[:, None]
        outi = np.where(
            np.nansum(np.where((times >= tt), pes, 0), axis=(0, 2)) > 0,
            np.nansum(
                np.where((times >= tt) & (times < tt + swin), pes, 0), axis=(0, 2)
            )
            / np.nansum(np.where((times >= tt), pes, 0), axis=(0, 2)),
            np.nansum(np.where((times >= tt), pes, 0), axis=(0, 2)),
        )
        return Array(nda=outi)

    else:
        outi = np.where(
            np.nansum(pes, axis=(0, 2)) > 0,
            np.nansum(
                np.where(
                    (times >= tge[:, None]) & (times <= tge[:, None] + swin), pes, 0
                ),
                axis=(0, 2),
            )
            / np.nansum(np.where((times >= tge[:, None]), pes, 0), axis=(0, 2)),
            np.nansum(pes, axis=(0, 2)),
        )
        return Array(nda=outi)


def get_time_shift(f_hit, f_dsp, f_tcm, chs, lim, trgr, tdefault, tmin, tmax) -> Array:
    store = LH5Store()
    energy_in_pe, _ = store.read(
        f"{chs[0]}/hit/energy_in_pe",
        f_hit,
    )
    peshape = energy_in_pe.view_as("np").shape
    times = np.zeros([len(chs), peshape[0], peshape[1]])

    tge = trgr
    if isinstance(trgr, VectorOfVectors):
        tge = tge.to_aoesa().nda
    elif isinstance(trgr, Array):
        tge = tge.nda
    if isinstance(tge, np.ndarray) and tge.ndim == 2:
        tge = np.where(np.isnan(tge).all(axis=1)[:, None], tdefault, tge)
        tge = np.nanmin(tge, axis=1)

    elif isinstance(tge, np.ndarray) and tge.ndim == 1:
        tge = np.where(np.isnan(tge), tdefault, tge)
    else:
        raise ValueError(f"Can't deal with t0 of type {type(trgr)}")

    tmi = tge - tmin
    tma = tge + tmax

    # load TCM data to define an event
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    for i in range(len(chs)):
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(chs[i][2:])]
        energy_in_pe = store.read(f"{chs[i]}/hit/energy_in_pe", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        trigger_pos = store.read(f"{chs[i]}/hit/trigger_pos", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        mask = (
            (trigger_pos < tma[:, None] / 16)
            & (trigger_pos > tmi[:, None] / 16)
            & (energy_in_pe > lim)
        )

        time = trigger_pos * 16
        time = np.where(mask, time, np.nan)
        times[i][idx_ch] = time

        t1d = np.nanmin(times, axis=(0, 2))

        return Array(t1d - tge)
