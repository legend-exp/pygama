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

import awkward as ak
import numpy as np
from lgdo import Array, VectorOfVectors
from lgdo.lh5 import LH5Store


# get SiPM coincidence window mask
def get_spm_mask(lim, trgr, tdefault, tmin, tmax, pe, times) -> np.ndarray:
    trig = trgr
    if isinstance(trgr, VectorOfVectors):
        trig = trig.to_aoesa().view_as("np")
    elif isinstance(trgr, Array):
        trig = trig.view_as("np")
    elif isinstance(trgr, ak.Array):
        if trgr.ndim == 1:
            trig = ak.to_numpy(trig)
        else:
            trig = ak.to_numpy(
                ak.fill_none(
                    ak.pad_none(trig, target=ak.max(ak.count(trig, axis=-1)), axis=-1),
                    np.nan,
                ),
                allow_missing=False,
            )
    if isinstance(trig, np.ndarray) and trig.ndim == 2:
        trig = np.where(np.isnan(trig).all(axis=1)[:, None], tdefault, trig)
        trig = np.nanmin(trig, axis=1)

    elif isinstance(trig, np.ndarray) and trig.ndim == 1:
        trig = np.where(np.isnan(trig), tdefault, trig)
    else:
        raise ValueError(f"Can't deal with t0 of type {type(trgr)}")

    tmi = trig - tmin
    tma = trig + tmax

    mask = (times < tma[:, None] / 16) & (times > tmi[:, None] / 16) & (pe > lim)
    return mask, trig


# get LAr indices according to mask per event over all channels
# mode 0 -> return pulse indices
# mode 1 -> return tcm indices
# mode 2 -> return rawids
# mode 3 -> return tcm_idx
def get_masked_tcm_idx(
    f_hit, f_dsp, f_tcm, chs, lim, trgr, tdefault, tmin, tmax, mode=0
) -> VectorOfVectors:
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")

    arr_lst = []
    for ch in chs:
        idx_ch = idx[ids == int(ch[2:])]
        energy_in_pe = store.read(f"{ch}/hit/energy_in_pe", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        trigger_pos = store.read(f"{ch}/hit/trigger_pos", f_hit, idx=idx_ch)[0].view_as(
            "np"
        )
        mask, _ = get_spm_mask(
            lim, trgr, tdefault, tmin, tmax, energy_in_pe, trigger_pos
        )

        if mode == 0:
            out_idx = np.repeat(
                np.arange(len(mask[0]))[:, None], repeats=len(mask), axis=1
            ).T
            out_idx = np.where(mask, out_idx, np.nan)
            out_idx = VectorOfVectors(
                flattened_data=out_idx.flatten()[~np.isnan(out_idx.flatten())],
                cumulative_length=np.cumsum(
                    np.count_nonzero(~np.isnan(out_idx), axis=1)
                ),
                dtype=int,
            ).view_as("ak", preserve_dtype=True)
        elif mode == 1:
            out_idx = np.where(mask, np.where(ids == int(ch[2:]))[0][:, None], np.nan)
            out_idx = VectorOfVectors(
                flattened_data=out_idx.flatten()[~np.isnan(out_idx.flatten())],
                cumulative_length=np.cumsum(
                    np.count_nonzero(~np.isnan(out_idx), axis=1)
                ),
                dtype=int,
            ).view_as("ak", preserve_dtype=True)
        elif mode == 2:
            out_idx = np.where(mask, int(ch[2:]), np.nan)
            out_idx = VectorOfVectors(
                flattened_data=out_idx.flatten()[~np.isnan(out_idx.flatten())],
                cumulative_length=np.cumsum(
                    np.count_nonzero(~np.isnan(out_idx), axis=1)
                ),
                dtype=int,
            ).view_as("ak", preserve_dtype=True)
        elif mode == 3:
            out_idx = np.where(mask, idx_ch[:, None], np.nan)
            out_idx = VectorOfVectors(
                flattened_data=out_idx.flatten()[~np.isnan(out_idx.flatten())],
                cumulative_length=np.cumsum(
                    np.count_nonzero(~np.isnan(out_idx), axis=1)
                ),
                dtype=int,
            ).view_as("ak", preserve_dtype=True)
        else:
            raise ValueError("Unknown mode")

        arr_lst.append(out_idx)

    return VectorOfVectors(array=ak.concatenate(arr_lst, axis=-1))


# get LAr energy per event over all channels
def get_energy(f_hit, f_dsp, f_tcm, chs, lim, trgr, tdefault, tmin, tmax) -> Array:
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    sum = np.zeros(np.max(idx) + 1)
    for ch in chs:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]
        energy_in_pe = store.read(f"{ch}/hit/energy_in_pe", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        trigger_pos = store.read(f"{ch}/hit/trigger_pos", f_hit, idx=idx_ch)[0].view_as(
            "np"
        )
        mask, _ = get_spm_mask(
            lim, trgr, tdefault, tmin, tmax, energy_in_pe, trigger_pos
        )
        pes = energy_in_pe
        pes = np.where(np.isnan(pes), 0, pes)
        pes = np.where(mask, pes, 0)
        chsum = np.nansum(pes, axis=1)
        sum[idx_ch] = sum[idx_ch] + chsum
    return Array(nda=sum)


# get LAr majority per event over all channels
def get_majority(f_hit, f_dsp, f_tcm, chs, lim, trgr, tdefault, tmin, tmax) -> Array:
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    maj = np.zeros(np.max(idx) + 1)
    for ch in chs:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]
        energy_in_pe = store.read(f"{ch}/hit/energy_in_pe", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        trigger_pos = store.read(f"{ch}/hit/trigger_pos", f_hit, idx=idx_ch)[0].view_as(
            "np"
        )
        mask, _ = get_spm_mask(
            lim, trgr, tdefault, tmin, tmax, energy_in_pe, trigger_pos
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
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    sum = np.zeros(np.max(idx) + 1)
    for ch in chs:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]
        energy_in_pe_dplms = store.read(
            f"{ch}/hit/energy_in_pe_dplms", f_hit, idx=idx_ch
        )[0].view_as("np")
        trigger_pos_dplms = store.read(
            f"{ch}/hit/trigger_pos_dplms", f_hit, idx=idx_ch
        )[0].view_as("np")
        mask, _ = get_spm_mask(
            lim, trgr, tdefault, tmin, tmax, energy_in_pe_dplms, trigger_pos_dplms
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
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    maj = np.zeros(np.max(idx) + 1)
    for ch in chs:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]
        energy_in_pe_dplms = store.read(
            f"{ch}/hit/energy_in_pe_dplms", f_hit, idx=idx_ch
        )[0].view_as("np")
        trigger_pos_dplms = store.read(
            f"{ch}/hit/trigger_pos_dplms", f_hit, idx=idx_ch
        )[0].view_as("np")
        mask, _ = get_spm_mask(
            lim, trgr, tdefault, tmin, tmax, energy_in_pe_dplms, trigger_pos_dplms
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
        mask, tge = get_spm_mask(
            lim, trgr, tdefault, tmin, tmax, energy_in_pe, trigger_pos
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
    # load TCM data to define an event
    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    spm_tmin = np.full(np.max(idx) + 1, np.inf)
    for i in range(len(chs)):
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(chs[i][2:])]
        energy_in_pe = store.read(f"{chs[i]}/hit/energy_in_pe", f_hit, idx=idx_ch)[
            0
        ].view_as("ak")
        trigger_pos = store.read(f"{chs[i]}/hit/trigger_pos", f_hit, idx=idx_ch)[
            0
        ].view_as("ak")
        mask, tge = get_spm_mask(
            lim, trgr, tdefault, tmin, tmax, energy_in_pe, trigger_pos
        )

        time = trigger_pos * 16
        time = ak.min(ak.nan_to_none(time[mask]), axis=-1)
        if not time:
            return Array(nda=np.zeros(len(spm_tmin)))
        time = ak.fill_none(time, tdefault)
        if not time:
            time = ak.to_numpy(time, allow_missing=False)
            spm_tmin = np.where(time < spm_tmin, time, spm_tmin)

    return Array(spm_tmin - tge)
