"""
Module for special event level routines for SiPMs

functions must take as the first 8 args in order:
- path to the hit file
- path to the dsp <file
- path to the tcm file
- hit LH5 root group
- dsp LH5 root group
- tcm LH5 root group
- pattern to cast table names to tcm channel ids
- list of channels processed
additional parameters are free to the user and need to be defined in the JSON
"""


import awkward as ak
import numpy as np
from lgdo import Array, VectorOfVectors
from lgdo.lh5 import LH5Store

from pygama.evt import utils


# get an 1D akward array from 0 to 2D array
# casted by minimum of a 2D array
def cast_trigger(
    trgr,
    tdefault: float,
    length: int = None,
) -> ak.Array:
    if isinstance(trgr, Array):
        return ak.fill_none(ak.nan_to_none(trgr.view_as("ak")), tdefault)

    elif isinstance(trgr, (VectorOfVectors)):
        return ak.fill_none(
            ak.min(ak.fill_none(trgr.view_as("ak"), tdefault), axis=-1), tdefault
        )

    elif isinstance(trgr, (ak.Array, ak.highlevel.Array)):
        if trgr.ndim == 1:
            return ak.fill_none(ak.nan_to_none(trgr), tdefault)
        elif trgr.ndim == 2:
            return ak.fill_none(
                ak.min(ak.fill_none(ak.nan_to_none(trgr), tdefault), axis=-1), tdefault
            )
        else:
            raise ValueError(f"Too many dimensions: {trgr.ndim}")
    elif isinstance(trgr, (float, int)) and isinstance(length, int):
        return ak.Array([trgr] * length)
    else:
        raise ValueError(f"Can't deal with t0 of type {type(trgr)}")


# get SiPM coincidence window mask
def get_spm_mask(
    lim: float, trgr: ak.Array, tmin: float, tmax: float, pe: ak.Array, times: ak.Array
) -> ak.Array:
    if trgr.ndim != 1:
        raise ValueError("trigger array muse be 1 dimensional!")
    if (len(trgr) != len(pe)) or (len(trgr) != len(times)):
        raise ValueError(
            f"All arrays must have same dimension across first axis len(pe)={len(pe)}, len(times)={len(times)}, len(trgr)={len(trgr)}"
        )

    tmi = trgr - tmin
    tma = trgr + tmax

    mask = (
        ((times * 16.0) < tma[:, None]) & ((times * 16.0) > tmi[:, None]) & (pe > lim)
    )
    return mask


# get LAr indices according to mask per event over all channels
# mode 0 -> return pulse indices
# mode 1 -> return tcm indices
# mode 2 -> return rawids
# mode 3 -> return tcm_idx
def get_masked_tcm_idx(
    f_hit,
    f_dsp,
    f_tcm,
    hit_group,
    dsp_group,
    tcm_group,
    tcm_id_table_pattern,
    chs,
    lim,
    trgr,
    tdefault,
    tmin,
    tmax,
    mode=0,
) -> VectorOfVectors:
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read(f"/{tcm_group}/array_id", f_tcm)[0].view_as("np")
    idx = store.read(f"/{tcm_group}/array_idx", f_tcm)[0].view_as("np")

    arr_lst = []

    if isinstance(trgr, (float, int)):
        tge = cast_trigger(trgr, tdefault, length=np.max(idx) + 1)
    else:
        tge = cast_trigger(trgr, tdefault, length=None)

    for ch in chs:
        idx_ch = idx[ids == utils.get_tcm_id_by_pattern(tcm_id_table_pattern, ch)]

        pe = store.read(f"{ch}/{hit_group}/energy_in_pe", f_hit, idx=idx_ch)[0].view_as(
            "np"
        )
        tmp = np.full((np.max(idx) + 1, len(pe[0])), np.nan)
        tmp[idx_ch] = pe
        pe = ak.drop_none(ak.nan_to_none(ak.Array(tmp)))

        # times are in sample units
        times = store.read(f"{ch}/{hit_group}/trigger_pos", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        tmp = np.full((np.max(idx) + 1, len(times[0])), np.nan)
        tmp[idx_ch] = times
        times = ak.drop_none(ak.nan_to_none(ak.Array(tmp)))

        mask = get_spm_mask(lim, tge, tmin, tmax, pe, times)

        if mode == 0:
            out_idx = ak.local_index(mask)[mask]

        elif mode == 1:
            out_idx = np.full((np.max(idx) + 1), np.nan)
            out_idx[idx_ch] = np.where(
                ids == utils.get_tcm_id_by_pattern(tcm_id_table_pattern, ch)
            )[0]
            out_idx = ak.drop_none(ak.nan_to_none(ak.Array(out_idx)[:, None]))
            out_idx = out_idx[mask[mask] - 1]

        elif mode == 2:
            out_idx = ak.Array(
                [utils.get_tcm_id_by_pattern(tcm_id_table_pattern, ch)] * len(mask)
            )
            out_idx = out_idx[:, None][mask[mask] - 1]

        elif mode == 3:
            out_idx = np.full((np.max(idx) + 1), np.nan)
            out_idx[idx_ch] = idx_ch
            out_idx = ak.drop_none(ak.nan_to_none(ak.Array(out_idx)[:, None]))
            out_idx = out_idx[mask[mask] - 1]

        else:
            raise ValueError("Unknown mode")

        arr_lst.append(out_idx)

    return VectorOfVectors(array=ak.concatenate(arr_lst, axis=-1))


def get_spm_ene_or_maj(
    f_hit,
    f_tcm,
    hit_group,
    tcm_group,
    tcm_id_table_pattern,
    chs,
    lim,
    trgr,
    tdefault,
    tmin,
    tmax,
    mode,
):
    if mode not in ["energy_hc", "energy_dplms", "majority_hc", "majority_dplms"]:
        raise ValueError("Unknown mode")

    # load TCM data to define an event
    store = LH5Store()
    ids = store.read(f"/{tcm_group}/array_id", f_tcm)[0].view_as("np")
    idx = store.read(f"/{tcm_group}/array_idx", f_tcm)[0].view_as("np")
    out = np.zeros(np.max(idx) + 1)

    if isinstance(trgr, (float, int)):
        tge = cast_trigger(trgr, tdefault, length=np.max(idx) + 1)
    else:
        tge = cast_trigger(trgr, tdefault, length=None)

    for ch in chs:
        idx_ch = idx[ids == utils.get_tcm_id_by_pattern(tcm_id_table_pattern, ch)]

        if mode in ["energy_dplms", "majority_dplms"]:
            pe = ak.drop_none(
                ak.nan_to_none(
                    store.read(
                        f"{ch}/{hit_group}/energy_in_pe_dplms", f_hit, idx=idx_ch
                    )[0].view_as("ak")
                )
            )

            # times are in sample units
            times = ak.drop_none(
                ak.nan_to_none(
                    store.read(
                        f"{ch}/{hit_group}/trigger_pos_dplms", f_hit, idx=idx_ch
                    )[0].view_as("ak")
                )
            )

        else:
            pe = ak.drop_none(
                ak.nan_to_none(
                    store.read(f"{ch}/{hit_group}/energy_in_pe", f_hit, idx=idx_ch)[
                        0
                    ].view_as("ak")
                )
            )

            # times are in sample units
            times = ak.drop_none(
                ak.nan_to_none(
                    store.read(f"{ch}/{hit_group}/trigger_pos", f_hit, idx=idx_ch)[
                        0
                    ].view_as("ak")
                )
            )

        mask = get_spm_mask(lim, tge[idx_ch], tmin, tmax, pe, times)
        pe = pe[mask]

        if mode in ["energy_hc", "energy_dplms"]:
            out[idx_ch] = out[idx_ch] + ak.to_numpy(ak.nansum(pe, axis=-1))

        else:
            out[idx_ch] = out[idx_ch] + ak.to_numpy(
                ak.where(ak.nansum(pe, axis=-1) > lim, 1, 0)
            )

    return Array(nda=out)


# get LAr energy per event over all channels
def get_energy(
    f_hit,
    f_dsp,
    f_tcm,
    hit_group,
    dsp_group,
    tcm_group,
    tcm_id_table_pattern,
    chs,
    lim,
    trgr,
    tdefault,
    tmin,
    tmax,
) -> Array:
    return get_spm_ene_or_maj(
        f_hit,
        f_tcm,
        hit_group,
        tcm_group,
        tcm_id_table_pattern,
        chs,
        lim,
        trgr,
        tdefault,
        tmin,
        tmax,
        "energy_hc",
    )


# get LAr majority per event over all channels
def get_majority(
    f_hit,
    f_dsp,
    f_tcm,
    hit_group,
    dsp_group,
    tcm_group,
    tcm_id_table_pattern,
    chs,
    lim,
    trgr,
    tdefault,
    tmin,
    tmax,
) -> Array:
    return get_spm_ene_or_maj(
        f_hit,
        f_tcm,
        hit_group,
        tcm_group,
        tcm_id_table_pattern,
        chs,
        lim,
        trgr,
        tdefault,
        tmin,
        tmax,
        "majority_hc",
    )


# get LAr energy per event over all channels
def get_energy_dplms(
    f_hit,
    f_dsp,
    f_tcm,
    hit_group,
    dsp_group,
    tcm_group,
    tcm_id_table_pattern,
    chs,
    lim,
    trgr,
    tdefault,
    tmin,
    tmax,
) -> Array:
    return get_spm_ene_or_maj(
        f_hit,
        f_tcm,
        hit_group,
        tcm_group,
        tcm_id_table_pattern,
        chs,
        lim,
        trgr,
        tdefault,
        tmin,
        tmax,
        "energy_dplms",
    )


# get LAr majority per event over all channels
def get_majority_dplms(
    f_hit,
    f_dsp,
    f_tcm,
    hit_group,
    dsp_group,
    tcm_group,
    tcm_id_table_pattern,
    chs,
    lim,
    trgr,
    tdefault,
    tmin,
    tmax,
) -> Array:
    return get_spm_ene_or_maj(
        f_hit,
        f_tcm,
        hit_group,
        tcm_group,
        tcm_id_table_pattern,
        chs,
        lim,
        trgr,
        tdefault,
        tmin,
        tmax,
        "majority_dplms",
    )


# Calculate the ETC in different trailing modes:
# trail = 0: Singlet window = [tge,tge+swin]
# trail = 1: Singlet window = [t_first_lar_pulse, t_first_lar_pulse+ swin]
# trail = 2: Like trail = 1, but t_first_lar_pulse <= tge is ensured
# min_first_pls_ene sets the minimum energy of the first pulse (only used in trail > 0)
# max_per_channel, maximum number of pes a channel is allowed to have, if above it gets excluded
def get_etc(
    f_hit,
    f_dsp,
    f_tcm,
    hit_group,
    dsp_group,
    tcm_group,
    tcm_id_table_pattern,
    chs,
    lim,
    trgr,
    tdefault,
    tmin,
    tmax,
    swin,
    trail,
    min_first_pls_ene,
    max_per_channel,
) -> Array:
    # load TCM data to define an event
    store = LH5Store()
    ids = store.read(f"/{tcm_group}/array_id", f_tcm)[0].view_as("np")
    idx = store.read(f"/{tcm_group}/array_idx", f_tcm)[0].view_as("np")
    pe_lst = []
    time_lst = []

    if isinstance(trgr, (float, int)):
        tge = cast_trigger(trgr, tdefault, length=np.max(idx) + 1)
    else:
        tge = cast_trigger(trgr, tdefault, length=None)

    for ch in chs:
        idx_ch = idx[ids == utils.get_tcm_id_by_pattern(tcm_id_table_pattern, ch)]

        pe = store.read(f"{ch}/{hit_group}/energy_in_pe", f_hit, idx=idx_ch)[0].view_as(
            "np"
        )
        tmp = np.full((np.max(idx) + 1, len(pe[0])), np.nan)
        tmp[idx_ch] = pe
        pe = ak.drop_none(ak.nan_to_none(ak.Array(tmp)))

        # times are in sample units
        times = store.read(f"{ch}/{hit_group}/trigger_pos", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        tmp = np.full((np.max(idx) + 1, len(times[0])), np.nan)
        tmp[idx_ch] = times
        times = ak.drop_none(ak.nan_to_none(ak.Array(tmp)))

        mask = get_spm_mask(lim, tge, tmin, tmax, pe, times)

        pe = pe[mask]

        # max pe mask
        max_pe_mask = ak.nansum(pe, axis=-1) < max_per_channel
        pe = ak.drop_none(
            ak.nan_to_none(ak.where(max_pe_mask, pe, ak.Array([[np.nan]])))
        )
        pe_lst.append(pe)

        times = times[mask] * 16
        times = ak.drop_none(
            ak.nan_to_none(ak.where(max_pe_mask, times, ak.Array([[np.nan]])))
        )
        time_lst.append(times)

    pe_all = ak.concatenate(pe_lst, axis=-1)
    time_all = ak.concatenate(time_lst, axis=-1)

    if trail > 0:
        t1d = ak.min(time_all[pe_all > min_first_pls_ene], axis=-1)

        if trail == 2:
            t1d = ak.where(t1d > tge, tge, t1d)

        mask_total = time_all > t1d
        mask_singlet = (time_all > t1d) & (time_all < t1d + swin)

    else:
        mask_total = time_all > tge
        mask_singlet = (time_all > tge) & (time_all < tge + swin)

    pe_singlet = ak.to_numpy(
        ak.fill_none(ak.nansum(pe_all[mask_singlet], axis=-1), 0), allow_missing=False
    )
    pe_total = ak.to_numpy(
        ak.fill_none(ak.nansum(pe_all[mask_total], axis=-1), 0), allow_missing=False
    )
    etc = np.divide(
        pe_singlet, pe_total, out=np.full_like(pe_total, np.nan), where=pe_total != 0
    )

    return Array(nda=etc)


# returns relative time shift of the first LAr pulse relative to the Ge trigger
def get_time_shift(
    f_hit,
    f_dsp,
    f_tcm,
    hit_group,
    dsp_group,
    tcm_group,
    tcm_id_table_pattern,
    chs,
    lim,
    trgr,
    tdefault,
    tmin,
    tmax,
) -> Array:
    store = LH5Store()
    # load TCM data to define an event
    ids = store.read(f"/{tcm_group}/array_id", f_tcm)[0].view_as("np")
    idx = store.read(f"/{tcm_group}/array_idx", f_tcm)[0].view_as("np")
    time_all = ak.Array([[] for x in range(np.max(idx) + 1)])

    if isinstance(trgr, (float, int)):
        tge = cast_trigger(trgr, tdefault, length=np.max(idx) + 1)
    else:
        tge = cast_trigger(trgr, tdefault, length=None)

    for ch in chs:
        idx_ch = idx[ids == utils.get_tcm_id_by_pattern(tcm_id_table_pattern, ch)]

        pe = store.read(f"{ch}/{hit_group}/energy_in_pe", f_hit, idx=idx_ch)[0].view_as(
            "np"
        )
        tmp = np.full((np.max(idx) + 1, len(pe[0])), np.nan)
        tmp[idx_ch] = pe
        pe = ak.drop_none(ak.nan_to_none(ak.Array(tmp)))

        # times are in sample units
        times = store.read(f"{ch}/{hit_group}/trigger_pos", f_hit, idx=idx_ch)[
            0
        ].view_as("np")
        tmp = np.full((np.max(idx) + 1, len(times[0])), np.nan)
        tmp[idx_ch] = times
        times = ak.drop_none(ak.nan_to_none(ak.Array(tmp)))

        mask = get_spm_mask(lim, tge, tmin, tmax, pe, times)

        # apply mask and convert sample units to ns
        times = times[mask] * 16

        time_all = ak.concatenate((time_all, times), axis=-1)

    out = ak.min(time_all, axis=-1)

    # Convert to 1D numpy array
    out = ak.to_numpy(ak.fill_none(out, np.inf), allow_missing=False)
    tge = ak.to_numpy(tge, allow_missing=False)

    return Array(out - tge)
