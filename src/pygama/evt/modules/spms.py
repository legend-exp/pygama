from __future__ import annotations

import awkward as ak
import numpy as np
from lgdo import lh5, types

from .. import utils


def gather_pulse_data(
    datainfo,
    tcm,
    channels,
    *,
    observable,
    pulse_mask=None,
    a_thr_pe=None,
    t_loc_ns=None,
    dt_range_ns=None,
    t_loc_default_ns=None,
    drop_empty=True,
) -> types.VectorOfVectors:
    # parse observables string. default to hit tier
    p = observable.split(".")
    tier = p[0] if len(p) > 1 else "hit"
    column = p[1] if len(p) > 1 else p[0]

    tierinfo = datainfo._asdict()[tier]

    # loop over selected channels and load hit data
    concatme = []
    for channel in channels:
        table_id = utils.get_tcm_id_by_pattern(tierinfo.table_fmt, channel)

        # determine list of indices found in the TCM that we want to load for channel
        idx = tcm.idx[tcm.id == table_id]

        # read the data in
        lgdo_obj = lh5.read(
            f"/{channel}/{tierinfo.group}/{column}", tierinfo.file, idx=idx
        )
        data = lgdo_obj.view_as(library="ak")

        # remove nans (this happens when SiPM data is stored as ArrayOfEqualSizedArrays)
        data = ak.drop_none(ak.nan_to_none(data))

        # increase the dimensionality by one (events)
        data = ak.unflatten(data, np.full(data.layout.length, 1, dtype="uint8"))

        concatme.append(data)

    # concatenate along the event axes (i.e. gather channels together)
    data = ak.concatenate(concatme, axis=1)

    # check if user wants to apply a mask
    if pulse_mask is None and any(
        [kwarg is not None for kwarg in (a_thr_pe, t_loc_ns, dt_range_ns)]
    ):
        # generate the time/amplitude mask from parameters
        pulse_mask = make_pulse_data_mask(
            datainfo,
            tcm,
            channels,
            a_thr_pe=a_thr_pe,
            t_loc_ns=t_loc_ns,
            dt_range_ns=dt_range_ns,
            t_loc_default_ns=t_loc_default_ns,
        )

    if pulse_mask is not None:
        if not isinstance(pulse_mask, ak.Array):
            pulse_mask = pulse_mask.view_as("ak")

        # apply the mask
        data = data[pulse_mask]

    # remove empty arrays = channels with no pulses
    if drop_empty:
        data = data[ak.count(data, axis=-1) > 0]

    return types.VectorOfVectors(data, attrs=utils.copy_lgdo_attrs(lgdo_obj))


def gather_tcm_id_data(
    datainfo,
    tcm,
    channels,
    *,
    pulse_mask=None,
    a_thr_pe=None,
    t_loc_ns=None,
    dt_range_ns=None,
    t_loc_default_ns=None,
    drop_empty=True,
) -> types.VectorOfVectors:
    # loop over selected channels and load hit data
    table_ids = [
        utils.get_tcm_id_by_pattern(datainfo.hit.table_fmt, channel)
        for channel in channels
    ]

    data = ak.Array(
        np.full(
            shape=(len(tcm.cumulative_length), len(table_ids)), fill_value=table_ids
        )
    )

    # check if user wants to apply a mask
    if drop_empty:
        if pulse_mask is None and any(
            [kwarg is not None for kwarg in (a_thr_pe, t_loc_ns, dt_range_ns)]
        ):
            # generate the time/amplitude mask from parameters
            pulse_mask = make_pulse_data_mask(
                datainfo,
                tcm,
                channels,
                a_thr_pe=a_thr_pe,
                t_loc_ns=t_loc_ns,
                dt_range_ns=dt_range_ns,
                t_loc_default_ns=t_loc_default_ns,
            )

        if pulse_mask is None:
            msg = "need a valid pulse mask in order to drop table_ids with no pulses"
            raise ValueError(msg)

        if not isinstance(pulse_mask, ak.Array):
            pulse_mask = pulse_mask.view_as("ak")

        if pulse_mask.ndim != 3:
            msg = "pulse_mask must be 3D"
            raise ValueError(msg)

        # convert the 3D mask to a 2D mask (can be used to filter table_ids)
        ch_mask = ak.sum(pulse_mask, axis=-1) > 0

        # apply the mask
        data = data[ch_mask]

    return types.VectorOfVectors(data)


# NOTE: the mask never gets the empty arrays removed
def make_pulse_data_mask(
    datainfo,
    tcm,
    channels,
    *,
    a_thr_pe=None,
    t_loc_ns=None,
    dt_range_ns=None,
    t_loc_default_ns=None,
) -> types.VectorOfVectors:
    # get the t0 of each single pulse
    pulse_t0 = gather_pulse_data(
        datainfo,
        tcm,
        channels,
        observable="hit.trigger_pos",
        drop_empty=False,
    )

    # HACK: handle units
    # HACK: remove me once units are fixed in the dsp tier
    if "units" in pulse_t0.attrs and pulse_t0.attrs["units"] == "ns":
        pulse_t0_ns = pulse_t0.view_as("ak")
    else:
        pulse_t0_ns = pulse_t0.view_as("ak") * 16

    pulse_amp = gather_pulse_data(
        datainfo,
        tcm,
        channels,
        observable="hit.energy_in_pe",
        drop_empty=False,
    ).view_as("ak")

    # (HPGe) trigger position can vary among events!
    if isinstance(t_loc_ns, types.Array):
        t_loc_ns = t_loc_ns.view_as("ak")

    if isinstance(t_loc_ns, ak.Array):
        if t_loc_ns.ndim != 1:
            msg = "t_loc_ns must be 0- or 1-dimensional"
            raise ValueError(msg)

        # NOTE: the assumption is that t0 is np.nan when missing -> replace
        # with default value
        t_loc_ns = ak.fill_none(ak.nan_to_none(t_loc_ns), t_loc_default_ns)

    mask = pulse_t0_ns == pulse_t0_ns

    if a_thr_pe is not None:
        mask = mask & (pulse_amp > a_thr_pe)

    if t_loc_ns is not None and dt_range_ns is not None:
        if not isinstance(dt_range_ns, (tuple, list)):
            msg = "dt_range_ns must be a tuple"
            raise ValueError(msg)

        mask = mask & (
            (pulse_t0_ns < (t_loc_ns + dt_range_ns[1]))
            & (pulse_t0_ns > (t_loc_ns + dt_range_ns[0]))
        )

    return types.VectorOfVectors(mask)
