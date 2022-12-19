from __future__ import annotations

import copy
import logging
from copy import deepcopy

import numpy as np

import pygama.lgdo as lgdo
from pygama.dsp.processing_chain import build_processing_chain as bpc
from pygama.lgdo import Array, ArrayOfEqualSizedArrays, Struct, Table

log = logging.getLogger(__name__)


def data_trimmer(lgdo_table: Table | Struct, dsp_dict: dict, group: str = None) -> None:
    """
    Takes in a :class:`.RawBuffer` that contains waveforms, performs user specified
    DSP on the table, and then updates the table in place.

    Parameters
    ----------
    lgdo_table
        An :class:`~.lgdo.table.Table` or :class:`~.lgdo.struct.Struct`
        that must contain waveforms so that the DSP can work!
    dsp_dict
        A dictionary containing the DSP config. See :mod:`pygama.dsp.proc_chain` for
        specifics of DSP config format.
    group
        The name of the group that the :class:`rb.lgdo` is being written to.
        If a matching key is found in the :class:`dsp_dict`, that config
        sub_dict is used to do the data trimming. If no match is found,
        then all valid waveform tables are trimmed with the same DSP.

    Notes
    -----
    The original "waveforms" column in the table is deleted!
    If "pass" is present as a the value of a group in the dsp_dict, no trimming is performed.
    Otherwise, dsp_dict is assumed to apply to all valid waveforms and is applied.
    The names of the outputs must follow what is in the example.

    Example dsp_dict
    ------------------


    .. code-block :: json

    {
        "spms": pass,

        "geds" : {
            "outputs" : [ "presummed_waveform", "t_sat_lo", "t_sat_hi"],
            "processors" : {
                "windowed_waveform": {
                    "start_index": 2000,
                    "end_index": -1000
                },
                "presummed_waveform": {
                    "function": "presum",
                    "module": "pygama.dsp.processors",
                    "args": ["waveform", "presummed_waveform(len(waveform)/16, 'f')"],
                    "unit": "ADC"
                },
                "t_sat_lo", "t_sat_hi": {
                    "function": "saturation",
                    "module": "pygama.dsp.processors",
                    "args": ["waveform", 16, "t_sat_lo", "t_sat_hi"],
                    "unit": "ADC"
                }
            }
        }
    }
    """
    # Now check the :meth:`lgdo.RawBufferLibrary`'s group and see if that there is a matching key in the dsp_dict, then take that sub dictionary.
    if group in dsp_dict.keys():
        dsp_dict = deepcopy(dsp_dict[group])

    # If there are no subdicts in dsp_dict, then we use the same dsp on all valid waveform tables in the daq file
    else:
        log.debug("No sub_dicts found, trimming all waveform tables with one dsp_dict")
        dsp_dict = deepcopy(dsp_dict)

    # If the user passes a value "pass" in the dsp_dict for this group, just skip on the data trimming
    if (type(dsp_dict) == str) & (dsp_dict == "pass"):
        return

    # if we want to window waveforms, we can do it outside of processing chain for the sake of memory
    if "windowed_waveform" in dsp_dict["processors"].keys():
        # find the start index from the dsp_dict
        start_index = int(dsp_dict["processors"]["windowed_waveform"]["start_index"])
        end_index = int(dsp_dict["processors"]["windowed_waveform"]["end_index"])

        wf_values = lgdo_table["waveform"]["values"].nda[:, start_index:end_index]
        t0 = process_windowed_t0(
            lgdo_table["waveform"]["t0"], lgdo_table["waveform"]["dt"], start_index
        )
        dt = lgdo_table["waveform"]["dt"]

        # Create the new waveform table
        wf_table = lgdo.WaveformTable(t0=t0, dt=dt, values=wf_values)

        # add this wf_table to the original table
        lgdo_table.add_field("windowed_waveform", wf_table, use_obj_size=True)

        # delete the fake "windowed_waveform" processor in dsp_dict, so that dsp_dict can work in build_processing_chain
        del dsp_dict["processors"]["windowed_waveform"]

    # execute the processing chain
    proc_chain, mask, dsp_out = bpc(lgdo_table, dsp_dict)
    proc_chain.execute()

    # For every processor in dsp_dict for this group, create a new entry in the lgdo table with that processor's name
    # If the processor returns a waveform, create a new waveform table and add it to the original lgdo table
    for proc in dsp_out.keys():

        # Check if the processor returned a waveform or not, denoted by an array of equal sized arrays
        if type(dsp_out[proc]) == ArrayOfEqualSizedArrays:
            # Process dt and t0 for specific dsp processor, can add new ones as necessary
            if proc == "presummed_waveform":
                # find the presum rate from the dsp_dict
                presum_rate_string = dsp_dict["processors"]["presummed_waveform"][
                    "args"
                ][1]
                presum_rate_start_idx = presum_rate_string.find("/") + 1
                presum_rate_end_idx = presum_rate_string.find(",")
                presum_rate = int(
                    presum_rate_string[presum_rate_start_idx:presum_rate_end_idx]
                )

                dt = process_presum_dt(lgdo_table["waveform"]["dt"], presum_rate)
                t0 = lgdo_table["waveform"]["t0"]

            else:
                t0 = lgdo_table["waveform"]["t0"]
                dt = lgdo_table["waveform"]["dt"]

            # Create the new waveform table
            new_obj = lgdo.WaveformTable(t0=t0, dt=dt, values=dsp_out[proc].nda)

        # otherwise, the processor returned just an array
        elif type(dsp_out[proc]) == Array:
            if (proc == "t_sat_lo") or (proc == "t_sat_hi"):
                new_obj = process_saturation(dsp_out[proc])

        # add this new_obj to the original table
        lgdo_table.add_field(proc, new_obj, use_obj_size=True)

    # remove the original waveform
    lgdo_table.pop("waveform")
    lgdo_table.update_datatype()


def process_presum_dt(dts: Array, presum_rate: int) -> Array:
    """
    Multiply a waveform's `dts` by the presumming rate, used for presummed waveforms.
    """
    # don't want to modify the original lgdo_table dts
    copy_dts = copy.deepcopy(dts)

    # change the dt by the presum rate
    copy_dts.nda *= presum_rate
    return copy_dts


def process_windowed_t0(t0s: Array, dts: Array, start_index: int) -> Array:
    """
    In order for the trimmed data to work well with `build_dsp`, we need
    to keep `t0` in its original units.

    So we transform ``start_index`` to the units of ``t0`` and add it to every
    ``t0`` value.
    """
    # don't want to modify the original lgdo_table dts
    copy_dts = copy.deepcopy(dts)
    copy_t0s = copy.deepcopy(t0s)

    # perform t0+start_index*dt to rewrite the new t0 in terms of sample
    start_index *= copy_dts.nda
    copy_t0s.nda += start_index
    return copy_t0s


def process_saturation(n_sat: Array) -> None:
    r"""
    If the :func:`saturation` DSP processor is used, the output is stored in floats.
    However, the output are just the number of times a waveform crosses the saturation,
    so it can be stored as an unsigned integer.
    """
    n_sat.nda = n_sat.nda.astype(np.uint16)
    return n_sat
