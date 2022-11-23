from __future__ import annotations

import copy
import json

import pygama.lgdo as lgdo
from pygama.dsp.processing_chain import build_processing_chain as bpc
from pygama.lgdo import Array, Struct, Table


def data_trimmer(lgdo_table: Table | Struct, dsp_config: str | dict) -> None:
    """
    Takes in an lgdo table that contains waveforms, performs user specified dsp on the table, and then updates the table in place

    Parameters
    ----------
    lgdo_table
        An lgdo Table or Struct that must contain waveforms so that the dsp can work!
    dsp_config
        Either the path to the dsp json config file to use, or a dictionary of dsp config

    Notes
    -----
    The original "waveforms" column in the table is deleted!
    """
    # Convert the dsp_config to a dict so that we can grab the constants from the dsp config
    if isinstance(dsp_config, str) and dsp_config.endswith(".json"):
        f = open(dsp_config)
        dsp_dict = json.load(f)
        f.close()
    # If we get a string that is in the correct format as a json file
    elif isinstance(dsp_config, str):
        dsp_dict = json.loads(dsp_config)
    # Or we could get a dict as the config
    elif isinstance(dsp_config, dict):
        dsp_dict = dsp_config

    # if we want to window waveforms, we can do it outside of processing chain for the sake of memory

    if "windowed_waveform" in dsp_dict["processors"].keys():
        # find the start index from the dsp_config
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

    # for every processed waveform, create a new waveform table and add it to the original lgdo table
    for proc in dsp_out.keys():

        # Process dt and t0 for specific dsp processor, can add new ones as necessary
        if proc == "presummed_waveform":
            # find the presum rate from the dsp_config
            presum_rate_string = dsp_dict["processors"]["presummed_waveform"]["args"][1]
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
        wf_table = lgdo.WaveformTable(t0=t0, dt=dt, values=dsp_out[proc].nda)

        # add this wf_table to the original table
        lgdo_table.add_field(proc, wf_table, use_obj_size=True)

    # remove the original waveform
    lgdo_table.pop("waveform")
    lgdo_table.update_datatype()


def process_presum_dt(dts: Array, presum_rate: int) -> Array:
    """
    Multiply a waveform's dts by the presumming rate, used for presummed waveforms
    """
    # don't want to modify the original lgdo_table dts
    copy_dts = copy.deepcopy(dts)

    # change the dt by the presum rate
    copy_dts.nda *= presum_rate
    return copy_dts


def process_windowed_t0(t0s: Array, dts: Array, start_index: int) -> Array:
    """
    Transform the t0 of a waveform to number of samples, and then shift by the start_index of a windowed waveform
    """
    # don't want to modify the original lgdo_table dts
    copy_dts = copy.deepcopy(dts)
    copy_t0s = copy.deepcopy(t0s)

    # perform t0/dt+start_index to rewrite the new t0 in terms of sample
    copy_t0s.nda /= copy_dts.nda
    copy_t0s.nda += start_index
    copy_t0s.attrs["units"] = "samples"
    return copy_t0s
