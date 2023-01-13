from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np

import pygama.lgdo as lgdo
from pygama.dsp.processing_chain import build_processing_chain as bpc
from pygama.lgdo import Array, ArrayOfEqualSizedArrays

if TYPE_CHECKING:
    from pygama.raw.raw_buffer import RawBuffer

log = logging.getLogger(__name__)


def buffer_processor(rb: RawBuffer) -> None:
    """
    Takes in a :class:`.RawBuffer`, performs any of the four processes specified from the :class:`.RawBuffer`'s ``proc_spec`` attribute:
    - windows objects with a name specified by the first argument passed in the ``proc_spec``, the window start and stop indices are the next two
    arguments, and then updates the rb.lgdo with a name specified by the last argument. If the object is an :func:`pygama.lgdo.WaveformTable`, then
    the ``t0`` and ``dt`` attributes are updated accordingly.
    - performs DSP given by the "dsp_config" key in the ``proc_spec``. See :module:`pygama.dsp` for more information on DSP config dictionaries.
    All fields in the output of the DSP are written to the rb.lgdo
    - drops any requested fields from the rb.lgdo
    - updates the data types of any field with its requested datatype in ``proc_spec``


    Parameters
    ----------
    rb
        A :class:`.RawBuffer` to be processed, must contain a ``proc_spec`` attribute


    Notes
    -----
    The original "waveforms" column in the table is deleted if requested! All updates to the rb.lgdo are done in place


    Example ``proc_spec`` in an :module:`pygama.raw.build_raw` ``out_spec``
    -----------------------------------------------------------------------


    .. code-block :: json


    {
        "FCEventDecoder" : {
        "g{key:0>3d}" : {
            "key_list" : [ [24,64] ],
            "out_stream" : "$DATADIR/{file_key}_geds.lh5:/geds",
            "proc_spec": {
                "window":
                    ["waveform", 100, -100, "windowed_waveform"],
                "dsp_config": {
                    "outputs": [ "presummed_waveform", "t_sat_lo", "t_sat_hi" ],
                    "processors": {
                        "presummed_waveform": {
                            "function": "presum",
                            "module": "pygama.dsp.processors",
                            "args": ["waveform", "presummed_waveform(len(waveform)/16, 'f')"],
                            "unit": "ADC"
                            },
                        "t_sat_lo, t_sat_hi": {
                            "function": "saturation",
                            "module": "pygama.dsp.processors",
                            "args": ["waveform", 16, "t_sat_lo", "t_sat_hi"],
                            "unit": "ADC"
                            }
                        }
                },
                "drop": {
                    "waveform"
                },
                "return_type": {
                    "windowed_waveform/values": "uint16",
                    "presummed_waveform/values": "uint32",
                    "t_sat_lo": "uint16",
                    "t_sat_hi": "uint16",
                }
            }
        },
        "spms" : {
            "key_list" : [ [6,23] ],
            "out_stream" : "$DATADIR/{file_key}_spms.lh5:/spms"
        },
    }
    """

    # Perform windowing, if requested
    if "window" in rb.proc_spec.keys():
        process_window(rb)

    # Read in and perform the DSP routine
    if "dsp_config" in rb.proc_spec.keys():
        process_dsp(rb)

    # Cast as requested dtype before writing to the table
    if "return_type" in rb.proc_spec.keys():
        process_return_type(rb)

    # Drop any requested columns from the table
    if "drop" in rb.proc_spec.keys():
        for drop_keys in rb.proc_spec["drop"]:
            rb.lgdo.pop(drop_keys)
            rb.lgdo.update_datatype()

    return None


def process_window(rb: RawBuffer) -> None:
    r"""
    Windows arrays of equal sized arrays according to specifications
    given in the rb.proc_spec "window" key.

    First checks if the rb.lgdo is a table or not. If it's not a table,
    then we only process it if its rb.out_name is the same as the window_in_name.

    If rb.lgdo is a table, special processing is done if the window_in_name field
    is an lgdo.WaveformTable in order to update the t0s. Otherwise, windowing of the field
    is performed without updating any of the other attributes.

    Parameters
    ----------
    rb
        A :class:`.RawBuffer` to be processed

    """
    # Read the window parameters from the proc_spec
    window_in_name = rb.proc_spec["window"][0]
    window_start_idx = int(rb.proc_spec["window"][1])
    window_end_idx = int(rb.proc_spec["window"][2])
    window_out_name = rb.proc_spec["window"][3]

    # Check if rb.lgdo is a table and if the window_in_name is a key
    if (isinstance(rb.lgdo, lgdo.Table) or isinstance(rb.lgdo, lgdo.Struct)) and (
        window_in_name in rb.lgdo.keys()
    ):
        # Now check if the window_in_name is a waveform table or not, if so we need to modify the t0s
        if isinstance(rb.lgdo[window_in_name], lgdo.WaveformTable):
            # modify the t0s
            t0s = process_windowed_t0(
                rb.lgdo[window_in_name].t0, rb.lgdo[window_in_name].dt, window_start_idx
            )

            # Window the waveform values
            array_of_arrays = rb.lgdo[window_in_name].values
            windowed_array_of_arrays = window_array_of_arrays(
                array_of_arrays, window_start_idx, window_end_idx
            )

            # Write to waveform table and then to file
            wf_table = lgdo.WaveformTable(
                t0=t0s, dt=rb.lgdo[window_in_name].dt, values=windowed_array_of_arrays
            )

            # add this wf_table to the original table
            rb.lgdo.add_field(window_out_name, wf_table, use_obj_size=True)

        # otherwise, it's (hopefully) just an array of equal sized arrays
        else:
            array_of_arrays = rb.lgdo[window_in_name]
            windowed_array_of_arrays = window_array_of_arrays(
                array_of_arrays, window_start_idx, window_end_idx
            )
            rb.lgdo.add_field(
                window_out_name, windowed_array_of_arrays, use_obj_size=True
            )

        return None

    # otherwise, rb.lgdo is some other type and we only process it if the rb.out_name is the same as window_in_name
    elif rb.out_name == window_in_name:
        array_of_arrays = rb.lgdo
        windowed_array_of_arrays = window_array_of_arrays(
            array_of_arrays, window_start_idx, window_end_idx
        )

        rb.out_name = window_out_name
        rb.lgdo = windowed_array_of_arrays

        return None

    else:
        raise KeyError(f"{window_in_name} not a valid key for this RawBuffer")


def window_array_of_arrays(
    array_of_arrays: ArrayOfEqualSizedArrays, window_start_idx: int, window_end_idx: int
) -> ArrayOfEqualSizedArrays:
    r"""
    Given an array of equal sized arrays, for each array it returns the view [window_start_idx:window_end_idx]
    """
    if isinstance(array_of_arrays, lgdo.ArrayOfEqualSizedArrays):
        return array_of_arrays.nda[:, window_start_idx:window_end_idx]
    else:
        raise TypeError(
            f"Do not know how to window an LGDO of type {type(array_of_arrays)}"
        )


def process_presum(
    rb: RawBuffer, presum_obj: ArrayOfEqualSizedArrays, dsp_dict: dict, proc: str
) -> lgdo.WaveformTable:
    r"""
    Finds the presum rate that was used to perform the DSP and writes this to a table.
    The name of a valid :func:`pygama.lgdo.WaveformTable` that DSP was performed on is used to extract the original
    ``dt`` so that they can be updated by the presum rate. Then the presummed table is added to the rb.lgdo

    Parameters
    ----------
    rb
        A :module:`.RawBuffer` containing an rb.lgdo to store the presummed object in
    presum_obj
        The :module:`pygama.lgdo.ArrayofEqualSizedArrays` containing the presummed output from the DSP
    dsp_dict
        The dictionary that was used to perform the DSP
    proc
        The processor name that called :func:`pygama.dsp._processors.presum` in the DSP
    """
    # find the presum rate from the dsp_dict
    presum_rate_string = dsp_dict["processors"][proc]["args"][1]
    presum_rate_start_idx = presum_rate_string.find("/") + 1
    presum_rate_end_idx = presum_rate_string.find(",")
    presum_rate = int(presum_rate_string[presum_rate_start_idx:presum_rate_end_idx])

    # Find the original lgdo field that was used to create the presummed output
    rb_lgdo_field = dsp_dict["processors"][proc]["args"][0]

    # make sure that this field is a waveform table, then process dts
    if isinstance(rb.lgdo[rb_lgdo_field], lgdo.WaveformTable):
        dt = process_presum_dt(rb.lgdo[rb_lgdo_field]["dt"], presum_rate)
        t0 = rb.lgdo[rb_lgdo_field]["t0"]
    else:
        raise TypeError(f"field {rb_lgdo_field} is not a valid lgdo.WaveformTable")

    # Create the new waveform table
    new_obj = lgdo.WaveformTable(t0=t0, dt=dt, values=presum_obj.nda)

    # Write the presum_rate as an array to the table as a new field
    presum_rate_array = lgdo.Array(shape=len(t0), dtype=np.uint16, fill_val=presum_rate)
    rb.lgdo.add_field("presum_rate", presum_rate_array)

    return new_obj


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
    In order for the processed data to work well with :module:`pygama.dsp.build_dsp`, we need
    to keep ``t0`` in its original units.

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


def process_dsp(rb: RawBuffer) -> None:
    r"""
    Run a provided DSP config from rb.proc_spec using build_processing_chain, and add specified outputs to the
    rb.lgdo.

    Notes
    -----
    rb.lgdo is assumed to be an lgdo.Table so that multiple DSP processor outputs can be written to it
    """
    # Load the dsp_dict
    dsp_dict = rb.proc_spec["dsp_config"]

    # execute the processing chain
    # This checks that the rb.lgdo is a table and that the field_name is present in the table
    proc_chain, mask, dsp_out = bpc(rb.lgdo, dsp_dict)
    proc_chain.execute()

    # For every processor in dsp_dict for this group, create a new entry in the lgdo table with that processor's name
    # If the processor returns a waveform, create a new waveform table and add it to the original lgdo table
    for proc in dsp_out.keys():
        # Check what DSP routine the processors output is from, and manipulate accordingly
        for dsp_proc in dsp_dict["processors"].keys():
            if proc in dsp_proc:
                # In the case of presumming, change the dts
                if dsp_dict["processors"][dsp_proc]["function"] == "presum":
                    new_obj = process_presum(rb, dsp_out[proc], dsp_dict, proc)
                # In the case of saturation, change the dtype
                if dsp_dict["processors"][dsp_proc]["function"] == "saturation":
                    new_obj = process_saturation(dsp_out[proc])
        rb.lgdo.add_field(proc, new_obj, use_obj_size=True)

    return None


def process_return_type(rb: RawBuffer) -> None:
    """
    Change the types of fields in an rb.lgdo according to the values specified in the ``proc_spec``'s ``return_type`` dictionary

    Notes
    -----
    This assumes that name provided points to an object in the rb.lgdo that has an `nda` attribute
    """
    type_list = rb.proc_spec["return_type"]
    for return_name in type_list.keys():
        # Take care of nested tables with a for loop
        path = return_name.split("/")
        return_value = rb.lgdo
        for key in path:
            return_value = return_value[key]

        # If we have a numpy array as part of the lgdo, recast its type
        if hasattr(return_value, "nda"):
            return_value.nda = return_value.nda.astype(np.dtype(type_list[return_name]))
        else:
            raise TypeError(f"Cannot recast an object of type {type(return_value)}")

    return None
