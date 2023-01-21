from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np

import pygama.lgdo as lgdo
from pygama.dsp.errors import ProcessingChainError
from pygama.dsp.processing_chain import build_processing_chain as bpc
from pygama.lgdo import Array, ArrayOfEqualSizedArrays, Table

if TYPE_CHECKING:
    from pygama.raw.raw_buffer import RawBuffer

log = logging.getLogger(__name__)


def buffer_processor(rb: RawBuffer) -> Table:
    r"""
    Takes in a :class:`.RawBuffer`, performs any processes specified in the :class:`.RawBuffer`'s ``proc_spec``
    attribute, and returns a :class:`pygama.lgdo.Table` with the processed buffer. This `tmp_table` shares columns with the :class:`.RawBuffer`'s lgdo,
    so no data is copied. Currently implemented attributes:

    - "window" (list): [in_name, start_index, stop_index, out_name]
      Windows objects with a name specified by the first argument passed in the ``proc_spec``, the window start and stop indices are the next two
      arguments, and then updates the rb.lgdo with a name specified by the last argument. If the object is an :func:`pygama.lgdo.WaveformTable`, then
      the ``t0`` attribute is updated accordingly. Although it is possible to use the DSP config to perform windowing, this hard-coded version
      avoids a conversion to float32.
    - "dsp_config" (dict): {dsp_config}
      Performs DSP given by the "dsp_config" key in the ``proc_spec``. See :mod:`pygama.dsp.processing_chain.build_processing_chain` for more information on DSP config dictionaries.
      All fields in the output of the DSP are written to the rb.lgdo
    - "drop" (list): ["waveform", "packet_ids"]
      Drops any requested fields from the rb.lgdo
    - "dtype_conv" (dict): {"presummed_waveform/values": "uint32", "t_sat_lo": "uint16"}
      Updates the data types of any field with its requested datatype in ``proc_spec``

    Parameters
    ----------
    rb
        A :class:`.RawBuffer` to be processed, must contain a ``proc_spec`` attribute

    Notes
    -----
    The original "waveforms" column in the table is not written to file if request! All updates are done on the
    `tmp_table`, which shares the fields with `rb.lgdo` and are done in place. The `tmp_table` is necessary so that
    the `rb.lgdo` keeps arrays needed by the table in the buffer.
    An example ``proc_spec`` in an :mod:`pygama.raw.build_raw` ``out_spec`` is below

    .. code-block:: json

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
                        "args": ["waveform", "presummed_waveform(shape=len(waveform)//16, period=waveform.period*16, offset=waveform.offset, 'f')"],
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
                "drop":
                    ["waveform", "packet_ids"],
                "dtype_conv": {
                    "presummed_waveform/values": "uint32",
                    "t_sat_lo": "uint16",
                    "t_sat_hi": "uint16",
                }
              }
           },
           "spms" : {
             "key_list" : [ [6,23] ],
             "out_stream" : "$DATADIR/{file_key}_spms.lh5:/spms"
           }
        }

    """
    # Check that there is a valid object to process
    if isinstance(rb.lgdo, lgdo.Table) or isinstance(rb.lgdo, lgdo.Struct):
        # Create the temporary table that will be written to file
        rb_table_size = rb.lgdo.size
        tmp_table = Table(size=rb_table_size)
        tmp_table.join(other_table=rb.lgdo)

    # This is needed if there is a "*" key expansion for decoders in build_raw
    # In the worst case, just return an unprocessed rb.lgdo
    else:
        log.info(f"Cannot process buffer with an lgdo of type {type(rb.lgdo)}")
        tmp_table = rb.lgdo
        return tmp_table

    # Perform windowing, if requested
    if "window" in rb.proc_spec.keys():
        process_window(rb, tmp_table)

    # Read in and perform the DSP routine
    if "dsp_config" in rb.proc_spec.keys():
        process_dsp(rb, tmp_table)

    # Cast as requested dtype before writing to the table
    if "dtype_conv" in rb.proc_spec.keys():
        process_dtype_conv(rb, tmp_table)

    # Drop any requested columns from the table
    if "drop" in rb.proc_spec.keys():
        process_drop(rb, tmp_table)

    return tmp_table


def process_window(rb: RawBuffer, tmp_table: Table) -> None:
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
    tmp_table
        A :class:`pygama.lgdo.Table` that shares columns with the `rb.lgdo`

    Notes
    -----
    This windowing hard-coded; it is done without calling :mod:`pygama.dsp.build_dsp` to avoid
    a conversion to float32.

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
            array_of_arrays = tmp_table[window_in_name].values
            windowed_array_of_arrays = window_array_of_arrays(
                array_of_arrays, window_start_idx, window_end_idx
            )

            # Write to waveform table and then to file
            wf_table = lgdo.WaveformTable(
                t0=t0s, dt=rb.lgdo[window_in_name].dt, values=windowed_array_of_arrays
            )

            # add this wf_table to the temporary table
            tmp_table.add_field(window_out_name, wf_table, use_obj_size=True)

        # otherwise, it's (hopefully) just an array of equal sized arrays
        else:
            array_of_arrays = tmp_table[window_in_name]
            windowed_array_of_arrays = window_array_of_arrays(
                array_of_arrays, window_start_idx, window_end_idx
            )
            tmp_table.add_field(
                window_out_name, windowed_array_of_arrays, use_obj_size=True
            )

        return None

    # otherwise, rb.lgdo is some other type and we only process it if the rb.out_name is the same as window_in_name
    elif rb.out_name == window_in_name:
        array_of_arrays = tmp_table
        windowed_array_of_arrays = window_array_of_arrays(
            array_of_arrays, window_start_idx, window_end_idx
        )

        rb.out_name = window_out_name
        tmp_table = windowed_array_of_arrays

        return None

    else:
        log.info(f"{window_in_name} not a valid key for this RawBuffer")
        return None


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


def process_windowed_t0(t0s: Array, dts: Array, start_index: int) -> Array:
    """
    In order for the processed data to work well with :mod:`pygama.dsp.build_dsp`, we need
    to keep ``t0`` in its original units.

    So we transform ``start_index`` to the units of ``t0`` and add it to every
    ``t0`` value.
    """
    # don't want to modify the original lgdo_table t0s
    copy_t0s = copy.deepcopy(t0s)

    # perform t0+start_index*dt to rewrite the new t0 in terms of sample
    start_index *= dts.nda
    copy_t0s.nda += start_index
    return copy_t0s


def process_dsp(rb: RawBuffer, tmp_table: Table) -> None:
    r"""
    Run a provided DSP config from rb.proc_spec using build_processing_chain, and add specified outputs to the
    rb.lgdo.

    Parameters
    ----------
    rb
        A :class:`.RawBuffer` that contains a `proc_spec` and an `lgdo` attribute
    tmp_table
        A :class:`pygama.lgdo.Table` that is temporarily created to be written to the raw file

    Notes
    -----
    rb.lgdo is assumed to be an lgdo.Table so that multiple DSP processor outputs can be written to it
    """
    # Load the dsp_dict
    dsp_dict = rb.proc_spec["dsp_config"]

    # Try building the processing chain
    try:
        # execute the processing chain
        # This checks that the rb.lgdo is a table and that the field_name is present in the table
        proc_chain, mask, dsp_out = bpc(rb.lgdo, dsp_dict)
    # Allow for exceptions, in the case of "*" key expansion in the build_raw out_spec
    except ProcessingChainError:
        log.info("DSP could not be performed")
        return None

    proc_chain.execute()

    # For every processor in dsp_dict for this group, create a new entry in the lgdo table with that processor's name
    # If the processor returns a waveform, create a new waveform table and add it to the original lgdo table
    for proc in dsp_out.keys():
        # # Check what DSP routine the processors output is from, and manipulate accordingly
        tmp_table.add_field(proc, dsp_out[proc], use_obj_size=True)

    return None


def process_dtype_conv(rb: RawBuffer, tmp_table: Table) -> None:
    """
    Change the types of fields in an rb.lgdo according to the values specified in the ``proc_spec``'s ``dtype_conv`` list.
    It operates in place on `tmp_table`.

    Notes
    -----
    This assumes that name provided points to an object in the rb.lgdo that has an `nda` attribute
    """
    type_list = rb.proc_spec["dtype_conv"]
    for return_name in type_list.keys():
        # Take care of nested tables with a for loop
        path = return_name.split("/")
        return_value = tmp_table
        for key in path:
            try:
                return_value = return_value[key]
            # Allow for exceptions in the case of "*" expansion in the build_raw out_spec
            except KeyError:
                log.info(f"{key} is not a valid key in the rb.lgdo")
                return None

        # If we have a numpy array as part of the lgdo, recast its type
        if hasattr(return_value, "nda"):
            return_value.nda = return_value.nda.astype(np.dtype(type_list[return_name]))
        else:
            raise TypeError(f"Cannot recast an object of type {type(return_value)}")

    return None


def process_drop(rb: RawBuffer, tmp_table: Table) -> None:
    for drop_keys in rb.proc_spec["drop"]:
        try:
            tmp_table.remove_column(drop_keys, delete=False)
        except KeyError:
            log.info(f"Cannot remove field {drop_keys} from rb.lgdo")
            return None
