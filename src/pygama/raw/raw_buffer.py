r"""
Utilities to manage data buffering for raw data conversion. This module manages
LGDO buffers and their corresponding output streams. Allows for one-to-many
mapping of input streams to output streams.

Primary Classes
---------------
:class:`.RawBuffer`: an LGDO (e.g. a table) along with buffer metadata, such as the
current write location, the list of keys (e.g. channels) that write to it, the
output stream it is associated with (if any), etc. Each
:class:`~.raw.data_decoder.DataDecoder` is associated with a
:class:`.RawBuffer` of a particular format.

:class:`.RawBufferList`: a collection of :class:`RawBuffer` with LGDO's that
all have the same structure (same type, same fields, etc). A
:class:`~.raw.data_decoder.DataDecoder` will write its output to a
:class:`.RawBufferList`.

:class:`.RawBufferLibrary`: a dictionary of :class:`RawBufferList`\ s, e.g. one
for each :class:`~.raw.data_decoder.DataDecoder`. Keyed by the decoder name.

:class:`.RawBuffer` supports a JSON short-hand notation, see
:meth:`.RawBufferLibrary.set_from_json_dict` for full specification.

Example JSON yielding a valid :class:`.RawBufferLibrary` is below. In the
example, the user would call ``RawBufferLibrary.set_from_json_dict(json_dict,
kw_dict)`` with ``kw_dict`` containing an entry for ``'file_key'``. The other
keywords ``{key}`` and ``{name}`` are understood by and filled in during
:meth:`.RawBufferLibrary.set_from_json_dict` unless overloaded in ``kw_dict``.
Note the use of the wildcard ``*``: this will match all other decoder names /
keys.

.. code-block :: json

    {
      "FCEventDecoder" : {
        "g{key:0>3d}" : {
          "key_list" : [ [24,64] ],
          "out_stream" : "$DATADIR/{file_key}_geds.lh5:/geds",
          "proc_spec": {
            "window": {"waveform", 10, 100, "windowed_waveform"}
          }
        },
        "spms" : {
          "key_list" : [ [6,23] ],
          "out_stream" : "$DATADIR/{file_key}_spms.lh5:/spms"
        },
        "puls" : {
          "key_list" : [ 0 ],
          "out_stream" : "$DATADIR/{file_key}_auxs.lh5:/auxs"
        },
        "muvt" : {
          "key_list" : [ 1, 5 ],
          "out_stream" : "$DATADIR/{file_key}_auxs.lh5:/auxs"
        }
      },
      "*" : {
        "{name}" : {
          "key_list" : [ "*" ],
          "out_stream" : "$DATADIR/{file_key}_{name}.lh5"
        }
      }
    }
"""
from __future__ import annotations

import os
from typing import Union

from pygama import lgdo
from pygama.lgdo.lh5_store import LH5Store
from pygama.raw.buffer_processor.buffer_processor import buffer_processor

LGDO = Union[lgdo.Scalar, lgdo.Struct, lgdo.Array, lgdo.VectorOfVectors]


class RawBuffer:
    r"""Base class to represent a buffer of raw data.

    A :class:`RawBuffer` is in essence a an LGDO object (typically a
    :class:`~.lgdo.table.Table`) to which decoded data will be written, along
    with some meta-data distinguishing what data goes into it, and where the
    LGDO gets written out. Also holds on to the current location in the buffer
    for writing.

    Attributes
    ----------
    lgdo
        the LGDO used as the actual buffer. Typically a
        :class:`~.lgdo.table.Table`. Set to ``None`` upon creation so that the
        user or a decoder can initialize it later.
    key_list
        a list of keys (e.g. channel numbers) identifying data to be written
        into this buffer. The key scheme is specific to the decoder with which
        the :class:`.RawBuffer` is associated. This is called `key_list`
        instead of `keys` to avoid confusion with the dict function
        :meth:`dict.keys`, i.e.  ``raw_buffer.lgdo.keys()``.
    out_stream
        the output stream to which the :class:`.RawBuffer`\ 's LGDO should be
        sent or written. A colon (``,``) can be used to separate the stream
        name/address from an in-stream path/port:
        - file example: ``/path/filename.lh5:/group``
        - socket example: ``198.0.0.100:8000``
    out_name
        the name or identifier of the object in the output stream.
    proc_spec
        a dictionary containing the following:
        - a DSP config file, passed as a dictionary, or as a path to a JSON file
        - an array containing: the name of an :class:`~.lgdo` object stored in the :class:`.RawBuffer` to be sliced,
        the start and end indices of the slice, and the new name for the sliced object
        - a dictionary of fields to drop
        - a dictionary of new fields and their return datatype
        these specifications are used to process the data with :meth:`.buffer_processor.buffer_processor.buffer_processor`,
        refer to the documentation for more details on how the format of `proc_spec` and how the processing is performed.
    """

    def __init__(
        self,
        lgdo: LGDO = None,
        key_list: list[int | str] = None,
        out_stream: str = "",
        out_name: str = "",
        proc_spec: dict = None,
    ) -> None:
        self.lgdo = lgdo
        self.key_list = [] if key_list is None else key_list
        self.out_stream = out_stream
        self.out_name = out_name
        self.proc_spec = proc_spec
        self.loc = 0
        self.fill_safety = 1

    def __len__(self) -> int:
        if self.lgdo is None:
            return 0
        if not hasattr(self.lgdo, "__len__"):
            return 1
        return len(self.lgdo)

    def is_full(self) -> bool:
        return (len(self) - self.loc) < self.fill_safety

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return (
            "RawBuffer(lgdo="
            + repr(self.lgdo)
            + ", key_list="
            + repr(self.key_list)
            + ", out_stream="
            + repr(self.out_stream)
            + ", out_name="
            + repr(self.out_name)
            + ", loc="
            + repr(self.loc)
            + ", fill_safety="
            + repr(self.fill_safety)
            + ", proc_spec="
            + repr(self.proc_spec)
            + ")"
        )


class RawBufferList(list):
    r"""A :class:`.RawBufferList` holds a collection of :class:`.RawBuffer`\ s
    of identical structure (same format LGDO's with the same fields).
    """

    def __init__(self) -> None:
        self.keyed_dict = None

    def get_keyed_dict(self) -> dict[int | str, RawBuffer]:
        r"""Returns a dictionary of :class:`.RawBuffer`\ s built from the
        buffers' `key_lists`.

        Different keys may point to the same buffer. Requires the buffers in
        the :class:`.RawBufferList` to have non-overlapping key lists.
        """
        if self.keyed_dict is None:
            self.keyed_dict = {}
            for rb in self:
                for key in rb.key_list:
                    self.keyed_dict[key] = rb
        return self.keyed_dict

    def set_from_json_dict(
        self, json_dict: dict, kw_dict: dict[str, str] = None
    ) -> None:
        """Set up a :class:`.RawBufferList` from a dictionary written in JSON
        shorthand. See :meth:`.RawBufferLibrary.set_from_json_dict` for details.

        Notes
        -----
        `json_dict` is changed by this function.
        """
        expand_rblist_json_dict(json_dict, {} if kw_dict is None else kw_dict)
        for name in json_dict:
            rb = RawBuffer()
            if "key_list" in json_dict[name]:
                rb.key_list = json_dict[name]["key_list"]
            if "out_stream" in json_dict[name]:
                rb.out_stream = json_dict[name]["out_stream"]
            if "proc_spec" in json_dict[name]:
                rb.proc_spec = json_dict[name][
                    "proc_spec"
                ]  # If you swap this with the next line, then key expansion doesn't work
            if "out_name" in json_dict[name]:
                rb.out_name = json_dict[name]["out_name"]
            else:
                rb.out_name = name
            self.append(rb)

    def get_list_of(self, attribute: str) -> list:
        """Return a list of values of :class:`.RawBuffer` attributes.

        Parameters
        ----------
        attribute
            The :class:`.RawBuffer` attribute queried to make the list.

        Returns
        -------
        values
            The list of values of `RawBuffer.attribute`.

        Examples
        --------
        >>> output_file_list = rbl.get_list_of('out_stream')
        """
        values = []
        for rb in self:
            if not hasattr(rb, attribute):
                continue
            val = getattr(rb, attribute)
            if val not in values:
                values.append(val)
        return values

    def clear_full(self) -> None:
        for rb in self:
            if rb.is_full():
                rb.loc = 0


class RawBufferLibrary(dict):
    r"""A :class:`.RawBufferLibrary` is a collection of
    :class:`.RawBufferList`\ s associated with the names of decoders that can
    write to them.
    """

    def __init__(self, json_dict: dict = None, kw_dict: dict[str, str] = None) -> None:
        if json_dict is not None:
            self.set_from_json_dict(json_dict, kw_dict)

    def set_from_json_dict(
        self, json_dict: dict, kw_dict: dict[str, str] = None
    ) -> None:
        r"""Set up a :class:`.RawBufferLibrary` from a dictionary written in
        JSON shorthand.

        Basic structure:

        .. code-block :: js

            {
            "list_name" : {
              "name" : {
                  "key_list" : [ "key1", "key2", "..." ],
                  "out_stream" : "out_stream_str",
                  "out_name" : "out_name_str" // (optional)
                  "proc_spec" : { // (optional)
                    "windowed":
                        ["waveform", 10, 100, "windowed_waveform"],

                  }
              }
            }

        By default ``name`` is used for the :class:`RawBuffer`\ 's ``out_name``
        attribute, but this can be overridden if desired by providing an
        explicit ``out_name``.

        Allowed shorthands, in order of expansion:

        * ``key_list`` may have entries that are 2-integer lists corresponding
          to the first and last integer keys in a contiguous range (e.g. of
          channels) that get stored to the same buffer. These simply get
          replaced with the explicit list of integers in the range. We use
          lists not tuples for JSON compliance.
        * The ``name`` can include ``{key:xxx}`` format specifiers, indicating
          that each key in ``key_list`` should be given its own buffer with the
          corresponding name.  The same specifier can appear in ``out_path`` to
          write the key's data to its own output path.
        * You may also include keywords in your ``out_stream`` and ``out_name``
          specification whose values get sent in via `kw_dict`. These get
          evaluated simultaneously with the ``{key:xxx}`` specifiers.
        * Environment variables can also be used in ``out_stream``. They get
          expanded after `kw_dict` is handled and thus can be used inside
          `kw_dict`.
        * ``list_name`` can use the wildcard ``*`` to match any other
          ``list_name`` known to a streamer.
        * ``out_stream`` and ``out_name`` can also include ``{name}``, to be
          replaced with the buffer's ``name``. In the case of
          ``list_name="*"``, ``{name}`` evaluates to ``list_name``.

        Parameters
        ----------
        json_dict
            loaded from a JSON file written in the allowed shorthand.
            `json_dict` is changed by this function.
        kw_dict
            dictionary of keyword-value pairs for substitutions into the
            ``out_stream`` and ``out_name`` fields.
        """
        for list_name in json_dict:
            if list_name not in self:
                self[list_name] = RawBufferList()
            self[list_name].set_from_json_dict(
                json_dict[list_name], {} if kw_dict is None else kw_dict
            )

    def get_list_of(self, attribute: str, unique: bool = True) -> list:
        """Return a list of values of :class:`.RawBuffer` attributes.

        Parameters
        ----------
        attribute
            The :class:`.RawBuffer` attribute queried to make the list.
        unique
            whether to remove duplicates.

        Returns
        -------
        values
            The list of values of `RawBuffer.attribute`.

        Examples
        --------
        >>> output_file_list = rbl.get_list_of('out_stream')
        """
        values = []
        for rb_list in self.values():
            values += rb_list.get_list_of(attribute)
        if unique:
            values = list(set(values))
        return values

    def clear_full(self) -> None:
        for rb_list in self.values():
            rb_list.clear_full()


def expand_rblist_json_dict(json_dict: dict, kw_dict: dict[str, str]) -> None:
    """Expand shorthands in a JSON dictionary representing a
    :class:`.RawBufferList`.

    See :meth:`.RawBufferLibrary.set_from_json_dict` for details.

    Notes
    -----
    The input JSON dictionary is changed by this function.
    """
    # get the original list of groups because we are going to change the
    # dict.keys() of json_dict inside the next list. Note: we have to convert
    # from dict_keys to list here otherwise the loop complains about changing
    # the dictionary during iteration
    buffer_names = list(json_dict.keys())
    for name in buffer_names:
        if name == "":
            raise ValueError("buffer name can't be empty")

        info = json_dict[name]  # changes to info will change json_dict[name]
        # make sure we have a key list
        if "key_list" not in info:
            raise ValueError(f"'{name}' is missing key_list")
            continue

        # find and expand any ranges in the key_list
        # do in a while loop with a controlled index since we are modifying
        # the length of the list inside the loop (be careful)
        i = 0
        while i < len(info["key_list"]):
            key_range = info["key_list"][i]
            # expand any 2-int lists
            if isinstance(key_range, list) and len(key_range) == 2:
                info["key_list"][i : i + 1] = range(key_range[0], key_range[1] + 1)
                i += key_range[1] - key_range[0]
            i += 1

        # Expand list_names if name contains a key-based formatter
        if "{key" in name:
            if len(info["key_list"]) == 1 and info["key_list"][0] == "*":
                continue  # will be handled later, once the key_list is known
            for key in info["key_list"]:
                expanded_name = name.format(key=key)
                json_dict[expanded_name] = info.copy()
                json_dict[expanded_name]["key_list"] = [key]
            json_dict.pop(name)

    # now re-iterate and expand out_paths
    for name, info in json_dict.items():
        if len(info["key_list"]) == 1 and info["key_list"][0] != "*":
            kw_dict["key"] = info["key_list"][0]
        if "out_stream" in info:
            if name != "*" and "{name" in info["out_stream"]:
                kw_dict["name"] = name
            try:
                info["out_stream"] = info["out_stream"].format(**kw_dict)
            except KeyError as msg:
                raise KeyError(
                    f"variable {msg} dereferenced in 'out_stream' not defined in kw_dict"
                )
            info["out_stream"] = os.path.expandvars(info["out_stream"])


def write_to_lh5_and_clear(
    raw_buffers: list[RawBuffer], lh5_store: LH5Store = None, wo_mode: str = "append"
) -> None:
    r"""Write a list of :class:`.RawBuffer`\ s to LH5 files and then clears
    them.

    Parameters
    ----------
    raw_buffers : list(RawBuffer)
        The list of RawBuffers to be written to file. Note this is not a
        RawBufferList because the RawBuffers may not have the same structure.
        If a raw_buffer has a `proc_spec` attribute, then :meth:`.buffer_processor.buffer_processor.buffer_processor`
        is used to process that raw_buffer.
    lh5_store : LH5Store or None
        Allows user to send in a store holding a collection of already open
        files (saves some time opening / closing files)
    wo_mode : str
        write mode, see also :meth:`.lgdo.lh5_store.LH5Store.write_object`
    """
    if lh5_store is None:
        lh5_store = lgdo.LH5Store()
    for rb in raw_buffers:
        if rb.lgdo is None or rb.loc == 0:
            continue  # no data to write
        ii = rb.out_stream.find(":")
        if ii == -1:
            filename = rb.out_stream
            group = "/"
        else:
            filename = rb.out_stream[:ii]
            group = rb.out_stream[ii + 1 :]
            if len(group) == 0:
                group = "/"  # in case out_stream ends with :

        # If a proc_spec if present for this RawBuffer, process that data and then write to the file!
        if rb.proc_spec is not None:
            # Perform the processing as requested in the `proc_spec` from `out_spec` in build_raw
            lgdo_to_write = buffer_processor(rb)
        else:
            lgdo_to_write = rb.lgdo

        # write if requested...
        if filename != "":
            lh5_store.write_object(
                lgdo_to_write,
                rb.out_name,
                filename,
                group=group,
                n_rows=rb.loc,
                wo_mode=wo_mode,
            )
        # and clear
        rb.loc = 0
