from __future__ import annotations

from bisect import bisect_right
from typing import Union

import numpy as np
import pandas as pd

from .. import Array, Scalar, Struct, VectorOfVectors
from . import utils
from .store import LH5Store

LGDO = Union[Array, Scalar, Struct, VectorOfVectors]


class LH5Iterator:
    """
    A class for iterating through one or more LH5 files, one block of entries
    at a time. This also accepts an entry list/mask to enable event selection,
    and a field mask.

    This class can be used either for random access:

    >>> lh5_obj, n_rows = lh5_it.read(entry)

    to read the block of entries starting at entry. In case of multiple files
    or the use of an event selection, entry refers to a global event index
    across files and does not count events that are excluded by the selection.

    This can also be used as an iterator:

    >>> for lh5_obj, entry, n_rows in LH5Iterator(...):
    >>>    # do the thing!

    This is intended for if you are reading a large quantity of data but
    want to limit your memory usage (particularly when reading in waveforms!).
    The ``lh5_obj`` that is read by this class is reused in order to avoid
    reallocation of memory; this means that if you want to hold on to data
    between reads, you will have to copy it somewhere!
    """

    def __init__(
        self,
        lh5_files: str | list[str],
        group: str,
        base_path: str = "",
        entry_list: list[int] | list[list[int]] = None,
        entry_mask: list[bool] | list[list[bool]] = None,
        field_mask: dict[str, bool] | list[str] | tuple[str] = None,
        buffer_len: int = 3200,
    ) -> None:
        """
        Parameters
        ----------
        lh5_files
            file or files to read from. May include wildcards and environment
            variables.
        group
            HDF5 group to read.
        base_path
            HDF5 path to prepend.
        entry_list
            list of entry numbers to read. If a nested list is provided,
            expect one top-level list for each file, containing a list of
            local entries. If a list of ints is provided, use global entries.
        entry_mask
            mask of entries to read. If a list of arrays is provided, expect
            one for each file. Ignore if a selection list is provided.
        field_mask
            mask of which fields to read. See :meth:`LH5Store.read_object` for
            more details.
        buffer_len
            number of entries to read at a time while iterating through files.
        """
        self.lh5_st = LH5Store(base_path=base_path, keep_open=True)

        # List of files, with wildcards and env vars expanded
        if isinstance(lh5_files, str):
            lh5_files = utils.expand_path(lh5_files, True)
        elif not isinstance(lh5_files, list):
            raise ValueError("lh5_files must be a string or list of strings")

        if entry_list is not None and entry_mask is not None:
            raise ValueError(
                "entry_list and entry_mask arguments are mutually exclusive"
            )

        self.lh5_files = [
            f for f_wc in lh5_files for f in utils.expand_path(f_wc, True)
        ]

        # Map to last row in each file
        self.file_map = np.array(
            [self.lh5_st.read_n_rows(group, f) for f in self.lh5_files], "int64"
        ).cumsum()
        self.group = group
        self.buffer_len = buffer_len

        if len(self.lh5_files) > 0:
            self.lh5_buffer = self.lh5_st.get_buffer(
                self.group,
                self.lh5_files[0],
                size=self.buffer_len,
                field_mask=field_mask,
            )
        else:
            raise RuntimeError(f"can't open any files from {lh5_files}")

        self.n_rows = 0
        self.current_entry = 0

        self.field_mask = field_mask

        # List of entry indices from each file
        self.entry_list = None
        if entry_list is not None:
            entry_list = list(entry_list)
            if isinstance(entry_list[0], int):
                entry_list.sort()
                i_start = 0
                self.entry_list = []
                for f_end in self.file_map:
                    i_stop = bisect_right(entry_list, f_end, lo=i_start)
                    self.entry_list.append(entry_list[i_start:i_stop])
                    i_start = i_stop

            else:
                self.entry_list = [[]] * len(self.file_map)
                for i_file, local_list in enumerate(entry_list):
                    self.entry_list[i_file] = list(local_list)

        elif entry_mask is not None:
            # Convert entry mask into an entry list
            if isinstance(entry_mask, pd.Series):
                entry_mask = entry_mask.values
            if isinstance(entry_mask, np.ndarray):
                self.entry_list = []
                f_start = 0
                for f_end in self.file_map:
                    self.entry_list.append(
                        list(np.nonzero(entry_mask[f_start:f_end])[0])
                    )
                    f_start = f_end
            else:
                self.entry_list = [[]] * len(self.file_map)
                for i_file, local_mask in enumerate(entry_mask):
                    self.entry_list[i_file] = list(np.nonzero(local_mask)[0])

        # Map to last entry of each file
        self.entry_map = (
            self.file_map
            if self.entry_list is None
            else np.array([len(elist) for elist in self.entry_list]).cumsum()
        )

    def read(self, entry: int) -> tuple[LGDO, int]:
        """Read the next chunk of events, starting at entry. Return the
        LH5 buffer and number of rows read."""
        i_file = np.searchsorted(self.entry_map, entry, "right")
        local_entry = entry
        if i_file > 0:
            local_entry -= self.entry_map[i_file - 1]
        self.n_rows = 0

        while self.n_rows < self.buffer_len and i_file < len(self.file_map):
            # Loop through files
            local_idx = self.entry_list[i_file] if self.entry_list is not None else None
            i_local = local_idx[local_entry] if local_idx is not None else local_entry
            self.lh5_buffer, n_rows = self.lh5_st.read_object(
                self.group,
                self.lh5_files[i_file],
                start_row=i_local,
                n_rows=self.buffer_len - self.n_rows,
                idx=local_idx,
                field_mask=self.field_mask,
                obj_buf=self.lh5_buffer,
                obj_buf_start=self.n_rows,
            )

            self.n_rows += n_rows
            i_file += 1
            local_entry = 0

        self.current_entry = entry
        return (self.lh5_buffer, self.n_rows)

    def __len__(self) -> int:
        """Return the total number of entries."""
        return self.entry_map[-1] if len(self.entry_map) > 0 else 0

    def __iter__(self) -> tuple[LGDO, int, int]:
        """Loop through entries in blocks of size buffer_len."""
        entry = 0
        while entry < len(self):
            buf, n_rows = self.read(entry)
            yield (buf, entry, n_rows)
            entry += n_rows
