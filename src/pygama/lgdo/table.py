"""
Implements a LEGEND Data Object representing a special struct of arrays of
equal length and corresponding utilities.
"""
from __future__ import annotations

import logging
from typing import Any, Union

import pandas as pd

from pygama.lgdo.array import Array
from pygama.lgdo.scalar import Scalar
from pygama.lgdo.struct import Struct
from pygama.lgdo.vectorofvectors import VectorOfVectors

LGDO = Union[Scalar, Struct, Array, VectorOfVectors]

log = logging.getLogger(__name__)


class Table(Struct):
    """A special struct of arrays or subtable columns of equal length.

    Holds onto an internal read/write location ``loc`` that is useful in
    managing table I/O using functions like :meth:`push_row`, :meth:`is_full`,
    and :meth:`clear`.

    Note
    ----
    If you write to a table and don't fill it up to its total size, be sure to
    resize it before passing to data processing functions, as they will call
    :meth:`__len__` to access valid data, which returns the ``size`` attribute.
    """
    # TODO: overload getattr to allow access to fields as object attributes?

    def __init__(self, size: int = None, col_dict: dict[str, LGDO] = {},
                 attrs: dict[str, Any] = {}) -> None:
        r"""
        Parameters
        ----------
        size
            sets the number of rows in the table. :class:`~.Array`\ s in
            `col_dict will be resized to match size if both are not ``None``.
            If `size` is left as ``None``, the number of table rows is
            determined from the length of the first array in `col_dict`. If
            neither is provided, a default length of 1024 is used.
        col_dict
            instantiate this table using the supplied named array-like LGDO's.
            Note 1: no copy is performed, the objects are used directly.
            Note 2: if `size` is not ``None``, all arrays will be resized to
            match it.  Note 3: if the arrays have different lengths, all will
            be resized to match the length of the first array.
        attrs
            A set of user attributes to be carried along with this LGDO.

        Notes
        -----
        the :attr:`loc` attribute is initialized to 0.
        """
        super().__init__(obj_dict=col_dict, attrs=attrs)

        # if col_dict is not empty, set size according to it
        # if size is also supplied, resize all fields to match it
        # otherwise, warn if the supplied fields have varying size
        if len(col_dict) > 0:
            do_warn = True if size is None else False
            self.resize(new_size=size, do_warn=do_warn)

        # if no col_dict, just set the size (default to 1024)
        else:
            self.size = size if size is not None else 1024

        # always start at loc=0
        self.loc = 0

    def datatype_name(self) -> str:
        """The name for this LGDO's datatype attribute."""
        return 'table'

    def __len__(self) -> int:
        """Provides ``__len__`` for this array-like class."""
        return self.size

    def resize(self, new_size: int = None, do_warn: bool = False) -> None:
        # if new_size = None, use the size from the first field
        for field, obj in self.items():
            if new_size is None:
                new_size = len(obj)
            elif len(obj) != new_size:
                if do_warn:
                    log.warning(f'warning: resizing field {field}',
                                f'with size {len(obj)} != {new_size}')
                if isinstance(obj, Table):
                    obj.resize(new_size)
                else:
                    obj.resize(new_size)
        self.size = new_size

    def push_row(self) -> None:
        self.loc += 1

    def is_full(self) -> bool:
        return self.loc >= self.size

    def clear(self) -> None:
        self.loc = 0

    def add_field(self, name: str, obj: LGDO,
                  use_obj_size: bool = False, do_warn=True) -> None:
        """Add a field (column) to the table.

        Use the name "field" here to match the terminology used in
        :class:`.Struct`.

        Parameters
        ----------
        name
            the name for the field in the table.
        obj
            the object to be added to the table.
        use_obj_size
            if ``True``, resize the table to match the length of `obj`.
        do_warn
            print or don't print useful info. Passed to :meth:`resize` when
            `use_obj_size` is ``True``.
        """
        if not hasattr(obj, '__len__'):
            raise TypeError('cannot add field of type', type(obj).__name__)

        super().add_field(name, obj)

        # check / update sizes
        if self.size != len(obj):
            new_size = len(obj) if use_obj_size else self.size
            self.resize(new_size=new_size)

    def add_column(self, name: str, obj: LGDO, use_obj_size: bool = False,
                   do_warn: bool = True) -> None:
        """Alias for :meth:`.add_field` using table terminology 'column'."""
        self.add_field(name, obj, use_obj_size=use_obj_size, do_warn=do_warn)

    def join(self, other_table: Table, cols: list[str] = None,
             do_warn: bool = True) -> None:
        """Add the columns of another table to this table.

        Notes
        -----
        Following the join, both tables have access to `other_table`'s fields
        (but `other_table` doesn't have access to this table's fields). No
        memory is allocated in this process. `other_table` can go out of scope
        and this table will retain access to the joined data.

        Parameters
        ----------
        other_table
            the table whose columns are to be joined into this table.
        cols
            a list of names of columns from `other_table` to be joined into
            this table.
        do_warn
            set to ``False`` to turn off warnings associated with mismatched
            `loc` parameter or :meth:`add_column` warnings.
        """
        if other_table.loc != self.loc and do_warn:
            log.warning(f"other_table.loc ({other_table.loc}) != self.loc({self.loc})")
        if cols is None:
            cols = other_table.keys()
        for name in cols:
            self.add_column(name, other_table[name], do_warn=do_warn)

    def get_dataframe(self, cols: list[str] = None,
                      copy: bool = False) -> pd.DataFrame:
        """Get a :class:`pandas.DataFrame` from the data in the table.

        Notes
        -----
        The requested data must be array-like, with the ``nda`` attribute.

        Parameters
        ----------
        cols
            a list of column names specifying the subset of the table's columns
            to be added to the dataframe.
        copy
            When ``True``, the dataframe allocates new memory and copies data
            into it. Otherwise, the raw ``nda``'s from the table are used directly.
        """
        df = pd.DataFrame(copy=copy)
        if cols is None:
            cols = self.keys()
        for col in cols:
            if not hasattr(self[col], 'nda'):
                log.warning(f'column {col} does not have an nda, skipping...')
            else:
                df[col] = self[col].nda

        return df
