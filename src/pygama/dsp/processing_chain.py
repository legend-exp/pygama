"""
This module provides routines for setting up and running signal processing
chains on waveform data.
"""
from __future__ import annotations

import ast
import importlib
import itertools as it
import json
import logging
import re
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from numba import vectorize

import pygama.lgdo as lgdo
from pygama.dsp.errors import DSPFatal, ProcessingChainError
from pygama.lgdo.lgdo_utils import expand_path
from pygama.math.units import Quantity, Unit
from pygama.math.units import unit_registry as ureg

log = logging.getLogger(__name__)

LGDO = Union[lgdo.Scalar, lgdo.Array, lgdo.VectorOfVectors, lgdo.Struct]

# Filler value for variables to be automatically deduced later
auto = "auto"

# Map from ast interpreter operations to functions to call and format string
ast_ops_dict = {
    ast.Add: (np.add, "{}+{}"),
    ast.Sub: (np.subtract, "{}-{}"),
    ast.Mult: (np.multiply, "{}*{}"),
    ast.Div: (np.divide, "{}/{}"),
    ast.FloorDiv: (np.floor_divide, "{}//{}"),
    ast.USub: (np.negative, "-{}"),
}


@dataclass
class CoordinateGrid:
    """Helper class that describes a system of units, consisting of a period
    and offset.

    `period` is a unitted :class:`pint.Quantity`, `offset` is a scalar in units
    of `period`, a :class:`pint.Unit` or a :class:`ProcChainVar`. In the last
    case, a :class:`ProcChainVar` variable is used to store a different offset
    for each event.
    """

    period: Quantity | Unit | str
    offset: Quantity | ProcChainVar | float | int

    def __post_init__(self) -> None:
        if isinstance(self.period, str):
            self.period = Quantity(1.0, self.period)
        elif isinstance(self.period, Unit):
            self.period *= 1  # make Quantity
        if isinstance(self.offset, (int, float)):
            self.offset = self.offset * self.period
        assert isinstance(self.period, Quantity) and isinstance(
            self.offset, (Quantity, ProcChainVar)
        )

    def __eq__(self, other: CoordinateGrid) -> bool:
        """True if values are equal; if offset is a variable, compares reference"""
        return self.period == other.period and (
            self.offset is other.offset
            if isinstance(self.offset, ProcChainVar)
            else self.offset == other.offset
        )

    def unit_str(self) -> str:
        string = format(self.period.u, "~")
        if string == "":
            string = str(self.period.u)
        return string

    def get_period(self, unit: str | Unit) -> float:
        if isinstance(unit, str):
            unit = ureg.Quantity(unit)
        return float(self.period / unit)

    def get_offset(self, unit: str | Unit = None) -> float:
        """Get the offset (convert)ed to unit. If `unit` is ``None`` use period."""
        if unit is None:
            unit = self.period
        elif isinstance(unit, str):
            unit = ureg.Quantity(unit)

        if isinstance(self.offset, ProcChainVar):
            return self.offset.get_buffer(CoordinateGrid(unit, 0))
        else:
            return float(self.offset / unit)

    def __str__(self) -> str:
        offset = (
            self.offset.name
            if isinstance(self.offset, ProcChainVar)
            else str(self.offset)
        )
        return f"({str(self.period)},{offset})"


class ProcChainVar:
    """Helper data class with buffer and information for internal variables in
    :class:`ProcessingChain`.

    Members can be set to ``auto`` to attempt to deduce these when adding this
    variable to a processor for the first time.
    """

    def __init__(
        self,
        proc_chain: ProcessingChain,
        name: str,
        shape: int | tuple[int, ...] = auto,
        dtype: np.dtype = auto,
        grid: CoordinateGrid = auto,
        unit: str | Unit = auto,
        is_coord: bool = auto,
    ) -> None:
        """
        Parameters
        ----------
        proc_chain
            ProcessingChain that contains this variable
        name
            Name of variable used to look it up
        shape
            Shape of variable, without buffer_len dimension
        dtype
            Data type of variable
        grid
            Coordinate grid associated with variable. This contains the
            period and offset of the variable. For variables where
            is_coord is True, use this to perform unit conversions
        unit
            Unit associated with variable during I/O.
        is_coord
            If True, variable represents an array index and can be converted
            into a unitted number using grid
        """
        assert isinstance(proc_chain, ProcessingChain) and isinstance(name, str)
        self.proc_chain = proc_chain
        self.name = name

        # ndarray containing data buffer of size block_width x shape
        # list of ndarrays in different coordinate systems if is_coord is true
        self._buffer: list | np.ndarray = None

        self.shape = shape
        self.dtype = dtype
        self.grid = grid
        self.unit = unit
        self.is_coord = is_coord

        log.debug(f"added variable: {self.description()}")

    # Use this to enforce type constraints and perform conversions
    def __setattr__(self, name: str, value: Any) -> None:
        if value is auto:
            pass

        elif name == "shape":
            if hasattr(value, "__iter__"):
                value = tuple(value)
            else:
                value = (int(value),)
            value = tuple(value)
            assert all(isinstance(d, int) for d in value)

        elif name == "dtype" and not isinstance(value, np.dtype):
            value = np.dtype(value)

        elif (
            name == "grid"
            and not isinstance(value, CoordinateGrid)
            and value is not None
        ):
            period, offset = value if isinstance(value, tuple) else value, 0
            value = CoordinateGrid(period, offset)

        elif name == "unit" and value is not None:
            value = str(value)

        elif name == "is_coord":
            value = bool(value)
            if value:
                if self._buffer is None:
                    self._buffer = []
                elif isinstance(self._buffer, np.ndarray):
                    self._buffer = [(self._buffer, CoordinateGrid(self.unit, 0))]

        super().__setattr__(name, value)

    def get_buffer(self, unit: str | Unit = None) -> np.ndarray:
        # If buffer needs to be created, do so now
        if self._buffer is None:
            if self.shape is auto:
                raise ProcessingChainError(f"cannot deduce shape of {self.name}")
            if self.dtype is auto:
                raise ProcessingChainError(f"cannot deduce shape of {self.name}")

            # create the buffer so that the array start is aligned in memory on a multiple of 64 bytes
            self._buffer = np.zeros(
                shape=(self.proc_chain._block_width,) + self.shape, dtype=self.dtype
            )

        # if variable isn't a coordinate, we're all set
        if self.is_coord is False or self.is_coord is auto:
            return self._buffer

        # if no unit is given, use the native unit
        if unit is None:
            if isinstance(self.unit, str):
                unit = CoordinateGrid(self.unit, 0.0)
        elif not isinstance(unit, CoordinateGrid):
            unit = CoordinateGrid(unit, 0.0)

        # if this is our first time accessing, no conversion is needed
        if len(self._buffer) == 0:
            if self.shape is auto:
                raise ProcessingChainError(f"cannot deduce shape of {self.name}")
            if self.dtype is auto:
                raise ProcessingChainError(f"cannot deduce shape of {self.name}")

            buff = np.zeros(
                shape=(self.proc_chain._block_width,) + self.shape, dtype=self.dtype
            )
            self._buffer.append((buff, unit))
            return buff

        # check if coordinate conversion has been done already
        for buff, grid in self._buffer:
            if grid == unit:
                return buff

        # If we get this far, add conversion processor to ProcChain and add new buffer to _buffer
        conversion_manager = UnitConversionManager(self, unit)
        self._buffer.append([conversion_manager.out_buffer, unit])
        self.proc_chain._proc_managers.append(conversion_manager)
        return conversion_manager.out_buffer

    @property
    def buffer(self):
        return self.get_buffer()

    @property
    def period(self):
        return self.grid.period if self.grid else None

    @property
    def offset(self):
        return self.grid.offset if self.grid else None

    def description(self) -> str:
        return (
            f"{self.name}(shape: {self.shape}, "
            f"dtype: {self.dtype}, grid: {self.grid}, "
            f"unit: {self.unit}, is_coord: {self.is_coord})"
        )

    def update_auto(
        self,
        shape: int | tuple[int, ...] = auto,
        dtype: np.dtype = auto,
        grid: CoordinateGrid = auto,
        unit: str | Unit = auto,
        is_coord: bool = auto,
        period: period = None,
        offset: offset = 0,
    ) -> None:
        """Update any variables set to ``auto``; leave the others alone. Emit a
        message only if anything was updated.
        """
        updated = False

        # Construct coordinate grid from period/offset if given
        if grid is auto and period is not None:
            if isinstance(offset, str):
                offset = self.get_variable(offset, expr_only=True)
            grid = CoordinateGrid(period, offset)

        if self.shape is auto and shape is not auto:
            self.shape = shape
            updated = True
        if self.dtype is auto and dtype is not auto:
            self.dtype = dtype
            updated = True
        if self.grid is auto and grid is not auto:
            self.grid = grid
            updated = True
        if self.unit is auto and unit is not auto:
            self.unit = unit
            updated = True
        if self.is_coord is auto and is_coord is not auto:
            self.is_coord = is_coord
            updated = True
        if updated:
            log.debug(f"updated variable: {self.description()}")

    def __str__(self) -> str:
        return self.name


class ProcessingChain:
    """A class to efficiently perform a sequence of digital signal processing
    (DSP) transforms.

    It contains a list of DSP functions and a set of constant values and named
    variables contained in fixed memory locations. When executing the
    :class:`ProcessingChain`, processors will act on the internal memory
    without allocating new memory in the process. Furthermore, the memory is
    allocated in blocks, enabling vectorized processing of many entries at
    once. To set up a :class:`ProcessingChain`, use the following methods:

    - :meth:`.link_input_buffer` bind a named variable to an external NumPy
      array to read data from
    - :meth:`.add_processor` add a dsp function and bind its inputs to a set of
      named variables and constant values
    - :meth:`.link_output_buffer` bind a named variable to an external NumPy
      array to write data into

    When calling these methods, the :class:`ProcessingChain` class will use
    available information to allocate buffers to the correct sizes and data
    types. For this reason, transforms will ideally implement the
    :class:`numpy.ufunc` class, enabling broadcasting of array dimensions. If
    not enough information is available to correctly allocate memory, it can be
    provided through the named variable strings or by calling add_vector or
    add_scalar.
    """

    def __init__(self, block_width: int = 8, buffer_len: int = None) -> None:
        """
        Parameters
        ----------
        block_width
            number of entries to simultaneously process.
        buffer_len
            length of input and output buffers. Should be a multiple of
            `block_width`
        """
        # Dictionary from name to scratch data buffers as ProcChainVar
        self._vars_dict = {}
        # list of processors with variables they are called on
        self._proc_managers = []
        # lists of I/O managers that handle copying data to/from external memory buffers
        self._input_managers = []
        self._output_managers = []

        self._block_width = block_width
        self._buffer_len = buffer_len

    def add_variable(
        self,
        name: str,
        dtype: np.dtype | str = auto,
        shape: int | tuple[int, ...] = auto,
        grid: CoordinateGrid = auto,
        unit: str | Unit = auto,
        is_coord: bool = auto,
        period: CoordinateGrid.period = None,
        offset: CoordinateGrid.offset = 0,
    ) -> ProcChainVar:
        """Add a named variable containing a block of values or arrays.

        Parameters
        ----------
        name : str
            name of variable
        dtype : numpy.dtype or str, optional, default='auto'
            default is ``None``, meaning `dtype` will be deduced later, if
            possible
        shape : int or tuple, optional, default='auto'
            length or shape tuple of element. Default is ``None``, meaning length
            will be deduced later, if possible
        grid : CoordinateGrid
            for variable, containing period and offset
        unit
            unit of variable
        period
            unit with period of waveform associated with object. Do not use if
            `grid` is provided
        offset
            unit with offset of waveform associated with object. Requires a
            `period` to be provided
        is_coord : bool
            if ``True``, transform value based on `period` and `offset`
        """
        self._validate_name(name, raise_exception=True)
        if name in self._vars_dict:
            raise ProcessingChainError(name + " is already in variable list")

        # Construct coordinate grid from period/offset if given
        if grid is auto and period is not None:
            if isinstance(offset, str):
                offset = self.get_variable(offset, expr_only=True)
            grid = CoordinateGrid(period, offset)

        var = ProcChainVar(
            self,
            name,
            shape=shape,
            dtype=dtype,
            grid=grid,
            unit=unit,
            is_coord=is_coord,
        )
        self._vars_dict[name] = var
        return var

    def link_input_buffer(
        self, varname: str, buff: np.ndarray | LGDO = None
    ) -> np.ndarray | LGDO:
        """Link an input buffer to a variable.

        Parameters
        ----------
        varname
            name of internal variable to copy into buffer at the end
            of processor execution. If variable does not yet exist, it will
            be created with a similar shape to the provided buffer.
        buff
            object to use as input buffer. If ``None``, create a new buffer
            with a similar shape to the variable.

        Returns
        -------
        buffer
            `buff` or newly allocated input buffer.
        """
        self._validate_name(varname, raise_exception=True)
        var = self.get_variable(varname, expr_only=True)
        if var is None:
            var = self.add_variable(varname)

        if not isinstance(var, ProcChainVar):
            raise ProcessingChainError(
                "Must link an input buffer to a processing chain variable"
            )

        # Create input buffer that will be linked and returned if none exists
        if buff is None:
            dtype = var.get_buffer().dtype

            if var is None:
                raise ProcessingChainError(
                    f"{varname} does not exist and no buffer was provided"
                )
            elif isinstance(var.grid, CoordinateGrid) and len(var.shape) == 1:
                buff = lgdo.WaveformTable(
                    size=self._buffer_len, wf_len=var.shape[0], dtype=dtype
                )
            elif len(var.shape) == 0:
                buff = lgdo.Array(shape=(self._buffer_len), dtype=dtype)
            elif len(var.shape) > 0:
                buff = lgdo.ArrayOfEqualSizedArrays(
                    shape=(self._buffer_len, *var.shape), dtype=dtype
                )
            else:
                buff = np.ndarray((self._buffer_len,) + var.shape, dtype)

        # Add the buffer to the input buffers list
        if isinstance(buff, np.ndarray):
            out_man = NumpyIOManager(buff, var)
        elif isinstance(buff, lgdo.ArrayOfEqualSizedArrays):
            out_man = LGDOArrayOfEqualSizedArraysIOManager(buff, var)
        elif isinstance(buff, lgdo.Array):
            out_man = LGDOArrayIOManager(buff, var)
        elif isinstance(buff, lgdo.WaveformTable):
            out_man = LGDOWaveformIOManager(buff, var)
        else:
            raise ProcessingChainError(
                "Could not link input buffer of unknown type", str(buff)
            )

        log.debug(f"added input buffer: {out_man}")
        self._input_managers.append(out_man)

        return buff

    def link_output_buffer(
        self, varname: str, buff: np.ndarray | LGDO = None
    ) -> np.ndarray | LGDO:
        """Link an output buffer to a variable.

        Parameters
        ----------
        varname
            name of internal variable to copy into buffer at the end of
            processor execution. If variable does not yet exist, it will be
            created with a similar shape to the provided buffer.
        buff
            object to use as output buffer. If ``None``, create a new buffer
            with a similar shape to the variable

        Returns
        -------
        buffer
            `buff` or newly allocated output buffer.
        """
        self._validate_name(varname, raise_exception=True)
        var = self.get_variable(varname, expr_only=True)
        if var is None:
            var = self.add_variable(varname)

        if not isinstance(var, ProcChainVar):
            raise ProcessingChainError(
                "must link an output buffer to a processing chain variable"
            )

        # Create output buffer that will be linked and returned if none exists
        if buff is None:
            dtype = var.get_buffer().dtype

            if var is None:
                raise ProcessingChainError(
                    varname + " does not exist and no buffer was provided"
                )
            elif isinstance(var.grid, CoordinateGrid) and len(var.shape) == 1:
                buff = lgdo.WaveformTable(
                    size=self._buffer_len, wf_len=var.shape[0], dtype=dtype
                )
            elif len(var.shape) == 0:
                buff = lgdo.Array(shape=(self._buffer_len), dtype=dtype)
            elif len(var.shape) > 0:
                buff = lgdo.ArrayOfEqualSizedArrays(
                    shape=(self._buffer_len, *var.shape), dtype=dtype
                )
            else:
                buff = np.ndarray((self._buffer_len,) + var.shape, dtype)

        # Add the buffer to the output buffers list
        if isinstance(buff, np.ndarray):
            out_man = NumpyIOManager(buff, var)
        elif isinstance(buff, lgdo.ArrayOfEqualSizedArrays):
            out_man = LGDOArrayOfEqualSizedArraysIOManager(buff, var)
        elif isinstance(buff, lgdo.Array):
            out_man = LGDOArrayIOManager(buff, var)
        elif isinstance(buff, lgdo.WaveformTable):
            out_man = LGDOWaveformIOManager(buff, var)
        else:
            raise ProcessingChainError(
                "could not link output buffer of unknown type", str(buff)
            )

        log.debug(f"added output buffer: {out_man}")
        self._output_managers.append(out_man)

        return buff

    def add_processor(
        self, func: np.ufunc, *args, signature: str = None, types: list[str] = None
    ) -> None:
        """Make a list of parameters from `*args`. Replace any strings in the
        list with NumPy objects from `vars_dict`, where able.
        """
        params = []
        kw_params = {}
        for _, param in enumerate(args):
            if isinstance(param, str):
                param = self.get_variable(param)
            if isinstance(param, dict):
                kw_params.update(param)
            else:
                params.append(param)

        proc_man = ProcessorManager(self, func, params, kw_params, signature, types)
        self._proc_managers.append(proc_man)

    def execute(self, start: int = 0, stop: int = None) -> None:
        """Execute the dsp chain on the entire input/output buffers."""
        if stop is None:
            stop = self._buffer_len
        for i in range(start, stop, self._block_width):
            self._execute_procs(i, min(i + self._block_width, self._buffer_len))

    def get_variable(
        self, expr: str, get_names_only: bool = False, expr_only: bool = False
    ) -> Any:
        r"""Parse string `expr` into a NumPy array or value, using the following
        syntax:

        - numeric values are parsed into ``int``\ s or ``float``\ s
        - units found in the :mod:`pint` package
        - other strings are parsed into variable names. If `get_names_only` is
          ``False``, fetch the internal buffer (creating it as needed). Else,
          return a string of the name
        - if a string is followed by ``(...)``, try parsing into one of the
          following expressions:

          - ``len(expr)``: return the length of the array found with `expr`
          - ``round(expr)``: return the value found with `expr` to the
            nearest integer
          - ``varname(shape, type)``: allocate a new buffer with the
            specified shape and type, using ``varname``. This is used if
            the automatic type and shape deduction for allocating variables
            fails

        - Unary and binary operators :obj:`+`, :obj:`-`, :obj:`*`, :obj:`/`,
          :obj:`//` are available. If a variable name is included in the
          expression, a processor will be added to the
          :class:`ProcessingChain` and a new buffer allocated to store the
          output
        - ``varname[slice]``: return the variable with a slice applied. Slice
          values can be ``float``\ s, and will have round applied to them
        - ``keyword = expr``: return a ``dict`` with a single element
          pointing from keyword to the parsed `expr`. This is used for
          `kwargs`. If `expr_only` is ``True``, raise an exception if we see
          this.

        If `get_names_only` is set to ``True``, do not fetch or allocate new
        arrays, instead return a list of variable names found in the expression.
        """
        names = []
        try:
            stmt = ast.parse(expr).body[0]
            var = self._parse_expr(stmt.value, expr, get_names_only, names)
        except Exception as e:
            raise ProcessingChainError("Could not parse expression:\n  " + expr) from e

        # Check if this is an arg (i.e. expr) or kwarg (i.e. assign)
        if not get_names_only:
            if isinstance(stmt, ast.Expr):
                return var
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                if expr_only:
                    raise ProcessingChainError(
                        "kwarg assignment is not allowed in this context\n  " + expr
                    )
                return {stmt.targets[0].id: var}
            else:
                raise ProcessingChainError("Could not parse expression:\n  " + expr)
        else:
            return names

    def _parse_expr(
        self, node: Any, expr: str, dry_run: bool, var_name_list: list[str]
    ) -> Any:
        """
        Helper function for :meth:`ProcessingChain.get_variable` that
        recursively evaluates the AST tree. Whenever we encounter a variable
        name, add it to `var_name_list` (which should begin as an empty list).
        Only add new variables and processors to the chain if `dry_run` is
        ``True``. Based on `this Stackoverflow
        answer <https://stackoverflow.com/a/9558001>`_.
        """
        if node is None:
            return None

        elif isinstance(node, ast.List):
            npparr = np.array(ast.literal_eval(expr))
            if len(npparr.shape) == 1:
                return npparr
            else:
                ProcessingChainError("only 1D arrays are supported: " + expr)

        elif isinstance(node, ast.Num):
            return node.n

        elif isinstance(node, ast.Str):
            return node.s

        elif isinstance(node, ast.Constant):
            return node.value

        # look for name in variable dictionary
        elif isinstance(node, ast.Name):
            # check if it is a unit
            if node.id in ureg:
                return ureg(node.id)

            # check if it is a variable
            var_name_list.append(node.id)
            if dry_run:
                return None

            val = self._vars_dict.get(node.id, None)
            if val is None:
                val = self.add_variable(node.id)
            return val

        # define binary operators (+,-,*,/)
        elif isinstance(node, ast.BinOp):
            lhs = self._parse_expr(node.left, expr, dry_run, var_name_list)
            rhs = self._parse_expr(node.right, expr, dry_run, var_name_list)
            if rhs is None or lhs is None:
                return None
            op, op_form = ast_ops_dict[type(node.op)]

            if not (isinstance(lhs, ProcChainVar) or isinstance(rhs, ProcChainVar)):
                return op(lhs, rhs)

            name = "(" + op_form.format(str(lhs), str(rhs)) + ")"
            if isinstance(lhs, ProcChainVar) and isinstance(rhs, ProcChainVar):
                # TODO: handle units/coords; for now make them match lhs
                out = ProcChainVar(self, name, is_coord=lhs.is_coord)
            elif isinstance(lhs, ProcChainVar):
                out = ProcChainVar(
                    self,
                    name,
                    lhs.shape,
                    lhs.dtype,
                    lhs.grid,
                    lhs.unit,
                    is_coord=lhs.is_coord,
                )
            else:
                out = ProcChainVar(
                    self,
                    name,
                    rhs.shape,
                    rhs.dtype,
                    rhs.grid,
                    rhs.unit,
                    is_coord=rhs.is_coord,
                )

            self._proc_managers.append(ProcessorManager(self, op, [lhs, rhs, out]))
            return out

        # define unary operators (-)
        elif isinstance(node, ast.UnaryOp):
            operand = self._parse_expr(node.operand, expr, dry_run, var_name_list)
            if operand is None:
                return None
            op, op_form = ast_ops_dict[type(node.op)]
            name = "(" + op_form.format(str(operand)) + ")"

            if isinstance(operand, ProcChainVar):
                out = ProcChainVar(
                    self,
                    name,
                    operand.shape,
                    operand.dtype,
                    operand.grid,
                    operand.unit,
                    operand.is_coord,
                )
                self._proc_managers.append(ProcessorManager(self, op, [operand, out]))
            else:
                out = op(out)

            return out

        elif isinstance(node, ast.Subscript):
            val = self._parse_expr(node.value, expr, dry_run, var_name_list)
            if val is None:
                return None
            if not isinstance(val, ProcChainVar):
                raise ProcessingChainError("Cannot apply subscript to", node.value)

            def get_index(slice_value):
                ret = self._parse_expr(slice_value, expr, dry_run, var_name_list)
                if ret is None:
                    return ret
                if isinstance(ret, Quantity):
                    ret = float(ret / val.period)
                if isinstance(ret, float):
                    round_ret = int(round(ret))
                    if abs(ret - round_ret) > 0.0001:
                        log.warning(
                            f"slice value {slice_value} is non-integer. Rounding to {round_ret}"
                        )
                    return round_ret
                return int(ret)

            if isinstance(node.slice, ast.Index):
                index = get_index(node.slice.value)
                out_buf = val[..., index]
                out_name = (f"{str(val)}[{index}]",)
                out_grid = None

            elif isinstance(node.slice, ast.Slice):
                sl = slice(
                    get_index(node.slice.lower),
                    get_index(node.slice.upper),
                    get_index(node.slice.step),
                )
                out_buf = val.buffer[..., sl]
                out_name = "{}[{}:{}{}]".format(
                    str(val),
                    "" if sl.start is None else str(sl.start),
                    "" if sl.stop is None else str(sl.stop),
                    "" if sl.step is None else ":" + str(sl.step),
                )

                if val.grid is None:
                    out_grid = None
                else:
                    pd = val.period
                    if sl.step is not None:
                        pd *= sl.step

                    off = val.offset
                    if sl.start is not None:
                        start = sl.start * val.period
                        if isinstance(off, ProcChainVar):
                            new_off = ProcChainVar(
                                self, name=f"({str(off)}+{str(start)})", is_coord=True
                            )
                            self._proc_managers.append(
                                ProcessorManager(self, np.add, [off, start, new_off])
                            )
                            off = new_off
                        else:
                            off += start
                    out_grid = CoordinateGrid(pd, off)

            elif isinstance(node.slice, ast.ExtSlice):
                # TODO: implement this...
                raise ProcessingChainError("ExtSlice still isn't implemented...")

            # Create our return variable and set the buffer to the slice
            out = ProcChainVar(
                self,
                out_name,
                shape=out_buf.shape[1:],
                dtype=out_buf.dtype,
                grid=out_grid,
                unit=val.unit,
                is_coord=val.is_coord,
            )
            out._buffer = [(out_buf, val._buffer[0][1])] if out.is_coord else out_buf
            return out

        # for name.attribute
        elif isinstance(node, ast.Attribute):
            # If we are looking for an attribute of a module (e.g. np.pi)
            if node.value.id in self.module_list:
                mod = self.module_list[node.value.id]
                attr = getattr(mod, node.attr)
                if not isinstance(attr, (int, float)):
                    raise ProcessingChainError(
                        f"Attribute {node.attr} from {node.value} is not"
                        f"an int or float..."
                    )
                return attr

            # Otherwise this is probably a ProcChainVar
            val = self._parse_expr(node.value, expr, dry_run, var_name_list)
            if val is None:
                return None
            return getattr(val, node.attr)

        # for func(args, kwargs)
        elif isinstance(node, ast.Call):
            func = self.func_list.get(node.func.id, None)
            args = [
                self._parse_expr(arg, expr, dry_run, var_name_list) for arg in node.args
            ]
            kwargs = {
                kwarg.arg: self._parse_expr(kwarg.value, expr, dry_run, var_name_list)
                for kwarg in node.keywords
            }
            if func is not None:
                return func(*args, **kwargs)
            elif self._validate_name(node.func.id):
                var_name = node.func.id
                var_name_list.append(var_name)
                if var_name in self._vars_dict:
                    var = self._vars_dict[var_name]
                    var.update_auto(*args, **kwargs)
                    return self._vars_dict[var_name]
                elif not dry_run:
                    return self.add_variable(var_name, *args, **kwargs)
                else:
                    return None

            else:
                raise ProcessingChainError(
                    f"do not recognize call to {func} with arguments "
                    f"{[str(arg.__dict__) for arg in node.args]}"
                )

        raise ProcessingChainError(f"cannot parse AST nodes of type {node.__dict__}")

    def _validate_name(self, name: str, raise_exception: bool = False) -> bool:
        """Check that name is alphanumeric, and not an already used keyword"""
        isgood = (
            re.match(r"\A\w+$", name)
            and name not in self.func_list
            and name not in ureg
            and name not in self.module_list
        )
        if raise_exception and not isgood:
            raise ProcessingChainError(f"{name} is not a valid variable name")
        return isgood

    def _execute_procs(self, begin: int, end: int) -> str:
        """Copy from input buffers to variables, call all the processors on
        their paired arg tuples, copy from variables to list of output buffers.
        """
        # Copy input buffers into proc chain buffers
        for in_man in self._input_managers:
            in_man.read(begin, end)

        # Loop through processors and run each one
        for proc_man in self._proc_managers:
            try:
                proc_man.execute()
            except DSPFatal as e:
                e.processor = str(proc_man)
                e.wf_range = (begin, end)
                raise e

        # copy from processing chain buffers into output buffers
        for out_man in self._output_managers:
            out_man.write(begin, end)

    def __str__(self) -> str:
        return (
            "Input variables:\n  "
            + "\n  ".join([str(in_man) for in_man in self._input_managers])
            + "\nProcessors:\n  "
            + "\n  ".join([str(proc_man) for proc_man in self._proc_managers])
            + "\nOutput variables:\n  "
            + "\n  ".join([str(out_man) for out_man in self._output_managers])
        )

    # Define functions that can be parsed by get_variable
    # Get length of ProcChainVar
    def _length(var: ProcChainVar) -> int:  # noqa: N805
        if var is None:
            return None
        if not isinstance(var, ProcChainVar):
            raise ProcessingChainError(f"cannot call len() on {var}")
        if not len(var.buffer.shape) == 2:
            raise ProcessingChainError(f"{var} has wrong number of dims")
        return var.buffer.shape[1]

    # round value
    def _round(var: ProcChainVar) -> float:  # noqa: N805
        if var is None:
            return None
        if not isinstance(var, ProcChainVar):
            return round(float(var))
        else:
            raise ProcessingChainError(
                "round() is not implemented for variables, only constants."
            )

    # dict of functions that can be parsed by get_variable
    func_list = {"len": _length, "round": _round}
    module_list = {"np": np, "numpy": np}


class ProcessorManager:
    """The class that calls processors and makes sure variables are compatible."""

    @dataclass
    class DimInfo:
        length: int  # length of arrays in this dimension
        grid: CoordinateGrid  # period and offset of arrays in this dimension

    def __init__(
        self,
        proc_chain: ProcessingChain,
        func: np.ufunc,
        params: list[str],
        kw_params: dict = None,
        signature: str = None,
        types: list[str] = None,
    ) -> None:

        assert (
            isinstance(proc_chain, ProcessingChain)
            and callable(func)
            and isinstance(params, list)
        )

        if kw_params is None:
            kw_params = {}

        # reference back to our processing chain
        self.proc_chain = proc_chain
        # callable function used to process data
        self.processor = func
        # list of parameters prior to converting to internal representation
        self.params = params
        # dict of keyword parameters prior to converting to internal rep
        self.kw_params = kw_params
        # list of raw values and buffers from params; we will fill this soon
        self.args = []
        # dict of kws -> raw values and buffers from params; we will fill this soon
        self.kwargs = {}

        # Get the signature and list of valid types for the function
        self.signature = func.signature if signature is None else signature
        if self.signature is None:
            self.signature = (
                ",".join(["()"] * func.nin) + "->" + ",".join(["()"] * func.nout)
            )

        # Get list of allowed type signatures
        if types is None:
            types = func.types.copy()
        if types is None:
            raise ProcessingChainError(
                f"""could not find a type signature list
                                           for {func.__name__}. Please supply a
                                           valid list of types."""
            )
        if not isinstance(types, list):
            types = [types]
        found_types = [typestr.replace("->", "") for typestr in types]

        # Make sure arrays obey the broadcasting rules, and make a dictionary
        # of the correct dimensions and unit system
        dims_list = re.findall(r"\((.*?)\)", self.signature)

        if not len(dims_list) == len(params) + len(kw_params):
            raise ProcessingChainError(
                f"expected {len(dims_list)} arguments from signature "
                f"{self.signature}; found "
                f"{len(params)}: ({', '.join([str(par) for par in params])})"
            )

        dims_dict = {}  # map from dim name -> DimInfo
        outerdims = []  # list of DimInfo
        grid = None  # period/offset to use for unit and coordinate conversions

        for ipar, (dims, param) in enumerate(
            zip(dims_list, it.chain(self.params, self.kw_params.values()))
        ):
            if not isinstance(param, ProcChainVar):
                continue

            # find type signatures that match type of array
            if param.dtype is not auto:
                arr_type = param.dtype.char
                found_types = [
                    type_sig
                    for type_sig in found_types
                    if np.can_cast(arr_type, type_sig[ipar])
                ]

            # fill out dimensions from dim signature and check if it works
            if param.shape is auto:
                continue
            fun_dims = [od for od in outerdims] + [
                d.strip() for d in dims.split(",") if d
            ]
            arr_dims = list(param.shape)
            arr_grid = param.grid if param.grid is not auto else None
            if not grid:
                grid = arr_grid

            # check if arr_dims can be broadcast to match fun_dims
            for i in range(max(len(fun_dims), len(arr_dims))):
                fd = fun_dims[-i - 1] if i < len(fun_dims) else None
                ad = arr_dims[-i - 1] if i < len(arr_dims) else None

                if isinstance(fd, str):
                    if fd in dims_dict:
                        this_dim = dims_dict[fd]
                        if not ad or this_dim.length != ad:
                            raise ProcessingChainError(
                                f"failed to broadcast array dimensions for "
                                f"{func.__name}. Could not find consistent value "
                                f"for dimension {fd}"
                            )
                        if not this_dim.grid:
                            dims_dict[fd].grid = arr_grid
                        elif arr_grid and arr_grid != this_dim.grid:
                            log.debug(
                                f"arrays of dimension {fd} for "
                                f"{func.__name__} do not have consistent period "
                                f"and offset!"
                            )
                    else:
                        dims_dict[fd] = self.DimInfo(ad, arr_grid)

                elif not fd:
                    # if we ran out of function dimensions, add a new outer dim
                    outerdims.insert(0, self.DimInfo(ad, arr_grid))

                elif not ad:
                    continue

                elif fd.length != ad:
                    # If dimensions disagree, either insert a broadcasted array dimension or raise an exception
                    if len(fun_dims) > len(arr_dims):
                        arr_dims.insert(len(arr_dims) - i, 1)
                    elif len(fun_dims) < len(arr_dims):
                        outerdims.insert(len(fun_dims) - i, self.DimInfo(ad, arr_grid))
                        fun_dims.insert(len(fun_dims) - i, ad)
                    else:
                        raise ProcessingChainError(
                            f"failed to broadcast array dimensions for "
                            f"{func.__name__}. Input arrays do not have "
                            f"consistent outer dimensions.  Require: "
                            f"{tuple(dim.length for dim in outerdims+fun_dims)}; "
                            f"found {tuple(arr_dims)} for {param}"
                        )
                elif not fd.grid:
                    outerdims[len(fun_dims) - i].grid = arr_grid

                elif arr_grid and fd.grid != arr_grid:
                    log.debug(
                        f"arrays of dimension {fd} for {func.__name__} "
                        f"do not have consistent period and offset!"
                    )

                arr_grid = None  # this is only used for inner most dim

        # Get the type signature we are using
        if not found_types:
            for param in it.chain(self.params, self.kw_params.values()):
                if not isinstance(param, ProcChainVar):
                    continue
            raise ProcessingChainError(
                f"could not find a type signature matching the types of the "
                f"variables given for {self} (types: {types})"
            )
        # Use the first types in the list that all our types can be cast to
        self.types = [np.dtype(t) for t in found_types[0]]

        # Finish setting up of input parameters for function
        # Iterate through args and then kwargs
        # Reshape variable arrays to add broadcast dimensions
        # Allocate new arrays as needed
        # Convert coords to right system of units as needed
        for _, ((arg_name, param), dims, dtype) in enumerate(
            zip(
                it.chain(zip(it.repeat(None), self.params), self.kw_params.items()),
                dims_list,
                self.types,
            )
        ):
            dim_list = outerdims.copy()
            for d in dims.split(","):
                if not d:
                    continue
                if d not in dims_dict:
                    # If it is an array lets get the length
                    if isinstance(param, np.ndarray):
                        dims_dict[d] = self.DimInfo(len(param), None)
                    else:
                        raise ProcessingChainError(
                            f"could not deduce dimension {d} for {param}"
                        )
                dim_list.append(dims_dict[d])
            shape = tuple(d.length for d in dim_list)
            this_grid = dim_list[-1].grid if dim_list else None

            if isinstance(param, ProcChainVar):
                # Deduce any automated descriptions of parameter
                unit = None
                is_coord = False
                if param.is_coord is True and grid is not None:
                    unit = str(grid.period.u)
                    this_grid = grid
                elif (
                    isinstance(param.unit, str)
                    and param.unit in ureg
                    and grid is not None
                    and ureg.is_compatible_with(grid.period, param.unit)
                ):
                    is_coord = True
                    this_grid = grid

                param.update_auto(
                    shape=shape,
                    dtype=np.dtype(dtype),
                    grid=this_grid,
                    unit=unit,
                    is_coord=is_coord,
                )

                if param.is_coord and not grid:
                    grid = param._buffer[0][1]
                param = param.get_buffer(grid)

                # reshape just in case there are some missing dimensions
                arshape = list(param.shape)
                for idim in range(-1, -1 - len(shape), -1):
                    if arshape[idim] != shape[idim]:
                        arshape.insert(len(arshape) + idim + 1, 1)
                param = param.reshape(tuple(arshape))

            elif isinstance(param, str):
                # Convert string into integer buffer if appropriate
                if np.issubdtype(dtype, np.integer):
                    try:
                        param = np.frombuffer(param.encode("ascii"), dtype).reshape(
                            shape
                        )
                    except (ValueError):
                        raise ProcessingChainError(
                            f"could not convert string '{param}' into"
                            f"byte-array of type {dtype} and shape {shape}"
                        )

            elif param is not None:
                # Convert scalar to right type, including units
                if isinstance(param, (Quantity, Unit)):
                    if ureg.is_compatible_with(ureg.dimensionless, param):
                        param = float(param)
                    elif not isinstance(
                        grid, CoordinateGrid
                    ) or not ureg.is_compatible_with(grid.period, param):
                        raise ProcessingChainError(
                            f"could not find valid conversion for {param}; "
                            f"CoordinateGrid is {grid}"
                        )
                    else:
                        param = float(param / grid.period)
                if np.issubdtype(dtype, np.integer):
                    param = dtype.type(round(param))
                else:
                    param = dtype.type(param)

            if arg_name is None:
                self.args.append(param)
            else:
                self.kwargs[arg_name] = param

        log.debug(f"added processor: {self}")

    def execute(self) -> None:
        self.processor(*self.args, **self.kwargs)

    def __str__(self) -> str:
        return (
            self.processor.__name__
            + "("
            + ", ".join(
                [str(par) for par in self.params]
                + [f"{k}={str(v)}" for k, v in self.kw_params.items()]
            )
            + ")"
        )


class UnitConversionManager(ProcessorManager):
    """A special processor manager for handling converting variables between unit systems."""

    @vectorize(nopython=True, cache=True)
    def convert(buf_in, offset_in, offset_out, period_ratio):  # noqa: N805
        return (buf_in + offset_in) * period_ratio - offset_out

    def __init__(self, var: ProcChainVar, unit: str | Unit) -> None:
        # reference back to our processing chain
        self.proc_chain = var.proc_chain
        # callable function used to process data
        self.processor = UnitConversionManager.convert
        # list of parameters prior to converting to internal representation
        self.params = [var, unit]
        self.kw_params = {}

        from_buffer, from_grid = var._buffer[0]
        period_ratio = from_grid.get_period(unit.period)
        self.out_buffer = np.zeros_like(from_buffer, dtype="float64")
        self.args = [
            from_buffer,
            from_grid.get_offset(),
            unit.get_offset(),
            period_ratio,
            self.out_buffer,
        ]
        self.kwargs = {}

        log.debug(f"added conversion: {self}")


class IOManager(metaclass=ABCMeta):
    """
    Base class. IOManagers will be associated with a type of input/output
    buffer, and must define a read and write for each one. __init__ methods
    should update variable with any information from buffer, and check that
    buffer and variable are compatible.
    """

    @abstractmethod
    def read(self, start: int, end: int) -> None:
        pass

    @abstractmethod
    def write(self, start: int, end: int) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


# Ok, this one's not LGDO
class NumpyIOManager(IOManager):
    """IO Manager for buffers that are numpy arrays"""

    def __init__(self, io_buf: np.ndarray, var: ProcChainVar) -> None:
        assert isinstance(io_buf, np.ndarray) and isinstance(var, ProcChainVar)

        var.update_auto(dtype=io_buf.dtype, shape=io_buf.shape[1:])

        if var.shape != io_buf.shape[1:] or var.dtype != io_buf.dtype:
            raise ProcessingChainError(
                f"numpy.array<{self.io_buf.shape}>({{{self.io_buf.dtype}}}@{self.io_buf.data}) "
                "is not compatible with variable {self.var}"
            )

        self.io_buf = io_buf
        self.var = var
        self.raw_var = var.buffer

    def read(self, start: int, end: int) -> None:
        np.copyto(
            self.raw_var[0 : end - start, ...], self.io_buf[start:end, ...], "unsafe"
        )

    def write(self, start: int, end: int) -> None:
        np.copyto(
            self.io_buf[start:end, ...], self.raw_var[0 : end - start, ...], "unsafe"
        )

    def __str__(self) -> str:
        return (
            f"{self.var} linked to numpy.array(shape={self.io_buf.shape}, "
            f"dtype={self.io_buf.dtype})"
        )


class LGDOArrayIOManager(IOManager):
    """IO Manager for buffers that are lgdo Arrays"""

    def __init__(self, io_array: lgdo.Array, var: ProcChainVar) -> None:
        assert isinstance(io_array, lgdo.Array) and isinstance(var, ProcChainVar)

        unit = io_array.attrs.get("units", None)
        var.update_auto(dtype=io_array.dtype, shape=io_array.nda.shape[1:], unit=unit)

        if isinstance(var.unit, CoordinateGrid):
            if unit is None:
                unit = var.unit.period.u
            elif ureg.is_compatible_with(var.unit.period, unit):
                unit = ureg.Quantity(unit).u
            else:
                raise ProcessingChainError(
                    f"LGDO array and variable {var} have incompatible units "
                    f"({var.unit.period.u} and {unit})"
                )

        if unit is None and var.unit is not None:
            io_array.attrs["units"] = str(var.unit)

        self.io_array = io_array
        self.raw_buf = io_array.nda
        self.var = var
        self.raw_var = var.get_buffer(unit)

        if (
            self.var.shape != self.io_array.nda.shape[1:]
            or self.raw_var.dtype != self.io_array.dtype
        ):
            raise ProcessingChainError(
                f"LGDO object "
                f"{self.io_buf.form_datatype()} is "
                f"incompatible with {str(self.var)}"
            )

    def read(self, start: int, end: int) -> None:
        np.copyto(
            self.raw_var[0 : end - start, ...], self.raw_buf[start:end, ...], "unsafe"
        )

    def write(self, start: int, end: int) -> None:
        np.copyto(
            self.raw_buf[start:end, ...], self.raw_var[0 : end - start, ...], "unsafe"
        )

    def __str__(self) -> str:
        return f"{self.var} linked to lgdo.Array(shape={self.io_array.nda.shape}, dtype={self.io_array.nda.dtype}, attrs={self.io_array.attrs})"


class LGDOArrayOfEqualSizedArraysIOManager(IOManager):
    """IO Manager for buffers that are numpy ArrayOfEqualSizedArrays"""

    def __init__(self, io_array: np.ArrayOfEqualSizedArrays, var: ProcChainVar) -> None:
        assert isinstance(io_array, lgdo.ArrayOfEqualSizedArrays) and isinstance(
            var, ProcChainVar
        )

        unit = io_array.attrs.get("units", None)
        var.update_auto(dtype=io_array.dtype, shape=io_array.nda.shape[1:], unit=unit)

        if isinstance(var.unit, CoordinateGrid):
            if unit is None:
                unit = var.unit.period.u
            elif ureg.is_compatible_with(var.unit.period, unit):
                unit = ureg.Quantity(unit).u
            else:
                raise ProcessingChainError(
                    f"LGDO array and variable {var} have incompatible units "
                    f"({var.unit.period.u} and {unit})"
                )

        if unit is None and var.unit is not None:
            io_array.attrs["units"] = str(var.unit)

        self.io_array = io_array
        self.raw_buf = io_array.nda
        self.var = var
        self.raw_var = var.get_buffer(unit)

        if (
            self.var.shape != self.io_array.nda.shape[1:]
            or self.raw_var.dtype != self.io_array.dtype
        ):
            raise ProcessingChainError(
                f"LGDO object "
                f"{self.io_buf.form_datatype()} is "
                f"incompatible with {str(self.var)}"
            )

    def read(self, start: int, end: int) -> None:
        np.copyto(
            self.raw_var[0 : end - start, ...], self.raw_buf[start:end, ...], "unsafe"
        )

    def write(self, start: int, end: int) -> None:
        np.copyto(
            self.raw_buf[start:end, ...], self.raw_var[0 : end - start, ...], "unsafe"
        )

    def __str__(self) -> str:
        return f"{self.var} linked to lgdo.ArrayOfEqualSizedArrays(shape={self.io_array.nda.shape}, dtype={self.io_array.nda.dtype}, attrs={self.io_array.attrs})"


class LGDOWaveformIOManager(IOManager):
    def __init__(self, wf_table: lgdo.WaveformTable, variable: ProcChainVar) -> None:
        assert isinstance(wf_table, lgdo.WaveformTable) and isinstance(
            variable, ProcChainVar
        )

        self.wf_table = wf_table
        self.wf_buf = wf_table.values.nda
        self.t0_buf = wf_table.t0.nda
        self.dt_buf = wf_table.dt.nda

        dt_units = wf_table.dt_units
        t0_units = wf_table.t0_units
        if dt_units is None:
            dt_units = t0_units
        elif t0_units is None:
            t0_units = dt_units

        # If needed create a new coordinate grid from the IO buffer
        if (
            variable.grid is auto
            and isinstance(dt_units, str)
            and dt_units in ureg
            and isinstance(t0_units, str)
            and t0_units in ureg
        ):
            grid = CoordinateGrid(
                ureg.Quantity(self.dt_buf[0], dt_units),
                ProcChainVar(
                    variable.proc_chain,
                    variable.name + "_dt",
                    shape=(),
                    dtype=self.t0_buf.dtype,
                    grid=None,
                    unit=dt_units,
                    is_coord=True,
                ),
            )
        else:
            grid = None

        self.var = variable
        self.var.update_auto(
            shape=self.wf_buf.shape[1:],
            dtype=self.wf_buf.dtype,
            grid=grid,
            unit=wf_table.values_units,
            is_coord=False,
        )

        if dt_units is None:
            dt_units = self.var.grid.unit_str()
            t0_units = self.var.grid.unit_str()

        self.wf_var = self.var.buffer

        self.t0_var = self.var.grid.get_offset(t0_units)
        self.variable_t0 = isinstance(self.t0_var, np.ndarray)
        if not self.variable_t0:
            self.t0_buf[:] = self.t0_var
        self.wf_table.t0_units = t0_units

        self.dt_buf[:] = self.var.grid.get_period(dt_units)
        self.wf_table.dt_units = dt_units

    def read(self, start: int, end: int) -> None:
        self.wf_var[0 : end - start, ...] = self.wf_buf[start:end, ...]
        self.t0_var[0 : end - start, ...] = self.t0_buf[start:end, ...]

    def write(self, start: int, end: int) -> None:
        self.wf_buf[start:end, ...] = self.wf_var[0 : end - start, ...]
        if self.variable_t0:
            self.t0_buf[start:end, ...] = self.t0_var[0 : end - start, ...]

    def __str__(self) -> str:
        return (
            f"{self.var} linked to pygama.lgdo.WaveformTable("
            f"values(shape={self.wf_table.values.nda.shape}, dtype={self.wf_table.values.nda.dtype}, attrs={self.wf_table.values.attrs}), "
            f"dt(shape={self.wf_table.dt.nda.shape}, dtype={self.wf_table.dt.nda.dtype}, attrs={self.wf_table.dt.attrs}), "
            f"t0(shape={self.wf_table.t0.nda.shape}, dtype={self.wf_table.t0.nda.dtype}, attrs={self.wf_table.t0.attrs}))"
        )


def build_processing_chain(
    lh5_in: lgdo.Table,
    dsp_config: dict | str,
    db_dict: dict = None,
    outputs: list[str] = None,
    block_width: int = 16,
) -> tuple[ProcessingChain, list[str], lgdo.Table]:
    """Produces a :class:`ProcessingChain` object and an LH5
    :class:`~.lgdo.table.Table` for output parameters from an input LH5
    :class:`~.lgdo.table.Table` and a JSON recipe.

    Parameters
    ----------
    lh5_in
        HDF5 table from which raw data is read. At least one row of entries
        should be read in prior to calling this!

    dsp_config
        A dictionary or JSON filename containing the recipes for computing DSP
        parameter from raw parameters. The format is as follows:

        .. code-block:: json

            {
               "outputs" : [ "par1", "par2" ]
               "processors" : {
                  "name1, name2" : {
                    "function" : "func1"
                    "module" : "mod1"
                    "args" : ["arg1", 3, "arg2"]
                    "kwargs" : {"key1": "val1"}
                    "init_args" : ["arg1", 3, "arg2"]
                    "unit" : ["u1", "u2"]
                    "defaults" : {"arg1": "defval1"}
                  }
               }
            }

        - ``outputs`` -- list of output parameters (strings) to compute by
          default. See `outputs` argument
        - ``processors`` -- configuration dictionary

          - ``name1, name2`` -- dictionary. key contains comma-separated
            names of parameters computed

            - ``function`` -- string, name of function to call.  Function
              should implement the :class:`numpy.gufunc` interface, a factory
              function returning a ``gufunc``, or an arbitrary function that
              can be mapped onto a ``gufunc``
            - ``module`` -- string, name of module containing function
            - ``args``-- list of strings or numerical values. Contains
              list of names of computed and input parameters or
              constant values used as inputs to function. Note that
              outputs should be fed by reference as args! Arguments read
              from the database are prepended with ``db``.
            - ``kwargs`` -- dictionary. Keyword arguments for
              :meth:`ProcesssingChain.add_processor`.
            - ``init_args`` --  list of strings or numerical values. List
              of names of computed and input parameters or constant values
              used to initialize a :class:`numpy.gufunc` via a factory
              function
            - ``unit`` -- list of strings. Units for parameters
            - ``defaults`` -- dictionary. Default value to be used for
              arguments read from the database

    db_dict
        A nested :class:`dict` pointing to values for database arguments. As
        instance, if a processor uses the argument ``db.trap.risetime``, it
        will look up ``db_dict['trap']['risetime']`` and use the found value.
        If no value is found, use the default defined in `dsp_config`.

    outputs
        List of parameters to put in the output LH5 table. If ``None``,
        use the parameters in the ``"outputs"`` list from `dsp_config`.

    block_width
        number of entries to process at once. To optimize performance,
        a multiple of 16 is preferred, but if performance is not an issue
        any value can be used.

    Returns
    -------
    (proc_chain, field_mask, lh5_out)
        - `proc_chain` -- :class:`ProcessingChain` object that is executed
        - `field_mask` -- list of input fields that are used
        - `lh5_out` -- output :class:`~.lgdo.table.Table` containing processed
          values
    """
    proc_chain = ProcessingChain(block_width, lh5_in.size)

    if isinstance(dsp_config, str):
        with open(expand_path(dsp_config)) as f:
            dsp_config = json.load(f)
    elif dsp_config is None:
        dsp_config = {"outputs": [], "processors": {}}
    elif isinstance(dsp_config, dict):
        # We don't want to modify the input!
        dsp_config = deepcopy(dsp_config)
    else:
        raise ValueError("dsp_config must be a dict, json file, or None")

    if outputs is None:
        outputs = dsp_config["outputs"]

    processors = dsp_config["processors"]

    # prepare the processor list
    multi_out_procs = {}
    db_parser = re.compile(r"db.[\w_.]+")
    for key, node in processors.items():
        # if we have multiple outputs, add each to the processesors list
        keys = [k for k in re.split(",| ", key) if k != ""]
        if len(keys) > 1:
            for k in keys:
                multi_out_procs[k] = key

        # find DB lookups in args and replace the values
        args = node["args"]
        for i, arg in enumerate(args):
            if not isinstance(arg, str):
                continue
            for db_var in db_parser.findall(arg):
                try:
                    db_node = db_dict
                    for key in db_var[3:].split("."):
                        db_node = db_node[key]
                    log.debug(f"database lookup: found {db_node} for {db_var}")
                except (KeyError, TypeError):
                    try:
                        db_node = node["defaults"][db_var]
                        log.debug(
                            f"Database lookup: using default value of {db_node} for {db_var}"
                        )
                    except (KeyError, TypeError):
                        raise ProcessingChainError(
                            f"""did not find {db_var} in database, and could
                                not find default value."""
                        )
                if arg == db_var:
                    arg = db_node
                else:
                    arg = arg.replace(db_var, str(db_node))
            args[i] = arg

        # parse the arguments list for prereqs, if not included explicitly
        if "prereqs" not in node:
            prereqs = []
            for arg in node["args"]:
                if not isinstance(arg, str):
                    continue
                for prereq in proc_chain.get_variable(arg, True):
                    if prereq not in prereqs and prereq not in keys:
                        prereqs.append(prereq)
            node["prereqs"] = prereqs

        log.debug(f"prereqs for {key} are {node['prereqs']}")

    processors.update(multi_out_procs)

    def resolve_dependencies(
        par: str, resolved: list[str], leafs: list[str], unresolved: list[str] = None
    ) -> None:
        """
        Recursive function to crawl through the parameters/processors and get a
        sequence of unique parameters such that parameters always appear after
        their dependencies. For parameters that are not produced by the
        :class:`ProcessingChain` (i.e. input/db parameters), add them to the
        list of leafs.

        .. [ref] https://www.electricmonk.nl/docs/dependency_resolving_algorithm/dependency_resolving_algorithm.html
        """
        if unresolved is None:
            unresolved = []

        if par in resolved:
            return
        elif par in unresolved:
            raise ProcessingChainError(
                f"Circular references detected for parameter '{par}'"
            )

        # if we don't find a node, this is a leaf
        node = processors.get(par)
        if node is None:
            if par not in leafs:
                leafs.append(par)
            return

        # if it's a string, that means it is part of a processor that returns multiple outputs (see above); in that case, node is a str pointing to the actual node we want
        if isinstance(node, str):
            resolve_dependencies(node, resolved, leafs, unresolved)
            return

        edges = node["prereqs"]
        unresolved.append(par)
        for edge in edges:
            resolve_dependencies(edge, resolved, leafs, unresolved)
        resolved.append(par)
        unresolved.remove(par)

    proc_par_list = []  # calculated from processors
    input_par_list = []  # input from file and used for processors
    copy_par_list = []  # copied from input to output
    out_par_list = []
    for out_par in outputs:
        if out_par not in processors:
            copy_par_list.append(out_par)
        else:
            resolve_dependencies(out_par, proc_par_list, input_par_list)
            out_par_list.append(out_par)

    log.debug(f"processing parameters: {proc_par_list}")
    log.debug(f"required input parameters: {input_par_list}")
    log.debug(f"copied output parameters: {copy_par_list}")
    log.debug(f"processed output parameters: {out_par_list}")

    # Now add all of the input buffers from lh5_in (and also the clk time)
    for input_par in input_par_list:
        buf_in = lh5_in.get(input_par)
        if buf_in is None:
            log.warning(
                f"I don't know what to do with {input_par}. Building output without it!"
            )
        try:
            proc_chain.link_input_buffer(input_par, buf_in)
        except Exception as e:
            raise ProcessingChainError(
                f"Exception raised while linking input buffer {input_par}."
            ) from e

    # now add the processors
    for proc_par in proc_par_list:
        recipe = processors[proc_par]
        try:
            module = importlib.import_module(recipe["module"])
            func = getattr(module, recipe["function"])
            args = recipe["args"]

            # Initialize the new variables, if needed
            if "unit" in recipe:
                new_vars = [k for k in re.split(",| ", proc_par) if k != ""]
                for i, name in enumerate(new_vars):
                    unit = recipe.get("unit", auto)
                    if isinstance(unit, list):
                        unit = unit[i]

                    proc_chain.add_variable(name, unit=unit)

            # get this list of kwargs
            kwargs = recipe.get("kwargs", {})  # might also need db lookup here

            # if init_args are defined, parse any strings and then call func
            # as a factory/constructor function
            try:
                init_args_in = recipe["init_args"]
                init_args = []
                init_kwargs = {}
                for _, arg in enumerate(init_args_in):
                    for db_var in db_parser.findall(arg):
                        try:
                            db_node = db_dict
                            for key in db_var[3:].split("."):
                                db_node = db_node[key]
                            log.debug(f"database lookup: found {db_node} for {db_var}")
                        except (KeyError, TypeError):
                            try:
                                db_node = recipe["defaults"][db_var]
                                log.debug(
                                    "database lookup: using default value of {db_node} for {db_var}"
                                )
                            except (KeyError, TypeError):
                                raise ProcessingChainError(
                                    f"did not find {db_var} in database, and "
                                    f"could not find default value."
                                )

                        if arg == db_var:
                            arg = db_node
                        else:
                            arg = arg.replace(db_var, str(db_node))

                    # see if string can be parsed by proc_chain
                    if isinstance(arg, str):
                        arg = proc_chain.get_variable(arg)
                    if isinstance(arg, dict):
                        init_kwargs.update(arg)
                    else:
                        init_args.append(arg)

                expr = ", ".join(
                    [f"{a}" for a in init_args]
                    + [f"{k}={v}" for k, v in init_kwargs.items()]
                )
                log.debug(f"building function from init_args: {func.__name__}({expr})")
                func = func(*init_args)
            except KeyError:
                pass

            proc_chain.add_processor(func, *args, **kwargs)
        except Exception as e:
            raise ProcessingChainError(
                "Exception raised while attempting to add processor:\n"
                + json.dumps(recipe, indent=2)
            ) from e

    # build the output buffers
    lh5_out = lgdo.Table(size=proc_chain._buffer_len)

    # add inputs that are directly copied
    for copy_par in copy_par_list:
        buf_in = lh5_in.get(copy_par)
        if buf_in is None:
            log.warning(
                f"I don't know what to do with {copy_par}. Building output without it!"
            )
        else:
            lh5_out.add_field(copy_par, buf_in)

    # finally, add the output buffers to lh5_out and the proc chain
    for out_par in out_par_list:
        try:
            buf_out = proc_chain.link_output_buffer(out_par)
            lh5_out.add_field(out_par, buf_out)
        except Exception as e:
            raise ProcessingChainError(
                f"Exception raised while linking output buffer {out_par}."
            ) from e

    field_mask = input_par_list + copy_par_list
    return (proc_chain, field_mask, lh5_out)
