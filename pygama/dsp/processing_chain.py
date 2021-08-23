from __future__ import annotations

import numpy as np
import json
import re
import ast
import itertools as it
import importlib
from abc import ABCMeta, abstractmethod

from dataclasses import dataclass, field
from typing import Union
from copy import deepcopy
from scimath.units import convert
from scimath.units.api import unit_parser
from scimath.units.unit import unit

from numba import vectorize

from pygama import lh5

from pygama.lgdo.array import Array
from pygama.lgdo.arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from pygama.lgdo.waveform_table import WaveformTable
from pygama.lgdo.vectorofvectors import VectorOfVectors

from pygama.math.units import unit_registry as ureg
from pygama.math.units import Unit, Quantity

# Filler value for variables to be automatically deduced later
auto = 'auto'

# Map from ast interpretter operations to functions to call and format string
ast_ops_dict = {ast.Add: (np.add, '{}+{}'),
                ast.Sub: (np.subtract, '{}-{}'),
                ast.Mult: (np.multiply, '{}*{}'),
                ast.Div: (np.divide, '{}/{}'),
                ast.FloorDiv: (np.floor_divide, '{}//{}'),
                ast.USub: (np.negative, '-{}')}

@dataclass
class CoordinateGrid:
    """Helper class that describes a system of units, consisting of a period
    and offset. Period is a unitted Quantity, offset is a scalar in units of
    period, a Unit or a ProcChainVar. In the last case, a ProcessingChain
    variable is used to store a different offset for each event"""
    period: Union[Quantity, Unit]
    offset: Union[float, int, Quantity, ProcChainVar]

    def __post_init__(self):
        if isinstance(self.period, Unit):
            self.period *= 1 # make Quantity
        if isinstance(self.offset, Quantity):
            self.offset = float(self.offset/self.period)

    def __eq__(self, other):
        # true if values are equal; if offset is a variable, compare reference
        return self.period==other.period and \
            (self.offset is other.offset if isinstance(self.offset, ProcChainVar) else self.offset==other.offset)

    def unit_str(self):
        str = format(self.period.u, '~')
        if str=='': str = str(self.period.u)
        return str

    def get_period(self, unit):
        if isinstance(unit, str): unit = ureg.Quantity(unit)
        return float(self.period/unit)

    def get_offset(self, unit=None):
        # Get the offset (convert)ed to unit. If unit is None use period
        if unit is None:
            if isinstance(self.offset, (int, float)):
                return self.offset
            unit = self.period.u
        if isinstance(unit, str): unit = ureg.Quantity(unit)

        if isinstance(self.offset, ProcChainVar):
            return self.offset.get_buffer(CoordinateGrid(unit, 0))
        elif isinstance(self.offset, Quantity):
            return float(self.offset/unit)
        else:
            return self.offset * float(self.period/unit)

    def __str__(self):
        offset = self.offset.name if isinstance(self.offset, ProcChainVar) \
            else str(self.offset)
        return "({},{})".format(str(self.period), offset)

class ProcChainVar:
    """Helper data class with buffer and information for internal variables in
    ProcessingChain. Members can be set to auto to attempt to deduce these
    when adding this variable to a processor for the first time"""
    def __init__(self,
                 proc_chain: ProcessingChain,
                 name: str,
                 shape: tuple = auto,
                 dtype: np.dtype = auto,
                 grid: CoordinateGrid = auto,
                 unit: str = auto,
                 is_coord: bool = auto ):
        assert isinstance(proc_chain, ProcessingChain) and isinstance(name, str)
        self.proc_chain = proc_chain
        self.name = name
        
        self.shape = shape
        self.dtype = dtype
        self.grid = grid
        self.unit = unit
        self.is_coord = is_coord

        # ndarray containing data buffer of size block_width x shape
        # list of ndarrays in different coordinate systems if is_coord is true
        self._buffer: Union[list, np.ndarray] = None
        
        self.proc_chain._print(2, 'Added variable:', self.description())

    # Use this to enforce type constraints and perform conversions
    def __setattr__(self, name, value):
        if value is auto:
            pass
        
        elif name=='shape':
            if isinstance(value, int):
                value = (value,)
            value = tuple(value)
            assert all(isinstance(d, int) for d in value)

        elif name=='dtype' and not isinstance(value, np.dtype):
            value = np.dtype(value)

        elif name=='grid' and not isinstance(value, CoordinateGrid) and not value is None:
            period, offset = value if isinstance(value, tuple) else value, 0
            
            if isinstance(period, str) and period in ureg:
                period = Quantity(period)
            elif not isinstance(period, (Quantity, Unit)):
                raise ProcessingChainError('Cannot parse '+str(period)+' into a valid period.')
            
            if isinstance(offset, str):
                offset = self.proc_chain.get_variable(offset)
            value = CoordinateGrid(period, offset)

        elif name=='unit' and not value is None:
            value = str(value)
            
        elif name=='is_coord':
            value = bool(value)
            if value:
                if self._buffer is None:
                    self._buffer = []
                elif isinstance(self._buffer, np.ndarray):
                    self._buffer = [(self._buffer, CoordinateGrid(self.unit))]

        super(ProcChainVar, self).__setattr__(name, value)
        
    def get_buffer(self, unit=None):
        # If buffer needs to be created, do so now
        if self._buffer is None:
            if self.shape is auto:
                raise ProcessingChainError("Cannot deduce shape of "+self.name)
            if self.dtype is auto:
                raise ProcessingChainError("Cannot deduce shape of "+self.name)

            # create the buffer so that the array start is aligned in memory on a multiple of 64 bytes
            self._buffer = np.zeros(shape=(self.proc_chain._block_width,)+self.shape, dtype=self.dtype)

        # if variable isn't a coordinate, we're all set
        if self.is_coord is False or self.is_coord is auto:
            return self._buffer

        # if no unit is given, use the native unit
        if unit is None:
            if isinstance(self.unit, str):
                unit = CoordinateGrid(self.unit)
        elif not isinstance(unit, CoordinateGrid):
            unit = CoordinateGrid(unit)

        # if this is our first time accessing, no conversion is needed
        if len(self._buffer)==0:
            if self.shape is auto:
                raise ProcessingChainError("Cannot deduce shape of "+self.name)
            if self.dtype is auto:
                raise ProcessingChainError("Cannot deduce shape of "+self.name)
            
            buff = np.zeros(shape=(self.proc_chain._block_width,)+self.shape, dtype=self.dtype)
            self._buffer.append((buff, unit))
            return buff
        
        # check if coordinate conversion has been done already
        for buff, grid in self._buffer:
            if grid == unit:
                return buff

        # If we get this far, add conversion processor to ProcChain and add new buffer to _buffer
        conversion_manager = UnitConversionManager(self, unit)
        self._buffer.append(conversion_manager.out_buffer, unit)
        self.proc_chain._proc_managers.append(conversion_manager)
        return out

    @property
    def buffer(self):
        return self.get_buffer()

    def description(self):
        return "{}({}, {}, coords: {}, unit: {}{})".format(self.name, \
            str(self.shape), str(self.dtype), str(self.grid), \
            str(self.unit), " (coord)" if self.is_coord is True else '' )

    # Update any variables set to auto; leave the others alone. Emit a messsage
    # only if anything was updated
    def update_auto(self, shape=auto, dtype=auto, grid=auto, unit=auto, is_coord=auto):
        updated = False
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
            self.proc_chain._print(2, 'Updated variable:', self.description())
        
    def __str__(self):
        return self.name

###############################################################################
############################### The Main Class ################################
###############################################################################

class ProcessingChain:
    """
    A ProcessingChain is used to efficiently perform a sequence of digital
    signal processing (dsp) transforms. It contains a list of dsp functions and
    a set of constant values and named variables contained in fixed memory
    locations. When executing the ProcessingChain, processors will act on the
    internal memory without allocating new memory in the process. Furthermore,
    the memory is allocated in blocks, enabling vectorized processing of many
    entries at once. To set up a ProcessingChain, use the following methods:
    1) add_input_buffer: bind a named variable to an external numpy array to
         read data from
    2) add_processor: add a dsp function and bind its inputs to a set of named
         variables and constant values
    3) add_output_buffer: bind a named variable to an external numpy array to
         write data into
    When calling these methods, the ProcessingChain class will use available
    information to allocate buffers to the correct sizes and data types. For
    this reason, transforms will ideally implement the numpy ufunc class,
    enabling broadcasting of array dimensions. If not enough information is
    available to correctly allocate memory, it can be provided through the
    named variable strings or by calling add_vector or add_scalar.
    """
    def __init__(self, block_width=8, buffer_len=None, verbosity=1):
        """Named arguments:
        - block_width: number of entries to simultaneously process.
        - buffer_len: length of input and output buffers. Should be a multiple
            of block_width
        - verbosity: integer indicating the verbosity level:
            0: Print nothing (except errors...)
            1: Print basic warnings (default)
            2: Print basic debug info
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
        self._verbosity = verbosity


    def add_variable(self, name, dtype=auto, shape=auto, grid=auto, unit=auto, is_coord=auto):
        """Add a named variable containing a block of values or arrays
        Parameters
        ----------
        name : name of variable
        dtype : numpy dtype or type string. Default is None, meaning dtype
            will be deduced later, if possible
        shape : length or shape tuple of element. Default is None, meaning
            length will be deduced later, if possible
        period : unit with period of waveform associated with object
        offset : unit with offset of waveform associated with object
        is_coord : if True, transform value based on period and offset
        """
        self._validate_name(name, raise_exception=True)
        if name in self._vars_dict:
            raise ProcessingChainError(name+' is already in variable list')
        
        var = ProcChainVar(self, name, shape=shape, dtype=dtype, grid=grid,
                           unit=unit, is_coord=is_coord)
        self._vars_dict[name] = var
        return var
    
    
    def link_input_buffer(self, varname, buff=None):
        """Link an input buffer to a variable
        Parameters
        ----------
        varname : name of internal variable to copy into buffer at the end
            of processor execution. If variable does not yet exist, it will
            be created with a similar shape to the provided buffer
        buff : numpy array or lgdo class to use as input buffer. If None,
            create a new buffer with a similar shape to the variable
        
        Return : buff or newly allocated input buffer
        """
        self._validate_name(varname, raise_exception=True)
        var = self.get_variable(varname)
        if var is None:
            var = self.add_variable(varname)
        
        if not isinstance(var, ProcChainVar):
            raise ProcessingChainError("Must link an input buffer to a processing chain variable")

        # Create input buffer that will be linked and returned if none exists
        if buff is None:
            if var is None:
                raise ProcessingChainError(varname+" does not exist and no buffer was provided")
            elif isinstance(var.grid, CoordinateGrid) and len(var.shape)==1:
                buff = WaveformTable(size = self._buffer_len,
                                     wf_len = var.shape[0],
                                     dtype = var.dtype)
            elif len(var.shape)==1:
                buff = Array(shape=(self._buffer_len), dtype = var.dtype)
            else:
                buff = np.ndarray((self._buffer_len,) + var.shape[1:], var.dtype)

        # Add the buffer to the input buffers list
        if isinstance(buff, np.ndarray):
            out_man = NumpyIOManager(buff, var)
        elif isinstance(buff, Array):
            out_man = LGDOsArrayIOManager(buff, var)
        elif isinstance(buff, WaveformTable):
            out_man = LGDOWaveformIOManager(buff, var)
        else:
            raise ProcessingChainError("Could not link input buffer of unknown type", str(buff))

        self._print(2, "Added input buffer:", str(out_man))
        self._input_managers.append(out_man)

        return buff
    
    
    def link_output_buffer(self, varname, buff=None):
        """Link an output buffer to a variable
        Parameters
        ----------
        varname : name of internal variable to copy into buffer at the end
            of processor execution. If variable does not yet exist, it will
            be created with a similar shape to the provided buffer
        buff : numpy array or lgdo class to use as output buffer. If None,
            create a new buffer with a similar shape to the variable
        
        Return : buff or newly allocated output buffer
        """
        self._validate_name(varname, raise_exception=True)
        var = self.get_variable(varname)
        if var is None:
            var = self.add_variable(varname)
        
        if not isinstance(var, ProcChainVar):
            raise ProcessingChainError("Must link an output buffer to a processing chain variable")

        # Create output buffer that will be linked and returned if none exists
        if buff is None:
            if var is None:
                raise ProcessingChainError(varname+" does not exist and no buffer was provided")
            elif isinstance(var.grid, CoordinateGrid) and len(var.shape)==1:
                buff = WaveformTable(size = self._buffer_len,
                                     wf_len = var.shape[0],
                                     dtype = var.dtype)
            elif len(var.shape)==1:
                buff = Array(shape=(self._buffer_len), dtype = var.dtype)
            else:
                buff = np.ndarray((self._buffer_len,) + var.shape[1:], var.dtype)

        # Add the buffer to the output buffers list
        if isinstance(buff, np.ndarray):
            out_man = NumpyIOManager(buff, var)
        elif isinstance(buff, Array):
            out_man = LGDOsArrayIOManager(buff, var)
        elif isinstance(buff, WaveformTable):
            out_man = LGDOWaveformIOManager(buff, var)
        else:
            raise ProcessingChainError("Could not link output buffer of unknown type", str(buff))

        self._print(2, "Added output buffer:", str(out_man))
        self._output_managers.append(out_man)

        return buff

    def add_processor(self, func, *args, signature=None, types=None):
        # make a list of parameters from *args. Replace any strings in the list
        # with numpy objects from vars_dict, where able
        params = []
        for i, param in enumerate(args):
            if(isinstance(param, str)):
                param_val = self.get_variable(param)
                if param_val is not None:
                    param=param_val
            params.append(param)

        proc_man = ProcessorManager(self, func, params, signature, types)
        self._proc_managers.append(proc_man)

    def execute(self, start=0, end=None):
        """Execute the dsp chain on the entire input/output buffers"""
        if end is None: end = self._buffer_len
        for begin in range(start, end, self._block_width):
            end = min(offset+self._block_width, self._buffer_len)
            self._execute_procs(offset, end)
        


    def get_variable(self, expr, get_names_only=False):
        """Parse string expr into a numpy array or value, using the following
        syntax:
          - numeric values are parsed into ints or floats
          - units found in the pint package
          - other strings are parsed into variable names. If get_name_only is
            False, fetch the internal buffer (creating it as needed). Else,
            return a string of the name
          - if a string is followed by (...), try parsing into one of the
            following expressions:
              len(expr): return the length of the array found with expr
              round(expr): return the value found with expr to the nearest int
              varname(shape, type): allocate a new buffer with the specified
                shape and type, using varname. This is used if the automatic
                type and shape deduction for allocating variables fails
          - Unary and binary operators +, -, *, /, // are available. If
              a variable name is included in the expression, a processor
              will be added to the ProcChain and a new buffer allocated
              to store the output
          - varname[slice]: return the variable with a slice applied. Slice
              values can be floats, and will have round applied to them
        If get_names_only is set to True, do not fetch or allocate new arrays,
          instead return a list of variable names found in the expression
        """
        names = []
        try:
            var = self._parse_expr(ast.parse(expr, mode='eval').body, \
                                    expr, get_names_only, names)
        except Exception as e:
            raise ProcessingChainError("Could not parse expression:\n  " + expr) from e
        
        if not get_names_only:
            return var
        else: return names


    def _parse_expr(self, node, expr, dry_run, var_name_list):
        """
        helper function for get_variable that recursively evaluates the AST tree
        based on: https://stackoverflow.com/a/9558001. Whenever we encounter
        a variable name, add it to var_name_list (which should begin as an
        empty list). Only add new variables and processors to the chain if
        dry_run is True
        """
        if node is None:
            return None

        elif isinstance(node, ast.Num):
            return node.n

        elif isinstance(node, ast.Str):
            return node.s
        
        elif isinstance(node, ast.Constant):
            return node.val

        # look for name in variable dictionary
        elif isinstance(node, ast.Name):
            # check if it is a unit
            if node.id in ureg:
                return ureg.Unit(node.id)

            #check if it is a variable
            var_name_list.append(node.id)
            val = self._vars_dict.get(node.id, None)
            return val

        # define binary operators (+,-,*,/)
        elif isinstance(node, ast.BinOp):
            lhs = self._parse_expr(node.left, expr, dry_run, var_name_list)
            rhs = self._parse_expr(node.right, expr, dry_run, var_name_list)
            if rhs is None or lhs is None: return None
            op, op_form = ast_ops_dict[type(node.op)]

            if not (isinstance(lhs, ProcChainVar) or isinstance(rhs, ProcChainVar)):
                return op(lhs, rhs)

            name = '('+op_form.format(str(lhs), str(rhs))+')'
            if isinstance(lhs, ProcChainVar) and isinstance(rhs, ProcChainVar):
                #TODO: handle units/coords; for now make them match lhs
                out = ProcChainVar(self, name, is_coord=lhs.is_coord)
            elif isinstance(lhs, ProcChainVar):
                out = ProcChainVar(self, name, lhs.shape, lhs.dtype, lhs.grid, lhs.unit, is_coord=lhs.is_coord)
            else:
                out = ProcChainVar(self, name, rhs.shape, rhs.dtype, rhs.grid, rhs.unit, is_coord=rhs.is_coord)
                
            self._proc_managers.append(ProcessorManager(self, op, [lhs, rhs, out]))
            return out

        # define unary operators (-)
        elif isinstance(node, ast.UnaryOp):
            operand = self._parse_expr(node.operand, expr, dry_run, var_name_list)
            if operand is None: return None
            op, op_form = ast_ops_dict[type(node.op)]
            name = '('+op_form.format(str(operand))+')'
            
            if isinstance(operand, ProcChainVar):
                out = ProcChainVar(self, name, operand.shape, operand.dtype,
                              operand.grid, operand.unit, operand.is_coord)
                self._proc_managers.append(ProcessorManager(self, op, [operand, out]))
            else:
                out = op(out)

            return out

        elif isinstance(node, ast.Subscript):
            # print(ast.dump(node))
            val = self._parse_expr(node.value, expr, dry_run, var_name_list)
            if val is None: return None
            if not isinstance(val, ProcChainVar):
                raise ProcessingChainError("Cannot apply subscript to", node.value)

            def get_index(slice_value):
                ret = self._parse_expr(slice_value, expr, dry_run, var_name_list)
                if isinstance(ret, Quantity):
                    ret = float(ret/val.grid.period)
                if isinstance(ret, float):
                    round_ret = int(round(ret))
                    if abs(ret - round_ret) > 0.0001:
                        self._print(0, 'Slice value', str(slice_value), \
                                    'is non-integer. Rounding to', round_ret)
                    return round_ret
                return int(ret)

            out = val
            if isinstance(node.slice, ast.Index):
                out.buffer = val[..., get_index(node.slice.value)]
            elif isinstance(node.slice, ast.Slice):
                sl = slice(get_index(node.slice.lower),
                           get_index(node.slice.upper),
                           get_index(node.slice.step) )
                out.buffer = val[..., sl]
                if sl.start is not None:
                    out.grid[1] += out.grid[0] * sl.start
                if sl.step is not None:
                    out.grid[0] *= sl.step

            elif isinstance(node.slice, ast.ExtSlice):
                slices = tuple(node.slice.dims)
                for i, sl in enumerate(slices):
                    if isinstance(sl, ast.index):
                        slices[i] = self._parse_expr(sl.value, expr, dry_run, var_name_list)
                    else:
                        slices[i] = slice(self._parse_expr(sl.upper, expr, dry_run, var_name_list),
                                          self._parse_expr(sl.lower, expr, dry_run, var_name_list),
                                          self._parse_expr(sl.step, expr, dry_run, var_name_list) )
                out.buffer = val[..., slices]
            return out

        # for name.attribute
        elif isinstance(node, ast.Attribute):
            val = self._parse_expr(node.value, expr, dry_run, var_name_list)
            if val is None: return None
            # get shape with buffer_len dimension removed
            if node.attr=='shape' and isinstance(val, np.ndarray):
                return val.shape[1:]

        # for func(args, kwargs)
        elif isinstance(node, ast.Call):
            func = self.func_list.get(node.func.id, None)
            args = [ self._parse_expr(arg, expr, dry_run, var_name_list) \
                     for arg in node.args ]
            kwargs = { kwarg.arg:self._parse_expr(kwarg.value, expr, dry_run, var_name_list) for kwarg in node.keywords }
            if func is not None:
                return func(self, *args, **kwargs)
            elif self._validate_name(node.func.id):
                var_name = node.func.id
                var_name_list.append(var_name)
                if var_name in self._vars_dict:
                    return var
                elif not dry_run:
                    return self.add_variable(var_name, *args, **kwargs)
                else:
                    return None

            else:
                raise ProcessingChainError("Do not recognize call to "+func+" with arguments " + str([str(arg.__dict__) for arg in node.args]))

        raise ProcessingChainError("Cannot parse AST nodes of type " + str(node.__dict__))

    # Get length of ProcChainVar
    def _length(self, var):
        if not isinstance(var, ProcChainVar):
            raise ProcessingChainError("Cannot call len() on " + str(var))
        if not len(var.buffer.shape)==2:
            raise ProcessingChainError(str(var)+" has wrong number of dims")
        return var.buffer.shape[1]

    def _validate_name(self, name, raise_exception=False):
        """Check that name is alphanumeric, and not an already used keyword"""
        isgood = re.match("\A\w+$", name) and name not in self.func_list \
            and not name in ureg
        if raise_exception and not isgood:
            raise ProcessingChainError(name+" is not a valid variable name")
        return isgood


    def _execute_procs(self, start, end):
        """
        copy from input buffers to variables
        call all the processors on their paired arg tuples
        copy from variables to list of output buffers
        """
        # Copy input buffers into proc chain buffers
        for in_man in self._input_managers:
            in_man.read(start, end)

        # Loop through processors and run each one
        for proc_man in self._proc_managers:
            try:
                proc_man.execute()
            except DSPFatal as e:
                e.processor = str(proc_man)
                e.wf_range = (start, end)
                raise e

        # copy from processing chain buffers into output buffers
        for out_man in self._output_managers:
            io_man.write(start, end)



    def _print(self, verbosity, *args, **kwargs):
        """Helper for output that checks verbosity before printing and
        converts things into strings. At verbosity 0, print to stderr"""
        if verbosity==0 and not 'file' in kwargs:
            kwargs['file']=sys.stderr
        if self._verbosity >= verbosity:
            print(*[str(arg) for arg in args], **kwargs)


    def __str__(self):
        return 'Input variables:\n  ' \
        + '\n  '.join([str(in_man) for in_man in self._input_managers]) \
        + '\nProcessors:\n  ' \
        + '\n  '.join([str(proc_man) for proc_man in self._proc_managers]) \
        + '\nOutput variables:\n  ' \
        + '\n  '.join([str(out_man) for out_man in self._output_managers])

    # Map from function names when using ast interpretter to ProcessorChain functions that can be called
    func_list = {'len':_length,
                 'round':round, }
    #'period':get_period,
    #'offset':get_offset }


########################################################################### 
# The class that calls processors and makes sure variables are compatible #
###########################################################################
class ProcessorManager:

    @dataclass
    class DimInfo:
        length : int # length of arrays in this dimension
        grid : CoordinateGrid # period and offset of arrays in this dimension
    
    def __init__(self, proc_chain, func, params, signature=None, types=None):
        assert isinstance(proc_chain, ProcessingChain) and callable(func) \
            and isinstance(params, list)

        # reference back to our processing chain
        self.proc_chain = proc_chain
        # callable function used to process data
        self.processor = func
        # list of parameters prior to converting to internal representation
        self.params = params
        # list of raw values and buffers from params; we will fill this soon
        self.raw_params = []
        
        # Get the signature and list of valid types for the function
        if signature is None:
            self.signature = func.signature
        if self.signature is None:
            self.signature = ','.join(['()']*func.nin) + '->' \
                           + ','.join(['()']*func.nout)
        
        # Get list of allowed type signatures
        if types is None:
            types = func.types.copy()
        if types is None:
            raise ProcessingChainError("Could not find a type signature list for " + func.__name__ + ". Please supply a valid list of types.")
        if not isinstance(types, list):
            types = [types]
        types = [ typestr.replace('->', '') for typestr in types]
        
        # Make sure arrays obey the broadcasting rules, and make a dictionary
        # of the correct dimensions and unit system
        dims_list = re.findall("\((.*?)\)", self.signature)

        if not len(dims_list)==len(params):
            raise ProcessingChainError("Expected {} arguments from signature {}; found {}: ({})".format(len(dims_list), self.signature, len(params), ', '.join([str(par) for par in params])))
        
        dims_dict = {} # map from dim name -> DimInfo
        outerdims = [] # list of DimInfo
        grid = None # period/offset to use for unit and coordinate conversions

        for ipar, dims in enumerate(dims_list):
            param = self.params[ipar]
            if not isinstance(param, ProcChainVar):
                continue

            # find type signatures that match type of array
            if param.dtype is not auto:
                arr_type = param.dtype.char
                types = [type_sig for type_sig in types if arr_type==type_sig[ipar]]

            # fill out dimensions from dim signature and check if it works
            if param.shape is auto:
                continue
            fun_dims = [od for od in outerdims] + \
                [d.strip() for d in dims.split(',') if d]
            arr_dims = list(param.shape)
            arr_grid = param.grid if param.grid is not auto else None
            if not grid:
                grid = arr_grid

            #check if arr_dims can be broadcast to match fun_dims
            for i in range(max(len(fun_dims), len(arr_dims))):
                fd = fun_dims[-i-1] if i<len(fun_dims) else None
                ad = arr_dims[-i-1] if i<len(arr_dims) else None
                
                if(isinstance(fd, str)):
                    if fd in dims_dict:
                        this_dim = dims_dict[fd]
                        if not ad or this_dim.length!=ad:
                            raise ProcessingChainError("Failed to broadcast array dimensions for "+func.__name__+". Could not find consistent value for dimension "+fd)
                        if not this_dim.grid:
                            dims_dict[fd].grid = arr_grid
                        elif arr_grid and arr_grid!=this_dim.grid:
                            self._print(0, "Arrays of dimension", fd, "for", func.__name__, "do not have consistent period and offset!")
                    else:
                        dims_dict[fd] = self.DimInfo(ad, arr_grid)
                
                elif not fd:
                    # if we ran out of function dimensions, add a new outer dim
                    outerdims.insert(0, self.DimInfo(ad, arr_grid))
                    
                elif not ad:
                    continue
                
                elif fd.length != ad:
                    # If dimensions disagree, either insert a broadcasted array dimension or raise an exception
                    if len(fun_dims)>len(arr_dims):
                        arr_dims.insert(len(arr_dims)-i, 1)
                    elif len(fun_dims)<len(arr_dims):
                        outerdims.insert(len(fun_dims)-i, self.DimInfo(ad, arr_grid))
                        fun_dims.insert(len(fun_dims)-i, ad)
                    else:
                        raise ProcessingChainError("Failed to broadcast array dimensions for "+func.__name__+". Input arrays do not have consistent outer dimensions.")
                elif not fd.grid:
                    outerdims[len(fun_dims)-i].grid = arr_grid
                
                elif arr_grid and fd.grid!=arr_grid:
                    self._print(0, "Arrays of dimension", fd, "for", func.__name__, "do not have consistent period and offset!")

                arr_grid = None # this is only used for inner most dim


        # Get the type signature we are using
        if(not types):
            error_str = "Could not find a type signature matching the types of the variables given for " + func.__name__ + str(tuple(args))
            for param, name in zip(self.params, args):
                if not isinstance(param, np.ndarray): continue
                error_str += '\n' + name + ': ' + str(param.dtype)
            raise ProcessingChainError(error_str)
        # Use the first types in the list that all our types can be cast to
        self.types = types[0]
        types = [np.dtype(t) for t in self.types]

        # Finish setting up of input parameters for function
        #   Reshape variable arrays to add broadcast dimensions
        #   Allocate new arrays as needed
        #   Convert coords to right system of units as needed
        for i, (param, dims, dtype) in enumerate(zip(self.params, dims_list, types)):
            dim_list = [d for d in outerdims] + \
                [dims_dict[d.strip()] for d in dims.split(',') if d]
            shape = tuple([d.length for d in dim_list])
            this_grid = dim_list[-1].grid if dim_list else None
            
            if isinstance(param, str):
                # Create a new variable with the right dimensions and such
                param = self.proc_chain.add_variable(param, dtype=np.dtype(dtype), shape=shape, grid=this_grid, unit=None, is_coord=False)
                self.params[i] = param
                self.raw_params.append(param.buffer)
                
            elif isinstance(param, ProcChainVar):
                # Deduce any automated descriptions of parameter
                unit = None
                is_coord = False
                if param.is_coord==True and grid is not None:
                    unit = str(grid.period.u)
                elif isinstance(param.unit, str) and param.unit in ureg and grid is not None and ureg.is_compatible_with(grid.period, param.unit):
                    is_coord = True

                param.update_auto(shape=shape,
                                  dtype=np.dtype(dtype),
                                  grid=this_grid,
                                  unit=unit,
                                  is_coord=is_coord)

                if param.is_coord and not grid:
                    grid = param.unit
                raw_param = param.get_buffer(grid)
                
                # reshape just in case there are some missing dimensions
                arshape = list(raw_param.shape)
                for idim in range(-1, -1-len(shape), -1):
                    if arshape[idim]!=shape[idim]:
                        arshape.insert(len(arshape)+idim+1, 1)
                self.raw_params.append(raw_param.reshape(tuple(arshape)))
                
            else:
                # Convert scalar to right type, including units
                if isinstance(param, (Quantity, Unit)):
                    if grid is None or not ureg.is_compatible_with(grid.period, param):
                        raise ProcessingChainError("Could not find valid conversion for " + str(param))
                    param = float(param/grid.period)
                if np.issubdtype(dtype, np.integer):
                    self.raw_params.append(dtype.type(round(param)))
                else:
                    self.raw_params.append(dtype.type(param))
        
        self.proc_chain._print(2, 'Added processor:', str(self))

    def execute(self):
        self.processor(*self.raw_params)

    def __str__(self):
        return self.processor.__name__ + '(' \
            + ", ".join([str(par) for par in self.params]) + ')'
        


# A special processor manager for handling converting variables between unit systems
class UnitConvertionManager(ProcessorManager):
    @vectorize(nopython=True, cache=True)
    def convert(buf_in, offset_in, offset_out, period_ratio):
        return (buf_in - offset_in) * period_ratio + offset_out
    
    def __init__(self, var, unit):
        # reference back to our processing chain
        self.proc_chain = var.proc_chain
        # callable function used to process data
        self.processor = UnitConversionManager.convert
        # list of parameters prior to converting to internal representation
        self.params = [var, unit]
        # list of raw values and buffers from params; we will fill this soon
        self.raw_params = []
        
        from_buffer, from_grid = var._buffer[0]
        period_ratio = from_grid.get_period(unit.period)
        self.out_buffer = np.zeros_like(from_buffer)
        self.raw_params = [from_buffer, from_grid.get_offset(),
                           unit.get_offset(), period_ratio, to_buffer]
        self.proc_chain._print(2, 'Added conversion:', str(self))
        
##########################################################################
# Now, classes to manage I/O from buffers into ProcessingChain variables #
##########################################################################

# Base class. IOManagers will be associated with a type of input/output
#  buffer, and must define a read and write for each one. __init__ methods
#  should update variable with any information from buffer, and check that
#  buffer and variable are compatible.
class IOManager(metaclass=ABCMeta):
    @abstractmethod
    def read(self, start: int, end: int):
        pass

    @abstractmethod
    def write(self, start: int, end: int):
        pass

    @abstractmethod
    def __str__(self):
        pass


# Ok, this one's not LGDO
class NumpyIOManager(IOManager):
    def __init__(self, io_buf, var):
        assert isinstance(io_buf, np.ndarray) \
            and isinstance(var, ProcChainVar)
        
        var.update_auto(dtype = io_buf.dtype,
                        shape = io_buf.shape[1:])
        
        if var.shape != io_buf.shape[1:] or var.dtype != io_buf.dtype:
            raise ProcessingChainError("numpy.array<{}>\{{}\}@{}) is not compatible with variable {}".format(self.io_buf.shape, self.io_buf.dtype, self.io_buf.data, str(self.var)))

        self.io_buf = io_buf
        self.var = var
        self.raw_var = var.buffer

    def read(self, start, end):
        np.copyto(self.raw_var[0:end-start, ...],
                  self.io_buf[start:end, ...], 'unsafe')

    def write(self, start, end):
        np.copyto(self.io_buf[start:end, ...],
                  self.raw_var[0:end-start, ...], 'unsafe')

    def __str__(self):
        return '{} linked to numpy.array({}, {})@{})'.format(str(self.var), self.io_buf.shape, self.io_buf.dtype, self.io_buf.data)


class LGDOArrayIOManager(IOManager):
    def __init__(self, io_array, variable):
        assert isinstance(io_buffer, np.ndarray) \
            and isinatance(variable, ProcChainVar)

        unit = io_array.attrs.get('units', None)
        var.update_auto(dtype = io_array.dtype,
                        shape = io_array.nda.shape[1:],
                        unit = unit )
        
        if var.shape != io_array.nda.shape[1:] or var.dtype != io_array.dtype:
            raise ProcessingChainError('LGDO object {}@{} is incompatible with {}'.format(self.io_buf.form_datatype(), self.raw_buf.data, str(self.var)))

        if isinstance(var.unit, CoordinateGrid):
            if unit is None:
                unit = var.unit.period.u
            elif ureg.is_compatible_with(var.unit.period, unit):
                unit = ureg.Quantity(unit).u
            else:
                raise ProcessingChainError("LGDO array and variable {} have incompatible units ({} and {})".format(str(var), str(var.unit.period.u), str(unit)))
        
        if unit is None and not var.unit is None:
            io_array.attrs['units'] = str(grid.period)
        ureg.is_compatible_with(grid.period, param.unit)
        self.io_array = io_array
        self.raw_buf = io_array.nda
        self.var = var.get_buffer(unit)
        

    def read(self, start, end):
        np.copyto(self.raw_var[0:end-start, ...],
                  self.raw_buf[start:end, ...], 'unsafe')

    def write(self, start, end):
        np.copyto(self.raw_buf[start:end, ...],
                  self.raw[0:end-start, ...], 'unsafe')

    def __str__(self):
        return '{} linked to {}'.format(str(self.var), str(self.io_buf))


# Waveforms
class LGDOWaveformIOManager(IOManager):
    def __init__(self, wf_table: WaveformTable, variable: ProcChainVar):
        assert isinstance(wf_table, WaveformTable) \
            and isinstance(variable, ProcChainVar)

        self.wf_table = wf_table
        self.wf_buf = wf_table.values.nda
        self.t0_buf = wf_table.t0.nda
        self.dt_buf = wf_table.dt.nda

        period = wf_table.dt_units
        if isinstance(period, str) and period in ureg:
            grid = CoordinateGrid(ureg.Quantity(self.dt_buf[0], period), self.t0_buf[0])
        else:
            grid = None
        
        self.var = variable
        self.var.update_auto(shape = self.wf_buf.shape[1:],
                             dtype = self.wf_buf.dtype,
                             grid = grid,
                             unit = wf_table.values_units,
                             is_coord = False)
        
        self.wf_var = self.var.buffer
        self.t0_var = self.var.grid.get_offset(wf_table.t0_units)

    def read(self, start, end):
        self.wf_var[0:end-start, ...] = self.wf_buf[start:end, ...]

    def write(self, start, end):
        self.wf_buf[start:end, ...] = self.wf_var[0:end-start, ...]
        self.t0_buf[start:end, ...] = self.t0_var[0:end-start, ...]

    def __str__(self):
        return  '{} linked to {}'.format(str(self.var), str(self.wf_table))
    


def build_processing_chain(lh5_in, dsp_config, db_dict = None,
                           outputs = None, verbosity=1, block_width=16):
    """
    Produces a ProcessingChain object and an lh5 table for output parameters
    from an input lh5 table and a json recipe.

    Returns (proc_chain, lh5_out):
    - proc_chain: ProcessingChain object that is bound to lh5_in and lh5_out;
      all you need to do is handle file i/o for lh5_in/out and run execute
    - lh5_out: output LH5 table

    Required arguments:
    - lh5_in: input LH5 table
    - config: dict or name of json file containing a recipe for
      constructing the ProcessingChain object produced by this function.
      config is formated as a json dict with different processors. Config
      should have a dictionary called processors, containing dictionaries
      of the following format:
        Key: parameter name: name of parameter produced by the processor.
             can optionally provide multiple, separated by spaces
        Values:
          processor (req): name of gufunc
          module (req): name of module in which to find processor
          prereqs (req): name of parameters from other processors and from
            input that are required to exist to run this
          args (req): list of arguments for processor, with variables passed
            by name or value. Names should either be inputs from lh5_in, or
            parameter names for other processors. Names of the format db.name
            will look up the parameter in the metadata.
          kwargs (opt): kwargs used when adding processors to proc_chain
          init_args (opt): args used when initializing a processor that has
            static data (for factory functions)
          default (opt): default value for db parameters if not found
          unit (opt): unit to be used for attr in lh5 file.
      There may also be a list called 'outputs', containing a list of parameters
      to put into lh5_out.

    Optional keyword arguments:
    - outputs: list of parameters to put in the output lh5 table. If None,
      use the parameters in the 'outputs' list from config
    - db_dict: a nested dict pointing to values for db args.
      e.g. if a processor uses arg db.trap.risetime, it will look up
          db_dict['trap']['risetime']
      and use the found value. If no value is found, use the default defined
      in the config file.
    - verbosity: verbosity level:
            0: Print nothing (except errors...)
            1: Print basic warnings (default)
            2: Print basic debug info
    - block_width: number of entries to process at once.
    """
    proc_chain = ProcessingChain(block_width, lh5_in.size, verbosity = verbosity)

    if isinstance(dsp_config, str):
        with open(dsp_config) as f:
            dsp_config = json.load(f)
    elif dsp_config is None:
        dsp_config = {'outputs':[], 'processors':{}}
    else:
        # We don't want to modify the input!
        dsp_config = deepcopy(dsp_config)

    if outputs is None:
        outputs = dsp_config['outputs']

    processors = dsp_config['processors']

    # prepare the processor list
    multi_out_procs = {}
    for key, node in processors.items():
        # if we have multiple outputs, add each to the processesors list
        keys = [k for k in re.split(",| ", key) if k!='']
        if len(keys)>1:
            for k in keys:
                multi_out_procs[k] = key

        # parse the arguments list for prereqs, if not included explicitly
        if not 'prereqs' in node:
            prereqs = []
            for arg in node['args']:
                if not isinstance(arg, str): continue
                for prereq in proc_chain.get_variable(arg, True):
                    if prereq not in prereqs and prereq not in keys and prereq != 'db':
                        prereqs.append(prereq)
            node['prereqs'] = prereqs

        if verbosity>=2:
            print("Prereqs for", key, "are", node['prereqs'])

    processors.update(multi_out_procs)

    # Recursive function to crawl through the parameters/processors and get
    # a sequence of unique parameters such that parameters always appear after
    # their dependencies. For parameters that are not produced by the ProcChain
    # (i.e. input/db parameters), add them to the list of leafs
    # https://www.electricmonk.nl/docs/dependency_resolving_algorithm/dependency_resolving_algorithm.html
    def resolve_dependencies(par, resolved, leafs, unresolved=[]):
        if par in resolved:
            return
        elif par in unresolved:
            raise ProcessingChainError('Circular references detected: %s -> %s' % (par, edge))

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

        edges = node['prereqs']
        unresolved.append(par)
        for edge in edges:
            resolve_dependencies(edge, resolved, leafs, unresolved)
        resolved.append(par)
        unresolved.remove(par)

    proc_par_list = [] # calculated from processors
    input_par_list = [] # input from file and used for processors
    copy_par_list = [] # copied from input to output
    out_par_list = []
    for out_par in outputs:
        if out_par not in processors:
            copy_par_list.append(out_par)
        else:
            resolve_dependencies(out_par, proc_par_list, input_par_list)
            out_par_list.append(out_par)

    if verbosity>0:
        print('Processing parameters:', str(proc_par_list))
        print('Required input parameters:', str(input_par_list))
        print('Copied output parameters:', str(copy_par_list))
        print('Processed output parameters:', str(out_par_list))

    # Now add all of the input buffers from lh5_in (and also the clk time)
    for input_par in input_par_list:
        buf_in = lh5_in.get(input_par)
        if buf_in is None:
            print("I don't know what to do with " + input_par + ". Building output without it!")
        proc_chain.link_input_buffer(input_par, buf_in)

    # now add the processors
    for proc_par in proc_par_list:
        recipe = processors[proc_par]
        module = importlib.import_module(recipe['module'])
        func = getattr(module, recipe['function'])
        args = recipe['args']

        # Initialize the new variables, if needed
        if 'unit' in recipe:
            new_vars = [k for k in re.split(",| ", recipe) if k!='']
            for i, name in enumerate(new_vars):
                unit = recipe.get('unit', auto)
                if isinstance(unit, list): unit = unit[i]
                
                proc_chain.add_variable(name, unit=unit)
            
        # Parse the list of args
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg[0:3]=='db.':
                lookup_path = arg[3:].split('.')
                try:
                    node = db_dict
                    for key in lookup_path:
                        node = node[key]
                    args[i] = node
                    if(verbosity>0):
                        print("Database lookup: found", node, "for", arg)
                except (KeyError, TypeError):
                    try:
                        args[i] = recipe['defaults'][arg]
                        if(verbosity>0):
                            print("Database lookup: using default value of", args[i], "for", arg)
                    except (KeyError, TypeError):
                        raise ProcessingChainError('Did not find', arg, 'in database, and could not find default value.')

        # get this list of kwargs
        kwargs = recipe.get('kwargs', {}) # might also need db lookup here
        
        # if init_args are defined, parse any strings and then call func
        # as a factory/constructor function
        try:
            init_args = recipe['init_args']
            for i, arg in enumerate(init_args):
                if isinstance(arg, str) and arg[0:3]=='db.':
                    lookup_path = arg[3:].split('.')
                    try:
                        node = db_dict
                        for key in lookup_path:
                            node = node[key]
                        init_args[i] = node
                        if(verbosity>0):
                            print("Database lookup: found", node, "for", arg)
                    except (KeyError, TypeError):
                        try:
                            init_args[i] = recipe['defaults'][arg]
                            if(verbosity>0):
                                print("Database lookup: using default value of", init_args[i], "for", arg)
                        except (KeyError, TypeError):
                            raise ProcessingChainError('Did not find', arg, 'in database, and could not find default value.')
                    arg = init_args[i]

                # see if string can be parsed by proc_chain
                if isinstance(arg, str):
                    init_args[i] = proc_chain.get_variable(arg)

            if(verbosity>1):
                print("Building function", func.__name__, "from init_args", init_args)
            func = func(*init_args)
        except KeyError:
            pass
        proc_chain.add_processor(func, *args, **kwargs)


    # build the output buffers
    lh5_out = lh5.Table(size=proc_chain._buffer_len)

    # add inputs that are directly copied
    for copy_par in copy_par_list:
        buf_in = lh5_in.get(copy_par)
        if buf_in is None:
            print("I don't know what to do with " + input_par + ". Building output without it!")
        else:
            lh5_out.add_field(copy_par, buf_in)
    
    # finally, add the output buffers to lh5_out and the proc chain
    for out_par in out_par_list:
        buf_out = proc_chain.link_output_buffer(out_par)
        lh5_out.add_field(out_par, lh5.Array(buf_out, attrs={"units":unit}) )
    
    return (proc_chain, lh5_out)

