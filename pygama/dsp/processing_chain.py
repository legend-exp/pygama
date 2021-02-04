import numpy as np
import json
import re
import ast
import itertools as it
import importlib
from copy import deepcopy
from scimath.units import convert
from scimath.units.api import unit_parser
from scimath.units.unit import unit

from pygama.core.units import *
from pygama import lh5

ast_ops_dict = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
                ast.Div: np.divide, ast.FloorDiv: np.floor_divide,
                ast.USub: np.negative}


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
    def __init__(self, block_width=8, buffer_len=None, clock_unit=None, verbosity=1):
        """Named arguments:
        - block_width: number of entries to simultaneously process.
        - buffer_len: length of input and output buffers. Should be a multiple
            of block_width
        - clock_unit: period or frequency of the clock for all waveforms.
            constant values with time units will be converted to this unit.
        - verbosity: integer indicating the verbosity level:
            0: Print nothing (except errors...)
            1: Print basic warnings (default)
            2: Print basic debug info
            3: Print friggin' everything!
        """
        # Dictionary with numpy arrays containing input and output variables
        self.__vars_dict = {}
        # Ordered list of processors and a tuple containing the bound parameters
        # as either constants or variables from vars_dict
        self.__proc_list = []
        self.__proc_strs = []
        # lists of tuple pairs of external buffers and internal buffers
        self.__input_buffers = {}
        # strings of input transforms and variable names for printings
        self.__output_buffers = {}

        self._block_width = block_width
        self._buffer_len = buffer_len
        self._clk = clock_unit
        self._verbosity = verbosity
        
        
    def add_waveform(self, name, dtype, length):
        """Add named variable containing a waveform block with fixed type and length"""
        self.__add_variable(name, dtype, (self._block_width, length))


    def add_scalar(self, name, dtype):
        """Add named variable containing a scalar block with fixed type"""
        self.__add_variable(name, dtype, (self._block_width))


    def add_input_buffer(self, varname, buff, dtype=None, buffer_len=None, unit=None):
        """Link an input buffer to a variable. The buffer should be a numpy
        ndarray with length that is a multiple of the buffer length, the block
        width, and the named variable length. If any of these are unknown we
        will try to deduce the right lengths. varname can be of the form:
          "varname(length, type)[range]"
        The optional (length, type) field is used to initialize a new variable.
        The optional [range] field copies from the buffer into a subrange of the
        array.

        Optional keyword args:
        - buffer_len: number of entries in buffer. This is needed if we cannot
          figure it out from the shape (i.e. multi-dimensional buffer fed as
          a one-dimensional array)
        - dtype: data type used for the variable. Use this if the variable is
          automatically allocated here, but with a different type from buff
        - unit specifies a unit to convert the input from. Unit must have same type as clk, or be a ratio to multiply the input by
        """
        self.__add_io_buffer(buff, varname, True, dtype, buffer_len, unit)


    def get_input_buffer(self, varname, dtype=None, buffer_len=None, unit=None):
        """Get the input buffer associated with varname. If there is no such
        buffer, create one with a shape compatible with the input variable and
        return it. The input varname has the form:
          "varname(length, type)[range]"
        The optional (length, type) field is used to initialize a new variable.
        The optional [range] field copies a subrange of the named array into the
        output buffer.

        Optional keyword arguments:
        - dtype can be used to specify the numpy data type of the buffer if it
          should differ from that held in varname
        - buffer_len specifies the buffer length, if it has not already been set
        - unit specifies a unit to convert the input from. Unit must have same type as clk, or be a ratio to multiply the input by
        """
        name = re.search('(\w+)', varname).group(0)
        if name not in self.__input_buffers:
            self.__add_io_buffer(None, varname, True, dtype, buffer_len, unit)
        return self.__input_buffers[name][0]


    def add_output_buffer(self, varname, buff, dtype=None, buffer_len=None, unit=None):
        """Link an output buffer to a variable. The buffer should be a numpy
        ndarray with length that is a multiple of the buffer length, the block
        width, and the named variable length. If any of these are unknown we
        will try to deduce the right lengths. varname can be of the form:
          "varname(length, type)[range]"
        The optional (length, type) field is used to initialize a new variable.
        The optional [range] field copies a subrange of the named array into the
        output buffer.

        Optional keyword args:
        - buffer_len: number of entries in buffer. This is needed if we cannot
          figure it out from the shape (i.e. multi-dimensional buffer fed as
          a one-dimensional array)
        - dtype: data type used for the variable. Use this if the variable is
          automatically allocated here, but with a different type from buff
        - unit specifies a unit to convert the output into. Unit must have same type as clk, or be a ratio to divide the output by

        """
        self.__add_io_buffer(buff, varname, False, dtype, buffer_len, unit)


    def get_output_buffer(self, varname, dtype=None, buffer_len=None, unit=None):
        """Get the output buffer associated with varname. If there is no such
        buffer, create one with a shape compatible with the input variable and
        return it. The input varname has the form:
          "varname(length, type)[range]"
        The optional (length, type) field is used to initialize a new variable.
        The optional [range] field copies a subrange of the named array into the
        output buffer.

        Optional keyword arguments:
        - dtype can be used to specify the numpy data type of the buffer if it
          should differ from that held in varname
        - buffer_len specifies the buffer length, if it has not already been set
        - unit specifies a unit to convert the output into. Unit must have same type as clk, or be a ratio to divide the output by
        """
        name = re.search('(\w+)', varname).group(0)
        if name not in self.__output_buffers:
            self.__add_io_buffer(None, varname, False, dtype, buffer_len, unit)
        return self.__output_buffers[name][0]


    def add_processor(self, func, *args, **kwargs):
        """
        Add a new processor and bind it to a set of parameters.
        - func should be a function implementing the numpy ufunc class. If not,
          then the signature and types keyword arguments will be necessary.
        - args is a list of names and constants. The names link to internal
          numpy array variables that will be bound to the function. Names can
          be given in the form:
            "varname(length, type)[range]"
          The optional (length, type) field is used to initialize a new variable
          if necessary. The optional [range] field copies binds only a subrange
          of the array to the function. Non-string constant values will be
          converted to the right type and bound to the function. Constants with
          scimath time units will be converted to the internal clock unit.
        - keyword arguments include:
          - signature: broadcasting signature for a ufunc. By default, use
            func.signature
          - types: a list of strings defining the types of arrays needed for
            the func. By default, use func.types
        """

        # Get the signature and list of valid types for the function
        signature = kwargs.get("signature", None)
        if(signature == None): signature = func.signature
        if(signature == None):
            signature = '->'
            for i in range(func.nin): signature = ',()'+signature
            for i in range(func.nout): signature = signature+'(),'
            signature = signature.strip(',')

        types = kwargs.get("types", None)
        if(types == None): types = func.types.copy()
        if(types == None):
            raise TypeError("Could not find a type signature list for " + func.__name__ + ". Please supply a valid list of types.")
        for i, typestr in enumerate(types):
            types[i]=typestr.replace('->', '')

        # make a list of parameters from *args. Replace any strings in the list
        # with numpy objects from vars_dict, where able
        params = []
        for i, param in enumerate(args):
            if(isinstance(param, str)):
                param_val = self.get_variable(param)
                if param_val is not None:
                    param=param_val
            params.append(param)

        # Make sure arrays obey the broadcasting rules, and make a dictionary
        # of the correct dimensions
        dims_list = re.findall("\((.*?)\)", signature)
        dims_dict = {}
        outerdims = []
        # print('\nsetting param:', params[-1])
        # print('dims list is:', dims_list)
        for ipar, dims in enumerate(dims_list):
            # print(ipar, dims, params[ipar])
            if not isinstance(params[ipar], np.ndarray):
                continue

            fun_dims = outerdims + [d.strip() for d in dims.split(',') if d]
            arr_dims = list(params[ipar].shape)
            #check if arr_dims can be broadcast to match fun_dims
            for i in range(max(len(fun_dims), len(arr_dims))):
                fd = fun_dims[-i-1] if i<len(fun_dims) else None
                ad = arr_dims[-i-1] if i<len(arr_dims) else None
                if(isinstance(fd, str)):
                    # Define the dimension or make sure it is consistent
                    if not ad or dims_dict.setdefault(fd, ad)!=ad:
                        raise ValueError("Failed to broadcast array dimensions for "+func.__name__+". Could not find consistent value for dimension "+fd)
                elif not fd:
                    # if we ran out of function dimensions, add a new outer dim
                    outerdims.insert(0, ad)
                elif not ad:
                    continue
                elif fd != ad:
                    # If dimensions disagree, either insert a broadcasted array dimension or raise an exception
                    if len(fun_dims)>len(arr_dims):
                        arr_dims.insert(len(arr_dims)-i, 1)
                    elif len(fun_dims)<len(arr_dims):
                        outerdims.insert(len(fun_dims)-i, ad)
                        fun_dims.insert(len(fun_dims)-i, ad)
                    else:
                        raise ValueError("Failed to broadcast array dimensions for "+func.__name__+". Input arrays do not have consistent outer dimensions.")

            # find type signatures that match type of array
            arr_type = params[ipar].dtype.char
            types = [type_sig for type_sig in types if arr_type==type_sig[ipar]]

        # Get the type signature we are using
        if(not types):
            raise TypeError("Could not find a type signature matching the types of the variables given for " + func.__name__)
        elif(len(types)>1):
            self.__print(1, "Found multiple compatible type signatures for this function:", types, "Using signature " + types[0] + ".")
        types = [np.dtype(t) for t in types[0]]

        # Reshape variable arrays to add broadcast dimensions and allocate new arrays as needed
        for i, param, dims, dtype in zip(range(len(params)), params, dims_list, types):
            shape = outerdims + [dims_dict[d.strip()] for d in dims.split(',') if d]
            if isinstance(param, str):
                params[i] = self.__add_var(param, dtype, shape)
            elif isinstance(param, np.ndarray):
                arshape = list(param.shape)
                for idim in range(-1, -1-len(shape), -1):
                    if arshape[idim]!=shape[idim]:
                        arshape.insert(len(arshape)+idim+1, 1)
                params[i] = param.reshape(tuple(arshape))
            else:
                if isinstance(param, unit):
                    param = convert(1, param, self._clk)
                if np.issubdtype(dtype, np.integer):
                    params[i] = dtype.type(round(param))
                else:
                    params[i] = dtype.type(param)

        # Make strings of input variables/constants for later printing
        proc_strs = []
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                proc_strs.append(arg)
            else:
                proc_strs.append(str(params[i]))
        proc_strs = tuple(proc_strs)

        self.__print(2, 'Added processor: ' + func.__name__ + str(proc_strs).replace("'", ""))

        # Add the function and bound parameters to the list of processors
        self.__proc_list.append((func, tuple(params)))
        self.__proc_strs.append(proc_strs)
        return None


    def execute(self, start=0, end=None):
        """Execute the dsp chain on the entire input/output buffers"""
        if end is None: end = self._buffer_len
        for begin in range(start, end, self._block_width):
            self.execute_block(begin)


    def execute_block(self, offset=0):
        """Execute the dsp chain on a sub-set of the input/output buffers
        starting at entry offset, with length equal to the internal block size.
        """
        end = min(offset+self._block_width, self._buffer_len)
        if self._verbosity<3:
            self.__execute_procs(offset, end)
        else:
            self.__execute_procs_verbose(offset, end)


    def get_variable(self, varname):
        """Get the numpy array holding the internal memory buffer used for a
        named variable. The varname has the format
          "varname(length, type)[range]"
        The optional (length, type) field is used to initialize a new variable
        if necessary. The optional [range] field fetches only a subrange
        of the array to the function."""
        return self.__parse_expr(ast.parse(varname, mode='eval').body)

    
    def __parse_expr(self, node):
        """
        helper function for get_variable that recursively evaluates the AST tree
        based on: https://stackoverflow.com/a/9558001.
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
            val = unit_parser.parse_unit(node.id)
            if val.is_valid():
                return convert(1, val, self._clk)

            #check if it is a variable
            val = self.__vars_dict.get(node.id, None)
            return val

        # define binary operators (+,-,*,/)
        elif isinstance(node, ast.BinOp):
            lhs = self.__parse_expr(node.left)
            rhs = self.__parse_expr(node.right)
            op = ast_ops_dict[type(node.op)]
            if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
                if not isinstance(lhs, np.ndarray):
                    if np.issubdtype(rhs.dtype, np.integer):
                        lhs = rhs.dtype.type(round(lhs))
                    else:
                        lhs = rhs.dtype.type(lhs)
                if not isinstance(rhs, np.ndarray):
                    if np.issubdtype(lhs.dtype, np.integer):
                        rhs = lhs.dtype.type(round(rhs))
                    else:
                        rhs = lhs.dtype.type(rhs)
                out = op(lhs, rhs)
                self.__proc_list.append((op, (lhs, rhs, out)))
                self.__proc_strs.append("Binary operator: " + op.__name__)
                return out
            return op(lhs, rhs)

        # define unary operators (-)
        elif isinstance(node, ast.UnaryOp):
            operand = self.__parse_expr(node.operand)
            op = ast_ops_dict[type(node.op)]
            out = op(operand)
            # if we have a np array, add to processor list
            if isinstance(out, np.ndarray):
                self.__proc_list.append((op, (operand, out)))
                self.__proc_strs.append("Unary operator: " + op.__name__)
            return out

        elif isinstance(node, ast.Subscript):
            # print(ast.dump(node))
            val = self.__parse_expr(node.value)
            if isinstance(node.slice, ast.Index):
                if isinstance(val, np.ndarray):
                    return val[..., self.__parse_expr(node.slice.value)]
                else:
                    return val[self.__parse_expr(node.slice.value)]
            elif isinstance(node.slice, ast.Slice):
                if isinstance(val, np.ndarray):
                    return val[..., slice(self.__parse_expr(node.slice.lower),
                                          self.__parse_expr(node.slice.upper),
                                          self.__parse_expr(node.slice.step) )]
                else:
                    print(self.__parse_expr(node.slice.upper))
                    return val[slice(self.__parse_expr(node.slice.upper),self.__parse_expr(node.slice.lower),self.__parse_expr(node.slice.step))]
            elif isinstance(node.slice, ast.ExtSlice):
                slices = tuple(node.slice.dims)
                for i, sl in enumerate(slices):
                    if isinstance(sl, ast.index):
                        slices[i] = self.__parse_expr(sl.value)
                    else:
                        slices[i] = slice(self.__parse_expr(sl.upper),
                                          self.__parse_expr(sl.lower),
                                          self.__parse_expr(sl.step) )
                return val[..., slices]

        # for name.attribute
        elif isinstance(node, ast.Attribute):
            val = self.__parse_expr(node.value)
            # get shape with buffer_len dimension removed
            if node.attr=='shape' and isinstance(val, np.ndarray):
                return val.shape[1:]

        # for func([args])
        elif isinstance(node, ast.Call):
            func = node.func.id
            # get length of 1D array variable
            if func=="len" and len(node.args)==1 and isinstance(node.args[0], ast.Name):
                var = self.__parse_expr(node.args[0])
                if isinstance(var, np.ndarray) and len(var.shape)==2:
                    return var.shape[1]
                else:
                    raise ValueError("len(): " + node.args[0].id + "has wrong number of dims")
            elif func=="round" and len(node.args)==1:
                var = self.__parse_expr(node.args[0])
                return int(round(var))
            # if this is a valid call to construct a new array, do so; otherwise raise an exception
            else:
                if len(node.args)==2:
                    shape = self.__parse_expr(node.args[0])
                    if isinstance(shape, (int, np.int32, np.int64)):
                        shape = (self._block_width, shape)
                    elif isinstance(shape, tuple):
                        shape = (self._block_width, ) + shape
                    else:
                        raise ValueError("Do not recognize call to "+func+" with arguments of types " + str([arg.__dict__ for arg in node.args]))
                    try: dtype = np.dtype(node.args[1].id)
                    except: raise ValueError("Do not recognize call to "+func+" with arguments of types " + str([arg.__dict__ for arg in node.args]))

                    if func in self.__vars_dict:
                        var = self.__vars_dict[func]
                        if not var.shape==shape and var.dtype==dtype:
                            raise ValueError("Requested shape and type for " + func + " do not match existing values")
                        return var
                    else:
                        var = np.zeros(shape, dtype, 'F')
                        self.__vars_dict[func] = var
                        self.__print(2, 'Added variable ' + func + ' with shape ' + str(tuple(shape)) + ' and type ' + str(dtype))

                        return var
                else:
                    raise ValueError("Do not recognize call to "+func+" with arguments " + str([str(arg.__dict__) for arg in node.args]))

        raise ValueError("Cannot parse AST nodes of type " + str(node.__dict__))


    def __add_var(self, name, dtype, shape):
        """
        Add an array of zeros to the vars dict called name and return it
        """
        if not re.match("\A\w+$", name):
            raise KeyError(name+' is not a valid alphanumeric name')
        if name in self.__vars_dict:
            raise KeyError(name+' is already in variable list')
        arr = np.zeros(shape, dtype)
        self.__vars_dict[name] = arr
        self.__print(2, 'Added variable ' + re.search('(\w+)', name).group(0) + ' with shape ' + str(tuple(shape)) + ' and type ' + str(dtype))
        return arr


    def __execute_procs(self, start, end):
        """
        copy from input buffers to variables
        call all the processors on their paired arg tuples
        copy from variables to list of output buffers
        """
        for buf, var, scale in self.__input_buffers.values():
            if scale:
                np.multiply(buf[start:end, ...], scale, var[0:end-start, ...])
            else:
                np.copyto(var[0:end-start, ...], buf[start:end, ...], 'unsafe')
        for func, args in self.__proc_list:
            func(*args)
        for buf, var, scale in self.__output_buffers.values():
            if scale:
                np.divide(var[0:end-start, ...], scale, buf[start:end, ...])
            else:
                np.copyto(buf[start:end, ...], var[0:end-start, ...], 'unsafe')

    
    def __execute_procs_verbose(self, start, end):
        """
        verbose version of __execute_procs. This is probably overkill, but it
        was done to minimize python calls in the non-verbose version
        """
        names = set(self.__vars_dict.keys())
        self.__print(3, 'Input:')
        for name, (buf, var, scale) in self.__input_buffers.items():
            if scale:
                np.multiply(buf[start:end, ...], scale, var[0:end-start, ...])
            else:
                np.copyto(var[0:end-start, ...], buf[start:end, ...], 'unsafe')
            self.__print(3, name+' = '+str(var))
            names.discard(name)

        self.__print(3, 'Processing:')
        for (func, args), strs in zip(self.__proc_list, self.__proc_strs):
            func(*args)
            self.__print(3, func.__name__ + str(strs).replace("'", ""))
            for name, arg in zip(strs, args):
                try:
                    names.remove(name)
                    self.__print(3, name+' = '+str(arg))
                except: pass

        self.__print(3, 'Output:')
        for name, (buf, var, scale) in self.__output_buffers.items():
            if scale:
                np.divide(var[0:end-start, ...], scale, buf[start:end, ...])
            else:
                np.copyto(buf[start:end, ...], var[0:end-start, ...], 'unsafe')
            self.__print(3, name+' = '+str(var))

    
    def __add_io_buffer(self, buff, varname, input, dtype, buffer_len, scale):
        """
        append a tuple with the buffer and variable to either the input buffer
        list (if input=true) or output buffer list (if input=false), making sure
        that buffer shapes are compatible
        """
        var = self.get_variable(varname)
        if buff is not None and not isinstance(buff, np.ndarray):
            raise ValueError("Buffers must be ndarrays.")

        # if buffer length is not defined, figure out what it should be
        if buffer_len is not None:
            if self._buffer_len is None:
                self._buffer_len = buffer_len
            elif self._buffer_len != buffer_len:
                raise ValueError("Buffer length was already set to a number different than the one provided. To change the buffer length, you must reset the buffers.")
        if not self._buffer_len:
            if buff is not None: self._buffer_len = buff.shape[0]
            else: self._buffer_len = self._block_width
            self.__print(1, "Setting i/o buffer length to " + str(self._buffer_len))

        # if a unit is given, convert it to a scaling factor
        if isinstance(scale, unit):
            scale = convert(1, scale, self._clk)

        # if no buffer was provided, make one
        returnbuffer=False
        if buff is None:
            if var is None:
                raise ValueError("Cannot make buffer for non-existent variable " + varname)
            # deduce dtype. If scale is used, force float type
            if not dtype:
                if not scale:
                    dtype=var.dtype
                else:
                    dtype=np.dtype('float'+str(var.dtype.itemsize*8))
            buff = np.zeros((self._buffer_len,)+var.shape[1:], dtype)
            returnbuffer=True

        # Check that the buffer length is correct. For 1D buffer, reshape
        # it if possible, assuming array of structure ordering
        if not buff.shape[0] == self._buffer_len:
            if buff.ndim==1 and len(buff)%self._buffer_len==0:
                buff = buff.reshape(self._buffer_len, len(buff)//self._buffer_len)
            else:
                raise ValueError("Buffer provided for " + varname + " is the wrong length.")

        # Check that shape of buffer is compatible with shape of variable.
        # If variable does not yet exist, add it here
        if var is None:
            if dtype is None:
                if not scale:
                    dtype=buff.dtype
                else:
                    dtype=np.dtype('float'+str(buff.dtype.itemsize*8))
            var = self.__add_var(varname, dtype, (self._block_width,)+buff.shape[1:])
        elif var.shape[1:] != buff.shape[1:]:
            raise ValueError("Provided buffer has shape " + str(buff.shape) + " which is not compatible with " + str(varname) + " shape " + str(var.shape))

        varname = re.search('(\w+)', varname).group(0)
        if input:
            self.__input_buffers[varname]=(buff, var, scale)
            self.__print(2, 'Binding input buffer of shape ' + str(buff.shape) + ' and type ' + str(buff.dtype) + ' to variable ' + varname + ' with shape ' + str(var.shape) + ' and type ' + str(var.dtype))
        else:
            self.__output_buffers[varname]=(buff, var, scale)
            self.__print(2, 'Binding output buffer of shape ' + str(buff.shape) + ' and type ' + str(buff.dtype) + ' to variable ' + varname + ' with shape ' + str(var.shape) + ' and type ' + str(var.dtype))

        if returnbuffer: return buff


    def __print(self, verbosity, *args):
        if self._verbosity >= verbosity:
            print(*args)


    def __str__(self):
        ret = 'Input variables: ' + str([name for name in self.__input_buffers.keys()])
        for proc, strs in zip(self.__proc_list, self.__proc_strs):
            ret += '\n' + proc[0].__name__ + str(strs)
        ret += '\nOutput variables: ' + str([name for name in self.__output_buffers.keys()])
        return ret.replace("'", "")




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
            3: Print friggin' everything!    
    - block_width: number of entries to process at once.
    """
    
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
    
    # for processors with multiple outputs, add separate entries to the processor list
    for key in list(processors):
        keys = [k for k in re.split(",| ", key) if k!='']
        if len(keys)>1:
            for k in keys:
                processors[k] = key
    
    # Recursive function to crawl through the parameters/processors and get
    # a sequence of unique parameters such that parameters always appear after
    # their dependencies. For parameters that are not produced by the ProcChain
    # (i.e. input/db parameters), add them to the list of leafs
    # https://www.electricmonk.nl/docs/dependency_resolving_algorithm/dependency_resolving_algorithm.html
    def resolve_dependencies(par, resolved, leafs, unresolved=[]):
        if par in resolved:
            return
        elif par in unresolved:
            raise Exception('Circular references detected: %s -> %s' % (par, edge))

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
        
    proc_chain = ProcessingChain(block_width, lh5_in.size, verbosity = verbosity)
    
    # Now add all of the input buffers from lh5_in (and also the clk time)
    for input_par in input_par_list:
        buf_in = lh5_in.get(input_par)
        if buf_in is None:
            print("I don't know what to do with " + input_par + ". Building output without it!")
        elif isinstance(buf_in, lh5.Array):
            proc_chain.add_input_buffer(input_par, buf_in.nda)
        elif isinstance(buf_in, lh5.Table):
            # check if this is waveform
            if 't0' and 'dt' and 'values' in buf_in:
                proc_chain.add_input_buffer(input_par, buf_in['values'].nda, 'float32')
                clk = buf_in['dt'].nda[0] * unit_parser.parse_unit(lh5_in['waveform']['dt'].attrs['units'])
                if proc_chain._clk is not None and proc_chain._clk != clk:
                    print("Somehow you managed to set multiple clock frequencies...Using " + str(proc_chain._clk))
                else:
                    proc_chain._clk = clk

    # now add the processors
    for proc_par in proc_par_list:
        recipe = processors[proc_par]
        module = importlib.import_module(recipe['module'])
        func = getattr(module, recipe['function'])
        args = recipe['args']
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
                except:
                    try:
                        args[i] = recipe['defaults'][arg]
                        if(verbosity>0):
                            print("Database lookup: using default value of", args[i], "for", arg)
                    except:
                        raise Exception('Did not find', arg, 'in database, and could not find default value.')
            
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
                    except:
                        try:
                            init_args[i] = recipe['defaults'][arg]
                            if(verbosity>0):
                                print("Database lookup: using default value of", init_args[i], "for", arg)
                        except:
                            raise Exception('Did not find', arg, 'in database, and could not find default value.')
                    arg = init_args[i]

                # see if string can be parsed by proc_chain
                if isinstance(arg, str):
                    try:
                        init_args[i] = proc_chain.get_variable(arg)
                    except:
                        pass
                    
            if(verbosity>1):
                print("Building function", func.__name__, "from init_args", init_args)
            func = func(*init_args)
        except:
            pass
        proc_chain.add_processor(func, *args, **kwargs)

    
    # build the output buffers
    lh5_out = lh5.Table(size=proc_chain._buffer_len)
    
    # add inputs that are directly copied
    for copy_par in copy_par_list:
        buf_in = lh5_in.get(copy_par)
        if isinstance(buf_in, lh5.Array):
            lh5_out.add_field(copy_par, buf_in)
        elif isinstance(buf_in, lh5.Table):
            # check if this is waveform
            if 't0' and 'dt' and 'values' in buf_in:
                lh5_out.add_field(copy_par, buf_in['values'])
                clk = buf_in['dt'].nda[0] * unit_parser.parse_unit(lh5_in['waveform']['dt'].attrs['units'])
                if proc_chain._clk is not None and proc_chain._clk != clk:
                    print("Somehow you managed to set multiple clock frequencies...Using " + str(proc_chain._clk))
                else:
                    proc_chain._clk = clk
        else:
            print("I don't know what to do with " + input_par + ". Building output without it!")
    
    # finally, add the output buffers to lh5_out and the proc chain
    for out_par in out_par_list:
        recipe = processors[out_par]
        # special case for proc with multiple outputs
        if isinstance(recipe, str):
            i = [k for k in re.split(",| ", recipe) if k!=''].index(out_par)
            recipe = processors[recipe]
            unit = recipe['unit'][i]
        else:
            unit = recipe['unit']
        
        try:
            scale = convert(1, unit_parser.parse_unit(unit), clk)
        except InvalidConversion:
            scale = None
        
        buf_out = proc_chain.get_output_buffer(out_par, unit=scale)
        lh5_out.add_field(out_par, lh5.Array(buf_out, attrs={"units":unit}) )
    return (proc_chain, lh5_out)
