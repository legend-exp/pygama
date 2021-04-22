import numpy as np
import json
import re
import ast
import itertools as it

from pygama.dsp.units import *
from pygama.dsp.errors import *

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
            raise ProcessingChainError("Could not find a type signature list for " + func.__name__ + ". Please supply a valid list of types.")
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
                        raise ProcessingChainError("Failed to broadcast array dimensions for "+func.__name__+". Could not find consistent value for dimension "+fd)
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
                        raise ProcessingChainError("Failed to broadcast array dimensions for "+func.__name__+". Input arrays do not have consistent outer dimensions.")

            # find type signatures that array can be cast into
            arr_type = params[ipar].dtype.char
            types = [type_sig for type_sig in types if np.can_cast(arr_type, type_sig[ipar])]

        # Get the type signature we are using
        if(not types):
            error_str = "Could not find a type signature matching the types of the variables given for " + func.__name__ + str(tuple(args))
            for param, name in zip(params, args):
                if not isinstance(param, np.ndarray): continue
                error_str += '\n' + name + ': ' + str(param.dtype)
            raise ProcessingChainError(error_str)
        # Use the first types in the list that all our types can be cast to
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

        self.__print(2, 'Added processor:', func.__name__ + str(proc_strs).replace("'", ""))

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
        self.__execute_procs(offset, end)


    def get_variable(self, expr, get_names_only=False):
        """Parse string expr into a numpy array or value, using the following
        syntax:
          - numeric values are parsed into ints or floats
          - units found in the dsp.units module are parsed into floats
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
        var = self.__parse_expr(ast.parse(expr, mode='eval').body, \
                                not get_names_only, names)
        if not get_names_only: return var
        else: return names

    
    def __parse_expr(self, node, allocate_memory, var_name_list):
        """
        helper function for get_variable that recursively evaluates the AST tree
        based on: https://stackoverflow.com/a/9558001. Whenever we encounter
        a variable name, add it to var_name_list (which should begin as an
        empty list). Only add new variables and processors to the chain if
        allocate_memory is True
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
                try:
                    return convert(1, val, self._clk)
                except:
                    return None

            #check if it is a variable
            var_name_list.append(node.id)
            val = self.__vars_dict.get(node.id, None)
            return val

        # define binary operators (+,-,*,/)
        elif isinstance(node, ast.BinOp):
            lhs = self.__parse_expr(node.left, allocate_memory, var_name_list)
            if lhs is None: return None
            rhs = self.__parse_expr(node.right, allocate_memory, var_name_list)
            if rhs is None: return None
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
                if allocate_memory:
                    self.__proc_list.append((op, (lhs, rhs, out)))
                    self.__proc_strs.append("Binary operator: " + op.__name__)
                return out
            return op(lhs, rhs)

        # define unary operators (-)
        elif isinstance(node, ast.UnaryOp):
            operand = self.__parse_expr(node.operand, allocate_memory, var_name_list)
            if operand is None: return None
            op = ast_ops_dict[type(node.op)]
            out = op(operand)
            # if we have a np array, add to processor list
            if isinstance(out, np.ndarray) and allocate_memory:
                self.__proc_list.append((op, (operand, out)))
                self.__proc_strs.append("Unary operator: " + op.__name__)
            return out

        elif isinstance(node, ast.Subscript):
            # print(ast.dump(node))
            val = self.__parse_expr(node.value, allocate_memory, var_name_list)
            if val is None: return None
            if isinstance(node.slice, ast.Index):
                if isinstance(val, np.ndarray):
                    return val[..., self.__parse_expr(node.slice.value, allocate_memory, var_name_list)]
                else:
                    return val[self.__parse_expr(node.slice.value, allocate_memory, var_name_list)]
            elif isinstance(node.slice, ast.Slice):
                if isinstance(val, np.ndarray):
                    return val[..., slice(self.__parse_expr(node.slice.lower, allocate_memory, var_name_list),
                                          self.__parse_expr(node.slice.upper, allocate_memory, var_name_list),
                                          self.__parse_expr(node.slice.step, allocate_memory, var_name_list) )]
                else:
                    return val[slice(self.__parse_expr(node.slice.upper, allocate_memory, var_name_list),self.__parse_expr(node.slice.lower, allocate_memory, var_name_list),self.__parse_expr(node.slice.step, allocate_memory, var_name_list))]
            elif isinstance(node.slice, ast.ExtSlice):
                slices = tuple(node.slice.dims)
                for i, sl in enumerate(slices):
                    if isinstance(sl, ast.index):
                        slices[i] = self.__parse_expr(sl.value, allocate_memory, var_name_list)
                    else:
                        slices[i] = slice(self.__parse_expr(sl.upper, allocate_memory, var_name_list),
                                          self.__parse_expr(sl.lower, allocate_memory, var_name_list),
                                          self.__parse_expr(sl.step, allocate_memory, var_name_list) )
                return val[..., slices]

        # for name.attribute
        elif isinstance(node, ast.Attribute):
            val = self.__parse_expr(node.value, allocate_memory, var_name_list)
            if val is None: return None
            # get shape with buffer_len dimension removed
            if node.attr=='shape' and isinstance(val, np.ndarray):
                return val.shape[1:]

        # for func([args])
        elif isinstance(node, ast.Call):
            func = node.func.id
            # get length of 1D array variable
            if func=="len" and len(node.args)==1 and isinstance(node.args[0], ast.Name):
                var = self.__parse_expr(node.args[0], allocate_memory, var_name_list)
                if var is None: return None
                if isinstance(var, np.ndarray) and len(var.shape)==2:
                    return var.shape[1]
                else:
                    raise ProcessingChainError("len(): " + node.args[0].id + "has wrong number of dims")
            elif func=="round" and len(node.args)==1:
                var = self.__parse_expr(node.args[0], allocate_memory, var_name_list)
                if var is None: return None
                return int(round(var))
            # if this is a valid call to construct a new array, do so; otherwise raise an exception
            else:
                if len(node.args)==2:
                    shape = self.__parse_expr(node.args[0], allocate_memory, var_name_list)
                    if shape is None:
                        shape = (self._block_width,)
                    if isinstance(shape, (int, np.int32, np.int64)):
                        shape = (self._block_width, shape)
                    elif isinstance(shape, tuple):
                        shape = (self._block_width, ) + shape
                    else:
                        raise ProcessingChainError("Do not recognize call to "+func+" with arguments of types " + str([arg.__dict__ for arg in node.args]))
                    try: dtype = np.dtype(node.args[1].id)
                    except: raise ProcessingChainError("Do not recognize call to "+func+" with arguments of types " + str([arg.__dict__ for arg in node.args]))
                    
                    var_name_list.append(func)
                    
                    if func in self.__vars_dict:
                        var = self.__vars_dict[func]
                        if not var.shape==shape and var.dtype==dtype:
                            raise ProcessingChainError("Requested shape and type for " + func + " do not match existing values")
                        return var
                    else:
                        var = np.zeros(shape, dtype, 'F')
                        if allocate_memory:
                            self.__vars_dict[func] = var
                            self.__print(2, 'Added variable', func, 'with shape', tuple(shape), 'and type', dtype)

                        return var
                else:
                    raise ProcessingChainError("Do not recognize call to "+func+" with arguments " + str([str(arg.__dict__) for arg in node.args]))

        raise ProcessingChainError("Cannot parse AST nodes of type " + str(node.__dict__))


    def __add_var(self, name, dtype, shape):
        """
        Add an array of zeros to the vars dict called name and return it
        """
        if not re.match("\A\w+$", name):
            raise ProcessingChainError(name+' is not a valid alphanumeric name')
        if name in self.__vars_dict:
            raise ProcessingChainError(name+' is already in variable list')
        arr = np.zeros(shape, dtype)
        self.__vars_dict[name] = arr
        self.__print(2, 'Added variable', re.search('(\w+)', name).group(0), 'with shape', tuple(shape), 'and type', dtype)
        return arr

    
    def __execute_procs(self, start, end):
        """
        copy from input buffers to variables
        call all the processors on their paired arg tuples
        copy from variables to list of output buffers
        """
        # Track names that have been printed so we only print each variable once
        if self._verbosity >= 3:
            names = set(self.__vars_dict.keys())
            self.__print(3, 'Input:')

        # Copy input buffers into proc chain buffers
        for name, (buf, var, scale) in self.__input_buffers.items():
            if scale:
                np.multiply(buf[start:end, ...], scale, var[0:end-start, ...])
            else:
                np.copyto(var[0:end-start, ...], buf[start:end, ...], 'unsafe')

            if self._verbosity >= 3:
                self.__print(3, name, '=', var)
                names.discard(name)

        # Loop through processors and run each one
        self.__print(3, 'Processing:')
        for (func, args), strs in zip(self.__proc_list, self.__proc_strs):
            try:
                func(*args)
            except DSPFatal as e:
                e.processor = func.__name__ + str(strs).replace("'", "")
                e.wf_range = (start, end)
                raise e
                
            if self._verbosity >= 3:
                self.__print(3, func.__name__ + str(strs).replace("'", ""))
                for name, arg in zip(strs, args):
                    try:
                        names.remove(name)
                        self.__print(3, name, '=', arg)
                    except: pass

        # copy from processing chain buffers into output buffers
        self.__print(3, 'Output:')
        for name, (buf, var, scale) in self.__output_buffers.items():
            if scale:
                np.divide(var[0:end-start, ...], scale, buf[start:end, ...])
            else:
                np.copyto(buf[start:end, ...], var[0:end-start, ...], 'unsafe')

            self.__print(3, name, '=', var)

    
    def __add_io_buffer(self, buff, varname, input, dtype, buffer_len, scale):
        """
        append a tuple with the buffer and variable to either the input buffer
        list (if input=true) or output buffer list (if input=false), making sure
        that buffer shapes are compatible
        """
        var = self.get_variable(varname)
        if buff is not None and not isinstance(buff, np.ndarray):
            raise ProcessingChainError("Buffers must be ndarrays.")

        # if buffer length is not defined, figure out what it should be
        if buffer_len is not None:
            if self._buffer_len is None:
                self._buffer_len = buffer_len
            elif self._buffer_len != buffer_len:
                raise ProcessingChainError("Buffer length was already set to a number different than the one provided. To change the buffer length, you must reset the buffers.")
        if not self._buffer_len:
            if buff is not None: self._buffer_len = buff.shape[0]
            else: self._buffer_len = self._block_width
            self.__print(1, "Setting i/o buffer length to", self._buffer_len)

        # if a unit is given, convert it to a scaling factor
        if isinstance(scale, unit):
            scale = convert(1, scale, self._clk)

        # if no buffer was provided, make one
        returnbuffer=False
        if buff is None:
            if var is None:
                raise ProcessingChainError("Cannot make buffer for non-existent variable " + varname)
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
                raise ProcessingChainError("Buffer provided for " + varname + " is the wrong length.")

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
            raise ProcessingChainError("Provided buffer has shape " + str(buff.shape) + " which is not compatible with " + str(varname) + " shape " + str(var.shape))

        varname = re.search('(\w+)', varname).group(0)
        if input:
            self.__input_buffers[varname]=(buff, var, scale)
            self.__print(2, 'Binding input buffer of shape', buff.shape, 'and type', buff.dtype, 'to variable', varname, 'with shape', var.shape, 'and type', var.dtype)
        else:
            self.__output_buffers[varname]=(buff, var, scale)
            self.__print(2, 'Binding output buffer of shape', buff.shape, 'and type', buff.dtype, 'to variable', varname, 'with shape', var.shape, 'and type', var.dtype)

        if returnbuffer: return buff


    def __print(self, verbosity, *args, **kwargs):
        """Helper for output that checks verbosity before printing and
        converts things into strings. At verbosity 0, print to stderr"""
        if verbosity==0 and not 'file' in kwargs:
            kwargs['file']=sys.stderr
        if self._verbosity >= verbosity:
            print(*[str(arg) for arg in args], **kwargs)


    def __str__(self):
        ret = 'Input variables: ' + str([name for name in self.__input_buffers.keys()])
        for proc, strs in zip(self.__proc_list, self.__proc_strs):
            ret += '\n' + proc[0].__name__ + str(strs)
        ret += '\nOutput variables: ' + str([name for name in self.__output_buffers.keys()])
        return ret.replace("'", "")

