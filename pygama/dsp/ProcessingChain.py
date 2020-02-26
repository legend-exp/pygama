import numpy as np
import re
import itertools as it
from scimath.units import convert
from scimath.units.unit import unit

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
        self.__vars_dict__ = {}
        # Ordered list of processors and a tuple containing the bound parameters
        # as either constants or variables from vars_dict
        self.__proc_list__ = []
        self.__proc_strs__ = []
        # lists of tuple pairs of external buffers and internal buffers
        self.__input_buffers__ = {}
        # strings of input transforms and variable names for printings
        self.__output_buffers__ = {}
        
        self.__block_width__ = block_width
        self.__buffer_len__ = buffer_len
        self.__clk__ = clock_unit
        self.__verbosity__ = verbosity

    def add_waveform(self, name, dtype, length):
        """Add named variable containing a waveform block with fixed type and length"""
        self.__add_variable__(name, dtype, (self.__block_width__, length))

    def add_scalar(self, name, dtype):
        """Add named variable containing a scalar block with fixed type"""
        self.__add_variable__(name, dtype, (self.__block_width__))

    def add_input_buffer(self, varname, buff, dtype=None, buffer_len=None):
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
        """
        self.__add_io_buffer__(buff, varname, True, dtype, buffer_len)

    def get_input_buffer(self, varname, dtype=None, buffer_len=None):
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
        """
        name = re.search('(\w+)', varname).group(0)
        if name not in self.__input_buffers__:
            self.__add_io_buffer__(None, varname, True, dtype, buffer_len)
        return self.__input_buffers__[name][0]

    def add_output_buffer(self, varname, buff, dtype=None, buffer_len=None):
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
        """
        self.__add_io_buffer__(buff, varname, False, dtype, buffer_len)

    def get_output_buffer(self, varname, dtype=None, buffer_len=None):
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
        """
        name = re.search('(\w+)', varname).group(0)
        if name not in self.__output_buffers__:
            self.__add_io_buffer__(None, varname, False, dtype, buffer_len)
        return self.__output_buffers__[name][0]
    
    
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
        for ipar, dims in enumerate(dims_list):
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
                        raise ValueError("Failed to broadcast array dimensions for "+func.__name__+". Could not find consistent value for dimension "+dim)
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
            self.__print__(1, "Found multiple compatible type signatures for this function:", types, "Using signature " + types[0] + ".")
        types = [np.dtype(t) for t in types[0]]
        
        # Reshape variable arrays to add broadcast dimensions and allocate new arrays as needed
        for i, param, dims, dtype in zip(range(len(params)), params, dims_list, types):
            shape = outerdims + [dims_dict[d.strip()] for d in dims.split(',') if d]
            if isinstance(param, str):
                params[i] = self.__add_var__(param, dtype, shape) 
            elif isinstance(param, np.ndarray):
                arshape = list(param.shape)
                for idim in range(-1, -1-len(shape), -1):
                    if arshape[idim]!=shape[idim]:
                        arshape.insert(len(arshape)+idim+1, 1)
                params[i] = param.reshape(tuple(arshape))
            elif isinstance(param, unit):
                params[i] = dtype.type(convert(1,param,self.__clk__))
            else:
                params[i] = dtype.type(param)
                
        # Make strings of input variables/constants for later printing
        proc_strs = []
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                proc_strs.append(re.search('(\w+)', arg).group(0))
            else:
                proc_strs.append(str(params[i]))
        proc_strs = tuple(proc_strs)
        
        self.__print__(2, 'Added processor: ' + func.__name__ + str(proc_strs).replace("'", ""))
        
        # Add the function and bound parameters to the list of processors
        self.__proc_list__.append((func, tuple(params)))
        self.__proc_strs__.append(proc_strs)
        return None

    def execute(self):
        """Execute the dsp chain on the entire input/output buffers"""
        for begin in range(0, self.__buffer_len__, self.__block_width__):
            self.execute_block(begin)
        
    def execute_block(self, offset=0):
        """Execute the dsp chain on a sub-set of the input/output buffers
        starting at entry offset, with length equal to the internal block size.
        """
        end = min(offset+self.__block_width__, self.__buffer_len__)
        if self.__verbosity__<3:
            self.__execute_procs__(offset, end)
        else:
            self.__execute_procs_verbose__(offset, end)
            
        
    def get_variable(self, varname):
        """Get the numpy array holding the internal memory buffer used for a
        named variable. The varname has the format
          "varname(length, type)[range]"
        The optional (length, type) field is used to initialize a new variable
        if necessary. The optional [range] field fetches only a subrange
        of the array to the function."""
        parse = re.match("\A(\w+)(\(.*\))?(\[.*\])?$", varname)
        if not parse:
            raise KeyError(varname+' could not be parsed')
        name, construct, slice = parse.groups()
        val = None
        
        if name in self.__vars_dict__:
            val = self.__vars_dict__[name]
        # if we did not varoable, but have a constructor expression, construct
        # a zeros-array using the size and data type in the constructor expr
        elif construct:
            args = [s.strip() for s in construct[1:-1].split(',')]
            if len(args)==1: # allocate scalar block
                try: val = np.zeros((self.__block_width__,), np.dtype(args[0]), 'F')
                except TypeError:
                    raise TypeError('Could not parse dtype from '+construct)
            elif len(args)==2: # allocate vector block
                try:
                    dtype = np.dtype(args[0])
                    del args[0]
                except TypeError:
                    try:
                        dtype = np.dtype(args[1])
                        del args[1]
                    except TypeError:
                        raise TypeError('Could not parse dtype and size from '+construct)
                try:
                    val = np.zeros((self.__block_width__, int(args[0])), dtype, 'F')
                except ValueError:
                    raise TypeError('Could not parse dtype and size from '+construct)
            self.__vars_dict__[name] = val
        # if variable was not defined and no constructor expression was found
        else: return None
        
        # if we found a bracketed range, return a slice of that range
        if slice:
            try:
                args = [int(i) if i else None for i in slice[1:-1].split(':')]
            except:
                raise TypeError('Could not slice array based on '+slice)
            if(len(args)==0): return val
            elif(len(args)==1): return val[..., args[0]]
            elif(len(args)==2): return val[..., args[0]:args[1]]
            elif(len(args)==3): return val[..., args[0]:args[1]:args[2]]
            else: raise TypeError('Could not slice array based on '+slice)
        else:
            return val
    
    # Add an array of zeros to the vars dict called name and return it
    def __add_var__(self, name, dtype, shape):
        if not re.match("\A\w+$", name):
            raise KeyError(name+' is not a valid alphanumeric name')
        if name in self.__vars_dict__:
            raise KeyError(name+' is already in variable list')
        arr = np.zeros(shape, dtype)
        self.__vars_dict__[name] = arr
        self.__print__(2, 'Added variable ' + re.search('(\w+)', name).group(0) + ' with shape ' + str(tuple(shape)) + ' and type ' + str(dtype))
        return arr

    
    # copy from input buffers to variables
    # call all the processors on their paired arg tuples
    # copy from variables to list of output buffers
    def __execute_procs__(self, start, end):
        for buf, var in self.__input_buffers__.values():
            np.copyto(var[0:end-start, ...], buf[start:end, ...], 'unsafe')
        for func, args in self.__proc_list__:
            func(*args)
        for buf, var in self.__output_buffers__.values():
            np.copyto(buf[start:end, ...], var[0:end-start, ...], 'unsafe')

    # verbose version of __execute_procs__. This is probably overkill, but it
    # was done to minimize python calls in the non-verbose version
    def __execute_procs_verbose__(self, start, end):
        names = set(self.__vars_dict__.keys())
        self.__print__(3, 'Input:')
        for name, (buf, var) in self.__input_buffers__.items():
            np.copyto(var[0:end-start, ...], buf[start:end, ...], 'unsafe')
            self.__print__(3, name+' = '+str(var))
            names.discard(name)
            
        self.__print__(3, 'Processing:')
        for (func, args), strs in zip(self.__proc_list__, self.__proc_strs__):
            func(*args)
            self.__print__(3, func.__name__ + str(strs).replace("'", ""))
            for name, arg in zip(strs, args):
                try:
                    names.remove(name)
                    self.__print__(3, name+' = '+str(arg))
                except: pass
                
        self.__print__(3, 'Output:')
        for name, (buf, var) in self.__output_buffers__.items():
            np.copyto(buf[start:end, ...], var[0:end-start, ...], 'unsafe')
            self.__print__(3, name+' = '+str(var))

    # append a tuple with the buffer and variable to either the input buffer
    # list (if input=true) or output buffer list (if input=false), making sure
    # that buffer shapes are compatible
    def __add_io_buffer__(self, buff, varname, input, dtype, buffer_len):
        var = self.get_variable(varname)
        if buff is not None and not isinstance(buff, np.ndarray):
            raise ValueError("Buffers must be ndarrays.")
        
        # if buffer length is not defined, figure out what it should be
        if buffer_len is not None:
            if self.__buffer_len__ is None:
                self.__buffer_len__ = buffer_len
            elif self.__buffer_len__ != buffer_len:
                raise ValueError("Buffer length was already set to a number different than the one provided. To change the buffer length, you must reset the buffers.")
        if not self.__buffer_len__:
            if buff is not None: self.__buffer_len__ = buff.shape[0]
            else: self.__buffer_len__ = self.__block_width__
            self.__print__(1, "Setting i/o buffer length to " + str(self.__buffer_len__))

        # if no buffer was provided, make one
        returnbuffer=False
        if buff is None:
            if var is None:
                raise ValueError("Cannot make buffer for variable that does not exist!")
            if not dtype: dtype=var.dtype
            buff = np.zeros((self.__buffer_len__,)+var.shape[1:], dtype)
            returnbuffer=True
            
        # Check that the buffer length is correct. For 1D buffer, reshape
        # it if possible, assuming array of structure ordering
        if not buff.shape[0] == self.__buffer_len__:
            if buff.ndim==1 and len(buff)%self.__buffer_len__==0:
                buff = buff.reshape(self.__buffer_len__, len(buff)//self.__buffer_len__)
            else:
                raise ValueError("Buffer provided for " + varname + " is the wrong length.")
            
        # Check that shape of buffer is compatible with shape of variable.
        # If variable does not yet exist, add it here
        if var is None:
            if dtype is None: dtype = buff.dtype
            var = self.__add_var__(varname, dtype, (self.__block_width__,)+buff.shape[1:])
        elif var.shape[1:] != buff.shape[1:]:
            raise ValueError("Provided buffer has shape " + str(buff.shape) + " which is not compatible with " + str(varname) + " shape " + str(var.shape))
        
        varname = re.search('(\w+)', varname).group(0)
        if input:
            self.__input_buffers__[varname]=(buff, var)
            self.__print__(2, 'Binding input buffer of shape ' + str(buff.shape) + ' and type ' + str(buff.dtype) + ' to variable ' + varname + ' with shape ' + str(var.shape) + ' and type ' + str(var.dtype))
        else:
            self.__output_buffers__[varname]=(buff, var)
            self.__print__(2, 'Binding output buffer of shape ' + str(buff.shape) + ' and type ' + str(buff.dtype) + ' to variable ' + varname + ' with shape ' + str(var.shape) + ' and type ' + str(var.dtype))

        if returnbuffer: return buff

    def __print__(self, verbosity, *args):
        if self.__verbosity__ >= verbosity:
            print(*args)

    def __str__(self):
        ret = 'Input variables: ' + str([name for name in self.__input_buffers__.keys()])
        for proc, strs in zip(self.__proc_list__, self.__proc_strs__):
            ret += '\n' + proc[0].__name__ + str(strs)
        ret += '\nOutput variables: ' + str([name for name in self.__output_buffers__.keys()])
        return ret.replace("'", "")
