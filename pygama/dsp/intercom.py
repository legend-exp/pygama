import numpy as np
import re
import itertools as it
from scimath.units import convert
from scimath.units.unit import unit

class Intercom:
    """
    TODO: Description of intercom 
          Verbose output options for debugging
    """
    def __init__(self, block_width=8, buffer_len=None, clock_unit=None):
        # Dictionary with numpy arrays containing input and output variables
        self.__vars_dict__ = {}
        # Ordered list of processors and a tuple containing the bound parameters
        # as either constants or variables from vars_dict
        self.__proc_list__ = []
        self.__block_width__ = block_width
        self.__buffer_len__ = buffer_len
        self.__input_buffers__ = []
        self.__output_buffers__ = []
        self.__clk__ = clock_unit

    def add_waveform(self, name, dtype, length):
        self.__add_variable__(name, dtype, (self.__block_width__, length))

    def add_scalar(self, name, dtype):
        self.__add_variable__(name, dtype, (self.__block_width__))

    def add_input_buffer(self, varname, buff, buffer_len=None):
        """
        Link an input buffer to a variable.
        """
        if(buffer_len is not None):
            if self.__buffer_len__ is None:
                self.__buffer_len__ = buffer_len
            elif self.__buffer_len__ != buffer_len:
                raise ValueError("Buffer length was already set to a number different than the one provided. To change the buffer length, you must reset the buffers.")
        
        if not isinstance(buff, np.ndarray) and buffer_library:
            buff = buffer_library[buff]
        self.__add_io_buffer__(buff, varname, True)

    def add_output_buffer(self, varname, buff, buffer_len=None):
        """
        Link an output buffer to a variable.
        """
        if(buffer_len is not None):
            if self.__buffer_len__ is None:
                self.__buffer_len__ = buffer_len
            elif self.__buffer_len__ != buffer_len:
                raise ValueError("Buffer length was already set to a number different than the one provided. To change the buffer length, you must reset the buffers.")
        
        if not isinstance(buff, np.ndarray) and buffer_library:
            buff = buffer_library[buff]
        self.__add_io_buffer__(buff, varname, False)


    
    def add_processor(self, func, *args, **kwargs):
        """
        Add a new processor and bind it to a set of parameters.
        - func should be a function taking numpy array args and no return
        - args should include the variables, by string, and values used to
          call func
        - keyword arguments include:
          - signature: broadcasting signature for a ufunc. By default, use
            func.signature
          - types: a list of strings defining the types of arrays needed for
            the func. By default, use func.types
        If parameters are given by value, they will be treated as constant
        and applied to all waveforms. If given by name, the parameter will be
        treated as a variable, and a block of memory will be allocated as needed
        and bound to the function.
        TODO: Add function lookup by string, for use with JSON files
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
                param_val = self.__get_var__(param)
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
            print("Found multiple compatible type signatures for this function:", types, "Using signature " + types[0] + ".")
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
                
        # Add the function and bound parameters to the list of processors
        self.__proc_list__.append((func, tuple(params)))
        return None

    def execute(self):
        for begin in range(0, self.__buffer_len__, self.__block_width__):
            self.execute_block(begin)
        
    def execute_block(self, offset=0):
        end = min(offset+self.__block_width__, self.__buffer_len__)
        self.__read_input_data__(offset, end)
        self.__execute_procs__()
        self.__write_output_data__(offset, end)
    
    
    # Add an array of zeros to the vars dict called name and return it
    def __add_var__(self, name, dtype, shape):
        if not re.match("\A\w+$", name):
            raise KeyError(name+' is not a valid alphanumeric name')
        if name in self.__vars_dict__:
            raise KeyError(name+' is already in variable list')
        arr = np.zeros(shape, dtype)
        self.__vars_dict__[name] = arr
        return arr

    
    # Parse variable name and fetch and return the array. Expected format is:
    #    varname(constructor vals)[slice vals]
    # where constructor vals and subrange vals are optional. Constructor vals
    # include the waveform length and type; if provided, and no variable is
    # found, create a new array. Slice vals are colon-separated values used
    # to define a slice
    def __get_var__(self, var):
        parse = re.match("\A(\w+)(\(.*\))?(\[.*\])?$", var)
        if not parse:
            raise KeyError(var+' could not be parsed')
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
    
    # call all the processors on their paired arg tuples
    def __execute_procs__(self):
        for func, args in self.__proc_list__:
            func(*args)
    
    # copy from list of input buffers to variables
    def __read_input_data__(self, start, end):
        for buffer, var in self.__input_buffers__:
            np.copyto(var[0:end-start, ...], buffer[start:end, ...], 'unsafe')
    
    # copy from variables to list of output buffers
    def __write_output_data__(self, start, end):
        for buffer, var in self.__output_buffers__:
            np.copyto(buffer[start:end, ...], var[0:end-start, ...], 'unsafe')

    # append a tuple with the buffer and variable to either the input buffer
    # list (if input=true) or output buffer list (if input=false), making sure
    # that buffer shapes are compatible
    def __add_io_buffer__(self, buff, varname, input):
        var = self.__get_var__(varname)
        if not isinstance(buff, np.ndarray):
            raise ValueError("Buffers must be ndarrays or valid indices for a provided library of ndarrays.")

        # if buffer length is not defined, figure out what it should be
        if not self.__buffer_len__:
            self.__buffer_len__ = buff.shape[0]
            print("Setting buffer length to " + self.__buffer_len__)
            
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
            var = self.__add_var__(varname, buff.dtype, (self.__block_width__,)+buff.shape[1:])
        elif var.shape[1:] != buff.shape[1:]:
            raise ValueError("Provided buffer has shape " + str(buff.shape) + " which is not compatible with " + str(varname) + " shape " + str(var.shape))

        if input: self.__input_buffers__.append((buff, var))
        else: self.__output_buffers__.append((buff, var))
        
