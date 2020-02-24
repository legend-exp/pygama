"""
io_base.py

base classes for reading/writing data
contains methods to save pandas dataframes to files.
subclasses:
     - digitizers.py -- Gretina4M, SIS3302, FlashCam, etc.
     - pollers.py -- MJDPreampDecoder, ISegHVDecoder, etc.
     
DataDecoders require you declare these before calling `super().__init__()`:
    * `self.digitizer_type`: a string naming the digitizer
    * `self.decoded_values`: a dictionary of variables and their attributes to
       convert to HDF5
"""

import numpy as np
import pandas as pd
from abc import ABC
import h5py

class DataDecoder(ABC):
    """Decodes DAQ stream data packets.

    The values that get decoded need to be described by a dict called
    'decoded_values' that helps determine how to set up the buffers and write
    them to file. See ORCAStruck3302 for an example.

    Subclasses should define a method for decoding data to a buffer like
    decode_packet(packet, data_buffer, packet_id, verbose=False)

    Garbage collection writes binary data as an array of uint32s to a
    variable-length array in the output file. If a problematic packet is found,
    call put_in_garbage(packet). User should set up an enum or bitbank of garbage
    codes to be stored along with the garbage packets.
    """
    def __init__(self, garbage_size=65536):
        self.garbage_buffer = np.empty(garbage_size, dtype='uint32')
        self.garbage_lensum = []
        self.garbage_ids = []
        self.garbage_codes = []

    @abstractmethod
    def decode_packet(packet, data_buffer, packet_id, verbose=False):
        ...


    def initialize_df_buffer(self, df_buffer):
        if not hasattr(self, 'decoded_values'):
            name = type(self).__name__
            print(name, 'Error: no decoded_values available for setting up buffer')
            return
        for name, attrs in self.decoded_values.items():
            df_buffer.add_field(name, attrs)


    def put_in_garbage(self, packet, packet_id, code):
        p32 = np.frombuffer(packet, dtype=np.uint32)
        start_loc = 0 if len(self.garbage_lensum) == 0 else self.garbage_lensum[-1]
        size = len(self.garbage_buffer)
        while start_loc + len(p32) > size:
            self.garbage_buffer.resize(2*size)
            size = 2*size
        self.garbage_buffer[start_loc:start_loc+len(p32)] = p32
        self.garbage_lensum.append(len(p32))
        self.garbage_ids.append(packet_id)
        self.garbage_codes.append(code)


    def get_garbage_usage(self):
        n_packets = len(self.garbage_lensum)
        used = 0 if n_packets == 0 else self.garbage_lensum[-1]
        used *= self.garbage_buffer.itemsize
        size = self.garbage_buffer.nbytes
        return n_packets, used, size


    def write_out_garbage(self, filename, group, code_attrs, lh5_store=None):
        if lh5_store is None: lh5_store = LH5Store()
        size = 0 if len(self.garbage_lensum) == 0 else self.garbage_lensum[-1]
        if size == 0: return

        # set group name according to decoder class name
        if not group.endswith('/'): group += '/'
        group += type(self).__name__

        # write the packets
        pgrp_attrs = '{ datatype: variable_length_array }'
        lh5_store.append_ndarray(filename, 'data', 
                                 self.garbage_buffer[:size]
                                 group = group+'/packets', 
                                 grp_attrs = pgrp_attrs)

        # write the packet lengths
        lh5_store.append_ndarray(filename, 'lensum', 
                                 np.ndarray(self.garbage_lensum, dtype='uint32')
                                 group=group+'/packets')

        # write the packet ids
        lh5_store.append_ndarray(filename, 'packet_ids', 
                                 np.ndarray(self.garbage_ids, dtype='uint32')
                                 group=group)

        # write the packet codes
        lh5_store.append_ndarray(filename, 'codes', 
                                 np.ndarray(self.garbage_ids, dtype='uint32')
                                 data_attrs = code_attrs, 
                                 group = group)

        # clear the garbage fields so that they can be reused if desired
        self.garbage_lensum.clear()
        self.garbage_codes.clear()
        self.garbage_ids.clear()



def get_lh5_datatypename(obj):
    """Get the LH5 datatype name of an LH5 object"""
    if isinstance(obj, LH5Table): return 'table'
    if isinstance(obj, LH5Struct): return 'struct'
    if np.isscalar(obj): return get_lh5_element_type(obj)
    if isinstance(obj, LH5FixedSizeArray): return 'fixedsize_array'
    if isinstance(obj, LH5ArrayOfEqualSizedArrays): return 'array_of_equalsized_arrays'
    # only options left are LH5Array and LH5VectorOfVectors
    if isinstance(obj, LH5Array): return 'array'


def get_lh5_element_type(obj):
    """Get the LH5 element type of a scalar or array"""
    if isinstance(obj, str): return 'string'
    if hasattr(obj, dtype):
        kind = obj.dtype.kind
        if kind == '?': return 'bool'
        if kind in ['b', 'B', 'V']: return 'blob'
        if kind in ['i', 'u', 'f']: return 'real'
        if kind == 'c': return 'complex'
        if kind in ['S', 'a', 'U']: return 'string'
    print('Cannot determine LH5 element_type for object of type', type(obj).__name__)
    return None



class LH5Struct(dict):
    """A dictionary with an optional set of attributes.

    Don't allow to instantiate with a dictionary -- have to add fields
    one-by-one using add_field() to keep datatype updated
    """
    # TODO: overload setattr to require add_field for setting
    def __init__(self, attrs={}):
        self.attrs = attrs
        self.attrs['datatype'] = get_lh5_datatypename(self)+'{}'


    def add_field(self, name, obj, attrs={}):
        self[name] = obj
        self.update_datatype()


    def update_datatype(self):
        datatype = get_lh5_datatypename(self)
        datatype += '{' + ','.join(self.keys()) + '}'
        self.attrs['datatype'] = datatype



class LH5Table(LH5Struct):
    """A special struct of array or subtable 'columns' of equal length."""
    # TODO: overload getattr to allow access to fields as object attributes?
    def __init__(self, size=1024, attrs={})
        super().__init__(attrs=attrs)
        self.size = size
        self.loc = 0


    def push_row(self):
        self.loc += 1


    def is_full(self):
        return self.loc >= self.size


    def clear(self):
        self.loc = 0


    def add_field(self, name, obj, attrs={}):
        if not isinstance(obj, LH5Table) and not isinstance(obj, LH5Array):
            print('LH5Table: Error: cannot add field of type', type(obj).__name__)
            return
        super().add_field(name, obj, attrs)


class LH5Scalar:
    """Holds a value and some attributes (like units)
    """
    def __init__(self, value, attrs={}):
        self.value = value
        self.attrs = attrs
        self.attrs['datatype'] = get_lh5_element_type(self.value)


class LH5Array:
    """Holds an ndarray and attributes
    """
    def __init__(self, nda, attrs={}):
        self.nda = nda
        self.dtype = nda.dtype
        self.attrs = attrs
        dt = get_lh5_datatypename(self)
        nD = str(len(self.nda.shape))
        et = get_lh5_element_type(self)
        self.attrs['datatype'] = dt + '<' + nD + '>{' + et + '}'


class LH5FixedSizeArray(LH5Array):
    """An array of fixed-size arrays

    Arrays with guaranteed shape along axes > 0: for example, an array of
    vectors will always length 3 on axis 1, and it will never change from
    application to application.  This data type is used for optimized memory
    handling on some platforms. We are not that sophisticated so we are just
    storing this identification for .lh5 validity, i.e. for now this class is
    just an alias.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

class LH5ArrayOfEqualSizedArrays(LH5Array):
    """An array of equal-sized arrays

    Arrays of equal size within a file but could be different from application
    to application. Canonical example: array of same-length waveforms.

    If shape is not "1D array of arrays of shape given by axes 1-N" (of nda)
    then specify the dimensionality split in the constructor.
    """
    def __init__(self, *args, dims=None, **kwargs):
        super().__init__(*args, **kwargs)
        dt = get_lh5_datatypename(self)
        nD = str(len(self.nda.shape))
        if dims is not None: nD = '.'.join([str(i) for i in dims])
        et = get_lh5_element_type(self)
        self.attrs['datatype'] = dt + '<' + nD + '>{' + et + '}'


class LH5VectorOfVectors:
    """A variable-length array of variable-length arrays

    For now only a 1D vector of 1D vectors is supported. Internal representation
    is as two ndarrays, one to store the flattened data contiguosly and one to
    store the cumulative sum of lengths of each vector. 
    """ 
    def __init__(self, *args, data_nda, lensum_nda, attrs={}, **kwargs):
        self.data_nda = data_nda
        self.lensum_nda = lensum_nda
        self.dtype = data_nda.dtype
        self.attrs = attrs
        et = get_lh5_element_type(self)
        self.attrs['datatype'] = 'array<1>{array<1>{' + et + '}}'


class LH5Store:
    def __init__(self, base_path='', keep_open=False):
        self.base_path = base_path
        self.keep_open = keep_open
        self.files = {}

    def gimme_file(self, lh5_file, mode):
        if isinstance(lh5_file, h5py.File): return lh5_file
        if lh5_file in self.files.keys(): return self.files[lh5_file]
        full_path = self.base_path + '/' + lh5_file
        h5f = h5py.File(full_path, mode)
        if self.keep_open: self.files[lh5_file] = h5f
        return h5f

    def gimme_group(self, group, base_group, grp_attrs=None):
        if isinstance(group, h5py.Group): return group
        if group in base_group: return base_group[group]
        group = base_group.create_group(group)
        if grp_attrs is not None: group.attrs.update(grp_attrs)
        return group


    def append_ndarray(self, lh5_file, ds, nda_data, data_attrs=None, group='/', grp_attrs=None):
        # Grab the file, group, and ds, creating as necessary along the way
        lh5_file = self.gimme_file(lh5_file, 'a')
        group = self.gimme_group(group, lh5_file, grp_attrs)

        # need to create dataset from nda_data the first time for speed
        # creating an empty dataset and appending to that is super slow!
        if not isinstance(ds, h5py.Dataset):
            if ds not in group:
                ds = group.create_dataset(ds, data=nda_data, maxshape=(None,))
                if data_attrs is not None: ds.attrs.update(data_attrs)
            else: ds = group[ds]

        # Now append
        old_len = ds.shape[0]
        add_len = nda_data.shape[0]
        ds.resize(old_len + add_len, axis=0)
        ds[-add_len:] = nda_data 


    @staticmethod
    def parse_datatype(datatype):
        """Parse datatype string and return type, shape, elements"""
        if '{' not in datatype: return 'scalar', (), datatype

        # for other datatypes, need to parse the datatype string
        import parse
        datatype, element_description = parse('{}{{{}}}', datatype)
        if datatype.endswith('>'): 
            datatype, dims = parse('{}<{}>')
            dims = [int(i) for i in dims.split(',')]
            return datatype, tuple(dims), element_description
        else return datatype, None, element_description.split(',')


    def read_object(self, lh5_file, path, start_row=0, n_rows=None, obj_buf=None):
        """Return an object and attributes for data at path in lh5_file

        Set n_rows to read out a subset of the first data axis (when possible)
        """
        h5f = self.gimme_file(lh5_file, 'r')
        if path not in h5f:
            print('LH5Store:', path, "not in", lh5_file)
            return None

        # get the datatype
        if 'datatype' not in h5f[path].attrs:
            print('LH5Store:', path, 'in file', lh5_file, 'is missing the datatype attribute')
            return None
        datatype = h5f[path].attrs
        datatype, shape, elements = self.parse_datatype(datatype)

        # scalars are dim-0 datasets
        if datatype == 'scalar': return h5f[path][()]

        # check if we have a vector of vectors of different size...
        is_vecvec = elements.startswith('array<1>')

        # read out "normal" arrays by slicing
        if datatype == 'array' and not is_vecvec:
            if n_rows == None: return h5f[path][()]
            else: return h5f[path][start_row:start_row+n_rows]

        # skip fixed-sized arrays
        if datatype == 'fixedsize_array': 
            print('LH5Store: fixed-sized array reading not implemented')
            return None

        # arrays of arrays get reshaped and sliced
        if datatype == 'array_of_equalsized_arrays':
            dimprod = np.prod(dims)
            nn = h5f[path].size / dimprod
            if n_rows == None or n_rows >= nn: 
               return h5f[path][()].reshape((nn,)+dims)
            stop_row = n_rows-start_row
            # put the slice before the reshape to avoid reading whole ds at once
            return h5f[path][start_row*dimprod:stop_row*dimprod].reshape(dims)

        # skip vector-of-vectors
        if is_vecvec:
            print('LH5Store: vector-of-vectors reading not implemented')
            return None

        # recursively build a struct, return as a dictionary
        if datatype == 'struct':
            struct = {}
            for field in element_description.split(','):
                struct[field] = self.read_object(lh5_file, path+'/'+field, start_row, n_rows)
            return struct

        # read a table into a dataframe
        if dataframe == 'table':
            col_dict = {}
            for field in element_description.split(','):
                col_dict[field] = self.read_object(lh5_file, path+'/'+field, start_row, n_rows)
            return pd.DataFrame(data=col_dict)

        print('LH5Store: don\'t know how to read datatype', datatype)
        return None


    def write_object(self, obj, name, lh5_file, group, scalar_attrs={}, append=True):
        """Write an object into an lh5_file

        obj should be a scalar or a LH5* object. If a scalar, you can optionally
        send in attributes (like units) via scalar_attrs

        Set append to true for non-scalar objects if you want to append along
        axis 0 (the first dimension) (or axis 0 of non-scalar subfields of
        structs)

        If obj is array-like and you only want to write out the first n rows, send
        it in as obj[:n].

        Note on scalar values: must be either a str or a fixed-size numpy type
            (see list at https://numpy.org/devdocs/user/basics.types.html )
            - If it has units, specify it in scalar_attrs['units']. 
            - Not appendable
            - Example:
              length = np.float64(1.)
              length_attrs = { 'units': 'm' }

        Arrays: Built off of ndarrays, which map well to HDF5 datasets.
            - appendable along axis 0
            - If it has units, specify it in obj_attrs['units']
            - type of array is determined by obj_attrs['datatype']:
              - matches 'array', 'array<N>{kind}', or no datatype: 
                just a standard array. Infer shape and dtype from ndarray
                attributes
              - matches 'fixedsize_array' or 'fixedsize_array<N>{kind}': 
                a fixed-sized array that guarantees some length e.g. along axis
                1, used for optimized memory handling on some platforms. Python
                is not that sophisticated so we are just storing this
                identification in the attributes. Shape and kind is inferred
                from ndarray attributes
              - matches 'array_of_equalsized_arrays' or
                'array_of_equalsized_arrays<N,M,...>{kind}':
                arrays of arrays of the same size, stored as an ndarray of
                dimension 1 + N + M + ... where we used ndarray's reshape to get
                the right "view" so that array[i] is the i'th N-dim array of
                M-dim arrays of ...
              - matches 'array{array}' or 'array<1>{array<1>{kind}}':
                vector of vectors. obj must be a tuple of ndarrays, where the
                first 

            
        """
        lh5_file = self.gimme_file(lh5_file, mode = 'a' if append else 'w')
        group = self.gimme_group(group, lh5_file)

        # struct or table
        if isinstance(obj, LH5Struct)
            if 'datatype' not in obj.attrs:
                print('LH5Store: misformed struct', name)
                return
            group = gimme_group(name, group, grp_attrs=obj.attrs)
            fields = obj.keys()
            obj_attrs['datatype'] = obj.gimme_datatype(datatype)
            for fieldname in obj.keys():
                self.write_object(obj[field], 
                                  fieldname, 
                                  lh5_file, 
                                  group, 
                                  obj_attrs=obj.field_attrs[field], 
                                  append=append)
            return

        # scalars
        elif np.isscalar(obj):
            ds = group.create_dataset(name, shape=(), data=obj)
            ds.attrs.update(obj_attrs)
            ds.attrs['datatype'] = get_lh5_element_type(obj)
            return

 
        # if we get this far, must be one of the LH5Array types
        else: 
            else: 
                if obj_type != 'ndarray': 
                    print('LH5Store: don\'t know how to write out a', obj_type, 'named', name)
                    return 

                # flat arrays: "bare" ndarrays
                if 'datatype' not in obj_attrs:
                    ndim = len(obj.shape)
                    obj_attrs['datatype'] = 'array<' + str(ndim) + '>{' + kind + '}'

                # fixed-sized arrays
                else if obj_attrs['datatype'] == 'fixedsize_array':
                    ndim = len(obj.shape)
                    obj_attrs['datatype'] += '<' + str(ndim) + '>{' + kind + '}'

                # arrays of arrays of same size
                else if obj_attrs['datatype'].startswith('array_of_equalsized_arrays<'):
                    if not obj_attrs['datatype'].endswith('>'):
                        print('LH5Store: improper obj_attrs', obj_attrs['datatype'], 'for', name)
                        return
                    obj_attrs['datatype'] += '{' + kind + '}'


                # need to create dataset from ndarray the first time for speed
                # creating an empty dataset and appending to that is super slow!
                if not append or name not in group:
                    ds = group.create_dataset(name, data=obj, maxshape=(None,))
                    ds.attrs.update(obj_attrs)
                    return
                
                # Now append
                ds = group[name]
                old_len = ds.shape[0]
                add_len = obj.shape[0]
                ds.resize(old_len + add_len, axis=0)
                ds[-add_len:] = obj 



class TableBuffer(pd.DataFrame)
    """Buffer object for pygama tables

    A collection of arrays of shape (N,...) where N is a fixed lenth
    Used in daq-to-raw data conversion, DSP, and parameter calibration as a
    buffer for IO with tables of data. LH5Store streams to/from this object
    """
    def __init__(self, size=1024)
        self.size = size
        self.loc = 0
        self.fields = {}
        self.fld_attrs = {}


    def push_row(self):
        self.loc += 1


    def is_full(self):
        return self.loc >= self.size


    def clear(self):
        self.loc = 0


    def add_field(self, field_name, field_data, field_attrs):

    def add_field(self, field_name, field_attrs):
        if field_name in self.fields:
            print('buffer already contains field ' + name)
            return

        # pop the dtype attribute since it's now a property of the field
        dtype = 'double' if 'dtype' not in field_attrs else field_attrs.pop('dtype')

        # handle simple scalar first 
        if 'datatype' not in field_attrs: 
            self.fields[field_name] = np.empty(self.size, dtype=dtype)

        # handle waveform data
        elif field_attrs['datatype'] == 'waveform':
            # need to be able to compute the shape for the buffer
            if 'length' not in field_attrs:
                print('TableBuffer error: time_series must have length attribute.')
                return

            # compute the shape
            if field_attrs['length'] == 'var':
                if 'length_estimate' not in field_attrs:
                    print('TableBuffer error: var-length time_series must have length_estimate attribute.')
                    return
                shape = (self.size,) + field_attrs['length_estimate']
            else: shape = (self.size,) + field_attrs['length']

            # allocate buffer for timeseries. Do it this way even for variable
            # length because DataFrames always have to have the same number of rows
            # in each column
            self[field_name] = [table for table in np.empty(shape, dtype=dtype)]

            # now add auxiliary fields if necessary
            if field_attrs['length'] == 'var':
                self[field_name+'_lensum'] = np.empty(self.size, dtype='uint32')
            if 'sampling_period' in field_attrs and field_attrs['sampling_period' == 'var':
                dtype = 'double'
                if 'sampling_period_dtype' in field_attrs: 
                    # pop the dtype because its redundant after this
                    dtype = field_attrs.pop('sampling_period_dtype')
                self[field_name+'_dt'] = np.empty(self.size, dtype=dtype)

        # check another 'dataset' value
        else
            print('TableBuffer error: unknown datatype', field_attrs['datatype'])

        # finally, append the attributews
        self.attrs['field_attrs'][field_name] = field_attrs
    

    def get_raw_buffer(self, col, n_rows):
        import ctypes as C
        from ctypes.util import find_library
        libc = C.CDLL(find_library('c'))
        libc.malloc.restype = C.c_void_p
        data_pointer = df[col][0].__array_interface__['data'][0]
        ctype = np.ctypeslib.as_ctypes_type(df[col][0].dtype)
        data_pointer = C.cast(data_pointer,C.POINTER(ctype))
        size = n_rows * np.prod(df[col][0].shape)
        return np.ctypeslib.as_array(data_pointer,shape=(size,))


    def append_to_lh5(self, lh5_file, group = '/', lh5_store = None, n_rows_to_write = None):
        # Exit if no data
        if len(self.keys()) == 0 or n_rows_to_write == 0: return
        if n_rows_to_write == None: n_rows_to_write = self.size

        # Manage lh5 data with LH5Store
        if lh5_store is None: lh5_store = LH5Store() 

        # separate the group and the field attributes needed for writing
        grp_attrs = self.attrs.copy()
        all_field_attrs = grp_attrs.pop('field_attrs')

        for field, field_attrs in all_field_attrs.items():
            # handle simple scalar first 
            if 'datatype' not in field_attrs: 
                nda_data = self[field].values[:n_rows_to_write]
                lh5_store.append_ndarray(filename, field, 
                                         nda_data, data_attrs=field_attrs, 
                                         group=group, grp_attrs=grp_attrs)

            # handle time_series data
            elif field_attrs['datatype'] == 'time_series':
                # will write to its own group
                if not group.endswith('/'): group += '/'
                group += field
                grp_attrs = field_attrs
                # write the raw data
                nda_data = get_raw_buffer(self, field, n_rows_to_write)
                lh5_store.append_ndarray(filename, 'data', nda_data, group, grp_attrs)
                # write out auxiliary fields as necessary
                if field+'_lensum' in self:
                    nda_data = self[field+'_lensum'].values[:n_rows_to_write]
                    lh5_store.append_ndarray(filename, 'lensum', nda_data, group=group)
                if field+'_dt' in self:
                    nda_data = self[field+'_dt'].values[:n_rows_to_write]
                    lh5_store.append_ndarray(filename, 'dt', nda_data, group=group)



def get_ccc(crate, card, channel):
    return (crate << 9) + ((card & 0xf) << 4) + (channel & 0xf)


def get_crate(ccc):
    return ccc >> 9


def get_card(ccc):
    return (ccc >> 4) & 0x1f


def get channel(ccc):
    return ccc * 0xf

