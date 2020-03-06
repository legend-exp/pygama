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
    def __init__(self, garbage_length=256, packet_size_guess=1024):
        self.garbage_table = LH5Table(garbage_length) 
        shape_guess=(garbage_length, packet_size_guess)
        self.garbage_table.add_field('packets', 
                                     LH5VectorOfVectors(shape_guess=shape_guess, dtype='uint8'))
        self.garbage_table.add_field('packet_id', 
                                     LH5Array(shape=garbage_length, dtype='uint32'))
        # TODO: add garbage codes enum attribute: user supplies in constructor
        # before calling super()
        self.garbage_table.add_field('garbage_code', 
                                     LH5Array(shape=garbage_length, dtype='uint32'))


    def initialize_lh5_table(self, lh5_table):
        if not hasattr(self, 'decoded_values'):
            name = type(self).__name__
            print(name, 'Error: no decoded_values available for setting up buffer')
            return
        size = lh5_table.size
        for field, attrs in self.decoded_values.items():
            if 'dtype' not in attrs:
                name = type(self).__name__
                print(name, 'Error: must specify dtype for', field)
                continue

            dtype = attrs.pop('dtype')
            if 'datatype' not in attrs:
                # no datatype: just a "normal" array 
                # allow to override "kind" for the dtype for lh5
                if 'kind' in attrs: 
                    attrs['datatype'] = 'array<1>{' + attrs.pop('kind') + '}'
                lh5_table.add_field(field, LH5Array(shape=size, dtype=dtype, attrs=attrs))
                continue

            datatype = attrs.pop('datatype')

            # handle waveforms from digitizers in a uniform way
            if datatype == 'waveform':
                wf_table = LH5Table(size) 

                # Build t0 array. No attributes for now
                # TODO: add more control over t0: another field to fill it?
                # Optional units attribute?
                wf_table.add_field('t0', LH5Array(nda=np.zeros(size, dtype='float'))) 

                # Build sampling period array with units attribute
                wf_per = attrs.pop('sample_period')
                dt_nda = np.full(size, wf_per, dtype='float')
                wf_per_units = attrs.pop('sample_period_units')
                dt_attrs = { 'units': wf_per_units }
                wf_table.add_field('dt', LH5Array(nda=dt_nda, attrs = dt_attrs))

                # Build waveform array. All non-popped attributes get sent
                # TODO: add vector of vectors and compression capabilities
                wf_len = attrs.pop('length')
                dims = [1,1]
                aoesa = LH5ArrayOfEqualSizedArrays(shape=(size,wf_len), dtype=dtype, dims=dims, attrs=attrs)
                wf_table.add_field('values', aoesa)

                lh5_table.add_field(field, wf_table)
                continue

            # If we get here, must be a LH5 datatype
            datatype, shape, elements = parse_datatype(datatype)

            if datatype == 'array_of_equalsized_arrays':
                length = attrs.pop('length')
                dims = [1,1]
                aoesa = LH5ArrayOfEqualSizedArrays(shape=(size,length), dtype=dtype, dims=dims, attrs=attrs)
                lh5_table.add_field(field, aoesa)
                continue

            if elements.startswith('array'): # vector-of-vectors
                length_guess = size
                if 'length_guess' in attrs: length_guess = attrs.pop('length_guess')
                vov = LH5VectorOfVectors(shape_guess=(size,length_guess), dtype=dtype, attrs=attrs)
                lh5_table.add_field(field, vov)
                continue

            else:
                name = type(self).__name__
                print(name, 'Error: do not know how to make a', datatype, 'for', field)


    def put_in_garbage(self, packet, packet_id, code):
        i_row = self.garbage_table.loc
        p8 = np.frombuffer(packet, dtype='uint8')
        self.garbage_table['packets'].set_vector(i_row, p8)
        self.garbage_table['packet_id'].nda[i_row] = packet_id
        self.garbage_table['garbage_codes'].nda[i_row] = code
        self.garbage_table.push_row()


    def write_out_garbage(self, filename, group='/', lh5_store=None):
        if lh5_store is None: lh5_store = LH5Store()
        n_rows = self.garbage_table.loc 
        if n_rows == 0: return 
        lh5_store.write_object(self.garbage_table, 'garbage', filename, group, n_rows=n_rows, append=True)
        self.garbage_table.clear()



class DataTaker(DataDecoder):
    pass



def get_lh5_datatype_name(obj):
    """Get the LH5 datatype name of an LH5 object"""
    if isinstance(obj, LH5Table): return 'table'
    if isinstance(obj, LH5Struct): return 'struct'
    if np.isscalar(obj): return get_lh5_element_type(obj)
    if isinstance(obj, LH5FixedSizeArray): return 'fixedsize_array'
    if isinstance(obj, LH5ArrayOfEqualSizedArrays): return 'array_of_equalsized_arrays'
    if isinstance(obj, LH5Array): return 'array'
    if isinstance(obj, LH5VectorOfVectors): return 'array'
    print('Cannot determine LH5 datatype name for object of type', type(obj).__name__)
    return None


def get_lh5_element_type(obj):
    """Get the LH5 element type of a scalar or array"""
    if isinstance(obj, str): return 'string'
    if hasattr(obj, 'dtype'):
        kind = obj.dtype.kind
        if kind == '?' or obj.dtype.name == 'bool': return 'bool'
        #FIXME: pygama will call all uint8's "blobs" by this logic...
        if kind in ['b', 'B', 'V']: return 'blob'
        if kind in ['i', 'u', 'f']: return 'real'
        if kind == 'c': return 'complex'
        if kind in ['S', 'a', 'U']: return 'string'
    print('Cannot determine LH5 element_type for object of type', type(obj).__name__)
    return None


def parse_datatype(datatype):
    """Parse datatype string and return type, shape, elements"""
    if '{' not in datatype: return 'scalar', (), datatype

    # for other datatypes, need to parse the datatype string
    from parse import parse
    datatype, element_description = parse('{}{{{}}}', datatype)
    if datatype.endswith('>'): 
        datatype, dims = parse('{}<{}>', datatype)
        dims = [int(i) for i in dims.split(',')]
        return datatype, tuple(dims), element_description
    else: return datatype, None, element_description.split(',')



class LH5Struct(dict):
    """A dictionary with an optional set of attributes.

    Don't allow to instantiate with a dictionary -- have to add fields
    one-by-one using add_field() to keep datatype updated
    """
    # TODO: overload setattr to require add_field for setting?
    def __init__(self, obj_dict={}, attrs={}):
        self.update(obj_dict)
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print(type(self).__name__ + ': Warning: datatype does not match obj_dict!')
                print('datatype: ', self.attrs['datatype'])
                print('obj_dict.keys(): ', obj_dict.keys())
                print('form_datatype(): ', self.form_datatype())
        else: self.attrs['datatype'] = self.form_datatype()


    def add_field(self, name, obj):
        self[name] = obj
        self.attrs['datatype'] = self.form_datatype()


    def form_datatype(self):
        datatype = get_lh5_datatype_name(self)
        datatype += '{' + ','.join(self.keys()) + '}'
        return datatype



class LH5Table(LH5Struct):
    """A special struct of array or subtable 'columns' of equal length."""
    # TODO: overload getattr to allow access to fields as object attributes?
    def __init__(self, size=1024, col_dict={}, attrs={}):
        super().__init__(obj_dict=col_dict, attrs=attrs)
        self.size = int(size)
        self.loc = 0


    def push_row(self):
        self.loc += 1


    def is_full(self):
        return self.loc >= self.size


    def clear(self):
        self.loc = 0


    def add_field(self, name, obj):
        if not isinstance(obj, LH5Table) and not isinstance(obj, LH5Array) and not isinstance(obj, LH5VectorOfVectors):
            print('LH5Table: Error: cannot add field of type', type(obj).__name__)
            return
        # TODO: check length of obj and make sure it matches size; perhaps
        # provide length override to for obj to be resized to self.size
        super().add_field(name, obj)

    def get_dataframe(self, *cols, copy=False):
        """Get a dataframe containing each of the columns given. If no columns
        are given, get include all fields as columns."""
        df = pd.DataFrame(copy=copy)
        if len(cols)==0:
            for col, dat in self.items():
                df[col] = dat.nda
        else:
            for col in cols:
                df[col] = self[col].nda
        return df


class LH5Scalar:
    """Holds just a value and some attributes (datatype, units, ...)
    """
    def __init__(self, value, attrs={}):
        self.value = value
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != get_lh5_element_type(self.value):
                print('LH5Scalar: Warning: datatype does not match value!')
                print('datatype: ', self.attrs['datatype'])
                print('type(value): ', type(value).__name__)
        else: self.attrs['datatype'] = get_lh5_element_type(self.value)



class LH5Array:
    """Holds an ndarray and attributes
    """
    def __init__(self, nda=None, shape=None, dtype=None, attrs={}):
        self.nda = nda if nda is not None else np.empty(shape, dtype=dtype)
        self.dtype = self.nda.dtype
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print(type(self).__name__ + ': Warning: datatype does not match nda!')
                print('datatype: ', self.attrs['datatype'])
                print('form_datatype(): ', self.form_datatype())
                print('dtype:', self.dtype)
        else: self.attrs['datatype'] = self.form_datatype()


    def form_datatype(self):
        dt = get_lh5_datatype_name(self)
        nD = str(len(self.nda.shape))
        et = get_lh5_element_type(self)
        return dt + '<' + nD + '>{' + et + '}'



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
        self.dims = dims
        super().__init__(*args, **kwargs)


    def form_datatype(self):
        dt = get_lh5_datatype_name(self)
        nD = str(len(self.nda.shape))
        if self.dims is not None: nD = ','.join([str(i) for i in self.dims])
        et = get_lh5_element_type(self)
        return dt + '<' + nD + '>{' + et + '}'



class LH5VectorOfVectors:
    """A variable-length array of variable-length arrays

    For now only a 1D vector of 1D vectors is supported. Internal representation
    is as two ndarrays, one to store the flattened data contiguosly and one to
    store the cumulative sum of lengths of each vector. 
    """ 
    def __init__(self, data_array=None, lensum_array=None, shape_guess=None, dtype=None, attrs={}):
        if lensum_array is None:
            self.lensum_array = LH5Array(shape=(shape_guess[0],), dtype=dtype)
        else: self.lensum_array = lensum_array
        if data_array is None:
            length = np.prod(shape_guess)
            self.data_array = LH5Array(shape=(length,), dtype=dtype)
        else: self.data_array = data_array
        self.dtype = self.data_array.dtype
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print('LH5VectorOfVectors: Warning: datatype does not match dtype!')
                print('datatype: ', self.attrs['datatype'])
                print('form_datatype(): ', self.form_datatype())
        else: self.attrs['datatype'] = self.form_datatype()


    def form_datatype(self):
        et = get_lh5_element_type(self)
        return 'array<1>{array<1>{' + et + '}}'


    def set_vector(self, i_vec, nda):
        """Insert vector nda at location i_vec.

        self.data_array is doubled in length until nda can be appended to it.
        """
        if i_vec<0 or i_vec>len(self.lensum_array.nda)-1:
            print('LH5VectorOfVectors: Error: bad i_vec', i_vec)
            return 
        if len(nda.shape) != 1:
            print('LH5VectorOfVectors: Error: nda had bad shape', nda.shape)
            return
        start = 0 if i_vec == 0 else self.lensum_array.nda[i_vec-1]
        end = start + len(nda)
        while end >= len(self.data_array.nda):
            self.data_array.nda.resize(2*len(self.data_array.nda))
        self.data_array.nda[start:end] = nda
        self.lensum_array.nda[i_vec] = end




class LH5Store:
    def __init__(self, base_path='', keep_open=False):
        self.base_path = base_path
        self.keep_open = keep_open
        self.files = {}

    def gimme_file(self, lh5_file, mode):
        if isinstance(lh5_file, h5py.File): return lh5_file
        if lh5_file in self.files.keys(): return self.files[lh5_file]
        if self.base_path != '': full_path = self.base_path + '/' + lh5_file
        else: full_path = lh5_file
        h5f = h5py.File(full_path, mode)
        if self.keep_open: self.files[lh5_file] = h5f
        return h5f

    def gimme_group(self, group, base_group, grp_attrs=None):
        if isinstance(group, h5py.Group): return group
        if group in base_group: return base_group[group]
        group = base_group.create_group(group)
        if grp_attrs is not None: group.attrs.update(grp_attrs)
        return group


    def read_object(self, name, lh5_file, start_row=0, n_rows=None, obj_buf=None):
        """Return an object and attributes for data at path=name in lh5_file

        Set start_row, n_rows to read out a subset of the first data axis (when possible)
        """
        #TODO: implement obj_buf. Ian's idea: add an iterator so one can do
        #      something like
        #      for data in lh5iterator(file, chunksize, nentries, ...):
        #          proc.execute()

        h5f = self.gimme_file(lh5_file, 'r')
        if name not in h5f:
            print('LH5Store:', name, "not in", lh5_file)
            return None

        # get the datatype
        if 'datatype' not in h5f[name].attrs:
            print('LH5Store:', name, 'in file', lh5_file, 'is missing the datatype attribute')
            return None
        datatype = h5f[name].attrs['datatype']
        datatype, shape, elements = parse_datatype(datatype)

        # scalars are dim-0 datasets
        if datatype == 'scalar': 
            if elements == 'bool':
                return LH5Scalar(value=np.bool(h5f[name][()]), attrs=h5f[name].attrs)
            return LH5Scalar(value=h5f[name][()], attrs=h5f[name].attrs)

        # recursively build a struct, return as a dictionary
        if datatype == 'struct':
            obj_dict = {}
            for field in elements:
                obj_dict[field] = self.read_object(name+'/'+field, h5f, start_row, n_rows)
            return LH5Struct(obj_dict=obj_dict, attrs=h5f[name].attrs)

        # read a table into a dataframe
        if datatype == 'table':
            # TODO: set the size and loc parameters
            col_dict = {}
            for field in elements:
                col_dict[field] = self.read_object(name+'/'+field, 
                                                   h5f, 
                                                   start_row=start_row, 
                                                   n_rows=n_rows)
            return LH5Table(col_dict=col_dict, attrs=h5f[name].attrs)

        # read out vector of vectors of different size
        if elements.startswith('array'):
            if start_row == 0: 
                lensum_array = self.read_object(name+'/cumulative_length', h5f, n_rows=n_rows)
                da_start = 0
            else:
                lensum_array = self.read_object(name+'/cumulative_length', 
                                                h5f, 
                                                start_row=start_row-1, 
                                                n_rows=n_rows+1)
                da_start = lensum_array.nda[0]
                lensum_array.nda = lensum_array.nda[1:]
            da_nrows = lensum_array.nda[-1] - da_start
            data_array = self.read_object(name+'/flattened_data', 
                                          h5f, 
                                          start_row=da_start, 
                                          n_rows=da_nrows)
            return LH5VectorOfVectors(data_array=data_array, lensum_array=lensum_array, attrs=h5f[name].attrs)


        # read out all arrays by slicing
        if 'array' in datatype:
            ds_n_rows = h5f[name].shape[0]
            if n_rows is None or n_rows > ds_n_rows - start_row: 
                n_rows = ds_n_rows - start_row
            nda = h5f[name][start_row:start_row+n_rows]
            if elements == 'bool': nda = nda.astype(np.bool)
            attrs=h5f[name].attrs
            if datatype == 'array': 
                return LH5Array(nda=nda, attrs=attrs)
            if datatype == 'fixedsize_array': 
                return LH5FixedSizeArray(nda=nda, attrs=attrs)
            if datatype == 'array_of_equalsized_arrays': 
                return LH5ArrayOfEqualSizedArrays(nda=nda, dims=shape, attrs=attrs)

        print('LH5Store: don\'t know how to read datatype', datatype)
        return None


    def write_object(self, obj, name, lh5_file, group='/', start_row=0, n_rows=None, append=True):
        """Write an object into an lh5_file

        obj should be a LH5* object. 

        Set append to true for non-scalar objects if you want to append along
        axis 0 (the first dimension) (or axis 0 of non-scalar subfields of
        structs)
        """
        lh5_file = self.gimme_file(lh5_file, mode = 'a' if append else 'r+')
        group = self.gimme_group(group, lh5_file)

        # FIXME: fail if trying to overwrite an existing object without appending?
        # FIXME: even in append mode, if you try to overwrite a ds, it will fail
        # unless you delete the ds first

        # struct or table
        if isinstance(obj, LH5Struct):
            group = self.gimme_group(name, group, grp_attrs=obj.attrs)
            fields = obj.keys()
            for field in obj.keys():
                self.write_object(obj[field], 
                                  field, 
                                  lh5_file, 
                                  group, 
                                  start_row=start_row,
                                  n_rows=n_rows,
                                  append=append)
            return

        # scalars
        elif isinstance(obj, LH5Scalar):
            ds = group.create_dataset(name, shape=(), data=obj.value)
            ds.attrs.update(obj.attrs)
            return

 
        # vector of vectors
        elif isinstance(obj, LH5VectorOfVectors):
            group = self.gimme_group(name, group, grp_attrs=obj.attrs)
            if n_rows is None or n_rows > obj.lensum_array.nda.shape[0] - start_row:
                n_rows = obj.lensum_array.nda.shape[0] - start_row
            self.write_object(obj.lensum_array,
                              'cumulative_length', 
                              lh5_file, 
                              group, 
                              start_row=start_row,
                              n_rows=n_rows,
                              append=append)
            da_start = 0 if start_row == 0 else obj.lensum_array.nda[start_row-1]
            da_n_rows = obj.lensum_array.nda[n_rows-1] - da_start
            self.write_object(obj.data_array,
                              'flattened_data', 
                              lh5_file, 
                              group, 
                              start_row=da_start,
                              n_rows=da_n_rows,
                              append=append)
            return

        # if we get this far, must be one of the LH5Array types
        elif isinstance(obj, LH5Array): 
            if n_rows is None or n_rows > obj.nda.shape[0] - start_row:
                n_rows = obj.nda.shape[0] - start_row
            nda = obj.nda[start_row:start_row+n_rows]
            if nda.dtype.name == 'bool': nda = nda.astype(np.uint8)
            # need to create dataset from ndarray the first time for speed
            # creating an empty dataset and appending to that is super slow!
            if not append or name not in group:
                maxshape = list(nda.shape)
                maxshape[0] = None
                maxshape = tuple(maxshape)
                ds = group.create_dataset(name, data=nda, maxshape=maxshape)
                ds.attrs.update(obj.attrs)
                return
            
            # Now append
            ds = group[name]
            old_len = ds.shape[0]
            add_len = nda.shape[0]
            ds.resize(old_len + add_len, axis=0)
            ds[-add_len:] = nda
            return

        else:
            print('LH5Store: do not know how to write', name, 'of type', type(obj).__name__)
            return




def get_ccc(crate, card, channel):
    return (crate << 9) + ((card & 0xf) << 4) + (channel & 0xf)


def get_crate(ccc):
    return ccc >> 9


def get_card(ccc):
    return (ccc >> 4) & 0x1f


def get_channel(ccc):
    return ccc & 0xf

