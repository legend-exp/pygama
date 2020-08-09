import os
import numpy as np
import pandas as pd
import h5py
import fnmatch

def get_lh5_datatype_name(obj):
    """Get the LH5 datatype name of an LH5 object"""
    if isinstance(obj, Table): return 'table'
    if isinstance(obj, Struct): return 'struct'
    if np.isscalar(obj): return get_lh5_element_type(obj)
    if isinstance(obj, FixedSizeArray): return 'fixedsize_array'
    if isinstance(obj, ArrayOfEqualSizedArrays): return 'array_of_equalsized_arrays'
    if isinstance(obj, Array): return 'array'
    if isinstance(obj, VectorOfVectors): return 'array'
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


def load_nda(f_list, par_list, tb_in):
    """
    given a list of files, a list of LH5 table parameters, and the HDF5 path,
    return a numpy array with all values for each parameter.
    """
    sto = Store()
    par_data = {par : [] for par in par_list}
    for f in f_list:
        for par in par_list:
            data = sto.read_object(f'{tb_in}/{par}', f)
            par_data[par].append(data.nda)
    par_data = {par : np.concatenate(par_data[par]) for par in par_list}
    return par_data


def load_dfs(f_list, par_list, tb_in):
    """
    given a list of files, a list of LH5 columns, and the HDF5 path,
    return a pandas DataFrame with all values for each parameter.
    """
    sto = Store()
    dfs = []
    for f in f_list:
        data = sto.read_object(f'{tb_in}', f)
        dfs.append(data.get_dataframe())
    return pd.concat(dfs)


class Struct(dict):
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


class Table(Struct):
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
        if not isinstance(obj, Table) and not isinstance(obj, Array) and not isinstance(obj, VectorOfVectors):
            print('Table: Error: cannot add field of type', type(obj).__name__)
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


class Scalar:
    """Holds just a value and some attributes (datatype, units, ...)
    """
    def __init__(self, value, attrs={}):
        self.value = value
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != get_lh5_element_type(self.value):
                print('Scalar: Warning: datatype does not match value!')
                print('datatype: ', self.attrs['datatype'])
                print('type(value): ', type(value).__name__)
        else: self.attrs['datatype'] = get_lh5_element_type(self.value)


class Array:
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


class FixedSizeArray(Array):
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
        

class ArrayOfEqualSizedArrays(Array):
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


class VectorOfVectors:
    """A variable-length array of variable-length arrays

    For now only a 1D vector of 1D vectors is supported. Internal representation
    is as two ndarrays, one to store the flattened data contiguosly and one to
    store the cumulative sum of lengths of each vector. 
    """ 
    def __init__(self, data_array=None, lensum_array=None, shape_guess=None, dtype=None, attrs={}):
        if lensum_array is None:
            self.lensum_array = Array(shape=(shape_guess[0],), dtype='uint32')
        else: self.lensum_array = lensum_array
        if data_array is None:
            length = np.prod(shape_guess)
            self.data_array = Array(shape=(length,), dtype=dtype)
        else: self.data_array = data_array
        self.dtype = self.data_array.dtype
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print('VectorOfVectors: Warning: datatype does not match dtype!')
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
            print('VectorOfVectors: Error: bad i_vec', i_vec)
            return 
        if len(nda.shape) != 1:
            print('VectorOfVectors: Error: nda had bad shape', nda.shape)
            return
        start = 0 if i_vec == 0 else self.lensum_array.nda[i_vec-1]
        end = start + len(nda)
        while end >= len(self.data_array.nda):
            self.data_array.nda.resize(2*len(self.data_array.nda))
        self.data_array.nda[start:end] = nda
        self.lensum_array.nda[i_vec] = end


class Store:
    def __init__(self, base_path='', keep_open=False):
        self.base_path = base_path
        self.keep_open = keep_open
        self.files = {}


    def gimme_file(self, lh5_file, mode):
        if isinstance(lh5_file, h5py.File): return lh5_file
        if lh5_file in self.files.keys(): return self.files[lh5_file]
        if self.base_path != '': full_path = self.base_path + '/' + lh5_file
        else: full_path = lh5_file
        if mode != 'r':
            directory = os.path.dirname(full_path)
            if directory != '' and not os.path.exists(directory): 
                os.makedirs(directory)
        h5f = h5py.File(full_path, mode)
        if self.keep_open: self.files[lh5_file] = h5f
        return h5f


    def gimme_group(self, group, base_group, grp_attrs=None):
        if isinstance(group, h5py.Group): return group
        if group in base_group: return base_group[group]
        group = base_group.create_group(group)
        if grp_attrs is not None: group.attrs.update(grp_attrs)
        return group


    def ls(self, lh5_file, group_path=''):
        """Print a list of the group names in the lh5 file in the style of a
        Unix ls command. Supports wildcards."""
        # To use recursively, make lh5_file a h5group instead of a string
        if isinstance(lh5_file, str):
            lh5_file = self.gimme_file(lh5_file, 'r')
            
            
        if group_path=='':
            group_path='*'
            
            
        
        splitpath = group_path.split('/', 1)
        matchingkeys = fnmatch.filter(lh5_file.keys(), splitpath[0])
        ret = []
        
        
        if len(splitpath)==1:
            
            return matchingkeys
        else:
            ret = []
            for key in matchingkeys:
                ret.extend([key + '/' + path for path in self.ls(lh5_file[key], splitpath[1])])
            return ret

            
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
            print('Store:', name, "not in", lh5_file)
            return None

        # get the datatype
        if 'datatype' not in h5f[name].attrs:
            print('Store:', name, 'in file', lh5_file, 'is missing the datatype attribute')
            return None
        datatype = h5f[name].attrs['datatype']
        datatype, shape, elements = parse_datatype(datatype)

        # scalars are dim-0 datasets
        if datatype == 'scalar': 
            if elements == 'bool':
                return Scalar(value=np.bool(h5f[name][()]), attrs=h5f[name].attrs)
            return Scalar(value=h5f[name][()], attrs=h5f[name].attrs)

        # recursively build a struct, return as a dictionary
        if datatype == 'struct':
            obj_dict = {}
            for field in elements:
                obj_dict[field] = self.read_object(name+'/'+field, h5f, start_row, n_rows)
            return Struct(obj_dict=obj_dict, attrs=h5f[name].attrs)

        # read a table into a dataframe
        if datatype == 'table':
            # TODO: set the size and loc parameters
            col_dict = {}
            for field in elements:
                col_dict[field] = self.read_object(name+'/'+field, 
                                                   h5f, 
                                                   start_row=start_row, 
                                                   n_rows=n_rows)
            return Table(col_dict=col_dict, attrs=h5f[name].attrs)

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
            return VectorOfVectors(data_array=data_array, lensum_array=lensum_array, attrs=h5f[name].attrs)


        # read out all arrays by slicing
        if 'array' in datatype:
            ds_n_rows = h5f[name].shape[0]
            if n_rows is None or n_rows > ds_n_rows - start_row: 
                n_rows = ds_n_rows - start_row
            nda = h5f[name][start_row:start_row+n_rows]
            if elements == 'bool': nda = nda.astype(np.bool)
            attrs=h5f[name].attrs
            if datatype == 'array': 
                return Array(nda=nda, attrs=attrs)
            if datatype == 'fixedsize_array': 
                return FixedSizeArray(nda=nda, attrs=attrs)
            if datatype == 'array_of_equalsized_arrays': 
                return ArrayOfEqualSizedArrays(nda=nda, dims=shape, attrs=attrs)

        print('Store: don\'t know how to read datatype', datatype)
        return None


    def write_object(self, obj, name, lh5_file, group='/', start_row=0, n_rows=None, append=True):
        """Write an object into an lh5_file

        obj should be a LH5 object. 

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
        if isinstance(obj, Struct):
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
        elif isinstance(obj, Scalar):
            ds = group.create_dataset(name, shape=(), data=obj.value)
            ds.attrs.update(obj.attrs)
            return

 
        # vector of vectors
        elif isinstance(obj, VectorOfVectors):
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

        # if we get this far, must be one of the Array types
        elif isinstance(obj, Array): 
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
            print('Store: do not know how to write', name, 'of type', type(obj).__name__)
            return
