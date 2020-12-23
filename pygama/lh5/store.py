import os
import numpy as np
import h5py
import fnmatch

from .lh5 import *
from .struct import Struct
from .array import Array
from .fsarray import Array
from .aoesa import ArrayOfEqualSizedArrays
from .scalar import Scalar
from .table import Table

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

            
    def get_buffer(self, name, lh5_file, size=None):
        """
        Returns an lh5 object appropriate for use as a pre-allocated buffer
        in a read loop. Sets size to size if object has a size.
        """
        obj, n_rows = self.read_object(name, lh5_file, n_rows=0)
        if hasattr(obj, 'resize') and size is not None: obj.resize(new_size=size)
        return obj



    def read_object(self, name, lh5_file, start_row=0, n_rows=None, obj_buf=None):
        """
        Returns tuple (obj, n_rows_read) for data at path=name in lh5_file
        obj is an lh5 object. If obj_buf is provided, obj = obj_buf
        Set start_row, n_rows to read out a subset of the first data axis (when possible).
        When n_rows are requested but fewer are available, this will be
        reflected in n_rows_read.
        n_rows will be returned as '1' for objects that don't have rows.
        """
        #TODO: implement obj_buf. Ian's idea: add an iterator so one can do
        #      something like
        #      for data in lh5iterator(file, chunksize, nentries, ...):
        #          proc.execute()

        h5f = self.gimme_file(lh5_file, 'r')
        if name not in h5f:
            print('Store:', name, "not in", lh5_file)
            return None, 0

        # get the datatype
        if 'datatype' not in h5f[name].attrs:
            print('Store:', name, 'in file', lh5_file, 'is missing the datatype attribute')
            return None, 0
        datatype = h5f[name].attrs['datatype']
        datatype, shape, elements = parse_datatype(datatype)

        # scalars are dim-0 datasets
        if datatype == 'scalar': 
            if obj_buf is not None:
                print("obj_buf not implemented for scalars.  Returning new object")
            if elements == 'bool':
                return Scalar(value=np.bool(h5f[name][()]), attrs=h5f[name].attrs), 1
            return Scalar(value=h5f[name][()], attrs=h5f[name].attrs), 1

        # recursively build a struct, return as a dictionary
        if datatype == 'struct':
            if obj_buf is not None:
                print("obj_buf not implemented for structs.  Returning new object")
            obj_dict = {}
            for field in elements:
                obj_dict[field], _ = self.read_object(name+'/'+field, h5f, start_row, n_rows)
            return Struct(obj_dict=obj_dict, attrs=h5f[name].attrs), 1

        # read a table into a dataframe
        if datatype == 'table':
            col_dict = {}

            # read out each of the fields
            rows_read = []
            for field in elements:
                fld_buf = None
                if obj_buf is not None:
                    if not isinstance(obj_buf, Table) or field not in obj_buf:
                        print("obj_buf for Table", name, 
                              "not formatted correctly. returning new object")
                        obj_buf = None
                    else: 
                        if n_rows is None: n_rows = len(obj_buf)
                        fld_buf = obj_buf[field]
                col_dict[field], n_rows_read = self.read_object(name+'/'+field, 
                                                                h5f, 
                                                                start_row=start_row, 
                                                                n_rows=n_rows,
                                                                obj_buf=fld_buf)
                rows_read.append(n_rows_read)
            # warn if all columns don't read in the same number of rows
            n_rows_read = rows_read[0]
            for n in rows_read[1:]:
                if n != n_rows_read:
                    print('table', name, 'got strange n_rows_read', n)
                    print(n_rows_read, 'was expected')

            # fields have been read out, now return a table
            if obj_buf is None: 
                table = Table(col_dict=col_dict, attrs=h5f[name].attrs)
                # set (write) loc to end of tree
                table.loc = n_rows_read
                return table, n_rows_read
            else:
                # We have read all fields into the object buffer. Run
                # checks: All columns should be the same size. So update
                # table's size as necessary, warn if any mismatches are found
                obj_buf.resize(do_warn=True)
                # set (write) loc to end of tree
                obj_buf.loc = n_rows_read
                #check attributes
                if set(obj_buf.attrs.keys()) != set(h5f[name].attrs.keys()):
                    print('warning: attrs mismatch')
                    print('obj_buf.attrs:', obj_buf.attrs)
                    print('h5f['+name+'].attrs:', h5f[name].attrs)
                return obj_buf, n_rows_read

        # read out vector of vectors of different size
        if elements.startswith('array'):
            if obj_buf is not None:
                if not isinstance(obj_buf, VectorOfVectors):
                    print("obj_buf for", name, "not a VectorOfVectors. returning new object")
                    obj_buf = None
                elif n_rows is None: n_rows = len(obj_buf)
            lensum_buf = None if obj_buf is None else obj_buf.lensum_array
            lensum_array, n_rows_read = self.read_object(name+'/cumulative_length', 
                                                         h5f, 
                                                         start_row=start_row, 
                                                         n_rows=n_rows,
                                                         obj_buf=lensum_buf)
            da_start = 0
            if start_row > 0 and n_rows_read > 0: 
                da_start = h5f[name+'/cumulative_length'][start_row-1]
            da_nrows = lensum_array.nda[n_rows_read-1] - da_start if n_rows_read > 0 else 0
            da_buf = None 
            if obj_buf is not None:
                da_buf = obj_buf.data_array
                # grow da_buf if necessary to hold the data
                if len(da_buf) < da_nrows: da_buf.resize(da_nrows)
            data_array, dummy_rows_read = self.read_object(name+'/flattened_data', 
                                                           h5f, 
                                                           start_row=da_start, 
                                                           n_rows=da_nrows,
                                                           obj_buf=da_buf)
            if obj_buf is not None: return obj_buf, n_rows_read
            return VectorOfVectors(data_array=data_array, 
                                   lensum_array=lensum_array, 
                                   attrs=h5f[name].attrs), n_rows_read


        # read out all arrays by slicing
        if 'array' in datatype:
            if obj_buf is not None:
                if not isinstance(obj_buf, Array):
                    print("obj_buf for", name, "not an Array. returning new object")
                    obj_buf = None
                elif n_rows is None: n_rows = len(obj_buf)

            # compute the number of rows to read
            ds_n_rows = h5f[name].shape[0]
            if n_rows is None or n_rows > ds_n_rows - start_row: 
                n_rows = ds_n_rows - start_row

            nda = None
            if obj_buf is not None:
                nda = obj_buf.nda
                if n_rows > 0:
                    h5f[name].read_direct(nda, 
                                          np.s_[start_row:start_row+n_rows], 
                                          np.s_[0:n_rows])
            else: 
                if n_rows == 0: 
                    tmp_shape = (0,) + h5f[name].shape[1:]
                    nda = np.empty(tmp_shape, h5f[name].dtype)
                else: nda = h5f[name][start_row:start_row+n_rows]
            if elements == 'bool': nda = nda.astype(np.bool)
            attrs=h5f[name].attrs
            if n_rows < 0: n_rows = 0
            if obj_buf is None:
                if datatype == 'array': 
                    return Array(nda=nda, attrs=attrs), n_rows
                if datatype == 'fixedsize_array': 
                    return FixedSizeArray(nda=nda, attrs=attrs), n_rows
                if datatype == 'array_of_equalsized_arrays': 
                    return ArrayOfEqualSizedArrays(nda=nda, 
                                                   dims=shape, 
                                                   attrs=attrs), n_rows
            else:
                if set(obj_buf.attrs.keys()) != set(attrs.keys()):
                    print('warning: attrs mismatch')
                    print('obj_buf.attrs:', obj_buf.attrs)
                    print('h5f['+name+'].attrs:', attrs)
                return obj_buf, n_rows


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
            # now write data array. Only write rows with data.
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
        
    def read_n_rows(self, name, lh5_file):
        """Look up the number of rows in an Array-like object called name
        in lh5_file. Return None if it is a scalar/struct."""
        # this is basically a stripped down version of read_object
        h5f = self.gimme_file(lh5_file, 'r')
        if name not in h5f:
            print('Store:', name, "not in", lh5_file)
            return None

        # get the datatype
        if 'datatype' not in h5f[name].attrs:
            print('Store:', name, 'in file', lh5_file, 'is missing the datatype attribute')
            return None, 0
        datatype = h5f[name].attrs['datatype']
        datatype, shape, elements = parse_datatype(datatype)

        # scalars are dim-0 datasets
        if datatype == 'scalar': 
            return None

        # recursively build a struct, return as a dictionary
        if datatype == 'struct':
            return None
        
        # read a table into a dataframe
        if datatype == 'table':
            # read out each of the fields
            rows_read = None
            for field in elements:
                fld_buf = None
                n_rows_read = self.read_n_rows(name+'/'+field, h5f)
                if not rows_read: rows_read = n_rows_read
                elif rows_read != n_rows_read:
                    print('table', name, 'got strange n_rows_read', n)
                    print(n_rows_read, 'was expected')
            return rows_read
        
        # read out vector of vectors of different size
        if elements.startswith('array'):
            lensum_buf = None
            return self.read_n_rows(name+'/cumulative_length', h5f)
        
        # read out all arrays by slicing
        if 'array' in datatype:
            # compute the number of rows to read
            return h5f[name].shape[0]

        print('Store: don\'t know how to read datatype', datatype)
        return None
