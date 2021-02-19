import sys, os
import numpy as np
import h5py
import fnmatch
from collections import defaultdict
from bisect import bisect_left

from .lh5_utils import *
from .scalar import Scalar
from .struct import Struct
from .table import Table
from .array import Array
from .fixedsizearray import FixedSizeArray
from .arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from .vectorofvectors import VectorOfVectors

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
        if mode == 'r' and not os.path.exists(full_path):
            print('file not found:', full_path)
            return None
        h5f = h5py.File(full_path, mode)
        if self.keep_open: self.files[lh5_file] = h5f
        return h5f


    def gimme_group(self, group, base_group, grp_attrs=None):
        if isinstance(group, h5py.Group): return group
        if group in base_group: return base_group[group]
        group = base_group.create_group(group)
        if grp_attrs is not None: group.attrs.update(grp_attrs)
        return group


    def ls(self, lh5_file, lh5_group=''):
        """Print a list of the group names in the lh5 file in the style of a
        Unix ls command. Supports wildcards."""
        # To use recursively, make lh5_file a h5group instead of a string
        if isinstance(lh5_file, str):
            lh5_file = self.gimme_file(lh5_file, 'r')
            
        if lh5_group=='':
            lh5_group='*'
            
        splitpath = lh5_group.split('/', 1)
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



    def read_object(self, name, lh5_file, start_row=0, n_rows=sys.maxsize, idx=None, field_mask=None, obj_buf=None, obj_buf_start=0, verbosity=0):
        """ Read LH5 object data from a file

        Parameters
        ----------
        name : str
            Name of the lh5 object to be read (including its group path)
        lh5_file : str or h5py File object, or a list of either
            The file(s) containing the object to be read oad out. If a list of
            files, array-like object data will be contatenated into the output
            object
        start_row : int (optional)
            Starting entry for the object read (for array-like objects). For a
            list of files, only applies to the first file.
        n_rows : int (optional)
            The maximum number of rows to read (for array-like objects). The
            actual number of rows read will be returned as one of the return
            values (see below)
        idx : index array or index tuple, or a list of index arrays/tuples (optional)
            For numpy-style "fancying indexing" for the read. Used to read out
            rows that pass some selection criteria. Only selection along the 1st
            axis is supported, so tuple arguments must be one-tuples.  If n_rows
            is not false, idx will be truncated to n_rows before reading. To use
            with a list of files, can pass in a list of idx's (one for each
            file) or use a long contiguous list (e.g. built from a previous
            identical read). If used in conjunction with start_row and n_rows,
            will be sliced to obey those constraints, where n_rows is
            interpreted as the (max) number of -selected- values (in idx) to be
            read out.
        field_mask : dict or defaultdict { str : bool } (optional)
            For tables and structs, determines which fields get written out.
            Only applies to immediate fields of the requested objects. If a dict
            is used, a defaultdict will be made with the default set to the
            opposite of the first element in the dict. This way if one specifies
            a few fields at "false", all but those fields will be read out,
            while if one specifies just a few fields as "true", only those
            fields will be read out.
        obj_buf : lh5 object (optional)
            Read directly into memory provided in obj_buf. Note: the buffer will
            be expanded to accommodate the data requested. To maintain the
            buffer length, send in n_rows = len(obj_buf)
        obj_buf_start : int (optional)
            Start location in obj_buf for read. For concatenating data to
            array-like objects
        verbosity : bool (optional)
            Turn on verbosity for debugging

        Returns
        -------
        (object, n_rows_read) : tuple
            object is the read-out object
            n_rows_read is the number of rows successfully read out. Essential
            for arrays when the amount of data is smaller than the object
            buffer.  For scalars and structs n_rows_read will be "1". For tables
            it is redundant with table.loc
        """
        #TODO: Ian's idea: add an iterator so one can do something like
        #      for data in lh5iterator(file, chunksize, nentries, ...):
        #          proc.execute()

        # Handle list-of-files recursively
        if isinstance(lh5_file, list):
            n_rows_read = 0
            for i, h5f in enumerate(lh5_file):
                if isinstance(idx, list) and len(idx) > 0 and not np.isscalar(idx[0]):
                    # a list of lists: must be one per file
                    idx_i = idx[i]
                elif idx is not None: 
                    # make idx a proper tuple if it's not one already
                    if not (isinstance(idx, tuple) and len(idx) == 1): idx = (idx,)
                    # idx is a long continuous array
                    n_rows_i = self.read_n_rows(name, h5f)
                    # find the length of the subset of idx that contains indices
                    # that are less than n_rows_i
                    n_rows_to_read_i = bisect_left(idx[0], n_rows_i)
                    # now split idx into idx_i and the remainder
                    idx_i = (idx[0][:n_rows_to_read_i],)
                    idx = (idx[0][n_rows_to_read_i:]-n_rows_i,)
                else: idx_i = None
                n_rows_i = n_rows-n_rows_read
                obj_buf, n_rows_read_i = self.read_object(name,
                                                          lh5_file[i],
                                                          start_row=start_row,
                                                          n_rows=n_rows_i,
                                                          idx=idx_i,
                                                          field_mask=field_mask,
                                                          obj_buf=obj_buf,
                                                          obj_buf_start=obj_buf_start,
                                                          verbosity=verbosity)
                n_rows_read += n_rows_read_i
                if n_rows_read >= n_rows or obj_buf == None:
                    return obj_buf, n_rows_read
                start_row = 0
                obj_buf_start += n_rows_read_i
            return obj_buf, n_rows_read

        # start read from single file. fail if the object is not found
        if verbosity > 0: print("reading", name, "from", lh5_file)

        # get the file from the store
        h5f = self.gimme_file(lh5_file, 'r')
        if name not in h5f:
            print('Store:', name, "not in", lh5_file)
            return None, 0

        # make idx a proper tuple if it's not one already
        if not (isinstance(idx, tuple) and len(idx) == 1): 
            if idx is not None: idx = (idx,)

        # get the object's datatype
        if 'datatype' not in h5f[name].attrs:
            print('Store:', name, 'in file', lh5_file, 'is missing the datatype attribute')
            return None, 0
        datatype = h5f[name].attrs['datatype']
        datatype, shape, elements = parse_datatype(datatype)

        # Scalar
        # scalars are dim-0 datasets
        if datatype == 'scalar': 
            value = h5f[name][()]
            if elements == 'bool': value = np.bool(value)
            if obj_buf is not None:
                obj_buf.value = value
                obj_buf.attrs.update(h5f[name].attrs)
                return obj_buf, 1
            else: return Scalar(value=value, attrs=h5f[name].attrs), 1


        # Struct
        # recursively build a struct, return as a dictionary
        if datatype == 'struct':

            # ignore obj_buf.
            # TODO: could append new fields or overwrite/concat to existing
            # fields. If implemented, get_buffer() above should probably also
            # (optionally?) prep buffers for each field
            if obj_buf is not None:
                print("obj_buf not implemented for structs.  Returning new object")

            # build field_mask
            if field_mask is None: field_mask = defaultdict(lambda : True)
            elif isinstance(field_mask, dict):
                default = True
                if len(field_mask) > 0:
                    default = not field_mask[field_mask.keys[0]]
                field_mask = defaultdict(lambda : default, field_mask)
            elif not isinstance(field_mask, defaultdict):
                print('bad field_mask of type', type(field_mask).__name__)
                return None, 0

            # loop over fields and read
            obj_dict = {}
            for field in elements:
                if not field_mask[field]: continue
                # TODO: it's strange to pass start_row, n_rows, idx to struct
                # fields. If they all had shared indexing, they should be in a
                # table... Maybe should emit a warning? Or allow them to be
                # dicts keyed by field name?
                obj_dict[field], _ = self.read_object(name+'/'+field, 
                                                      h5f, 
                                                      start_row=start_row, 
                                                      n_rows=n_rows, 
                                                      idx=idx,
                                                      verbosity=verbosity)
            # modify datatype in attrs if a field_mask was used
            attrs = dict(h5f[name].attrs)
            if field_mask is not None:
                selected_fields = []
                for field in elements:
                    if field_mask[field]: selected_fields.append(field)
                attrs['datatype'] =  'struct' + '{' + ','.join(selected_fields) + '}'
            return Struct(obj_dict=obj_dict, attrs=attrs), 1

        # Below here is all array-like types. So trim idx if needed
        if idx is not None:
            # chop off indices < start_row
            i_first_valid = bisect_left(idx[0], start_row)
            idxa = idx[0][i_first_valid:]
            # don't readout more than n_rows indices
            idx = (idxa[:n_rows],) # works even if n_rows > len(idxa)

        # Table
        # read a table into a dataframe
        if datatype == 'table':
            col_dict = {}

            # build field_mask
            if field_mask is None: field_mask = defaultdict(lambda : True)
            elif isinstance(field_mask, dict):
                default = True
                if len(field_mask) > 0:
                    default = not (field_mask[list(field_mask.keys())[0]])
                field_mask = defaultdict(lambda : default, field_mask)
            elif not isinstance(field_mask, defaultdict):
                print('bad field_mask of type', type(field_mask).__name__)
                return None, 0

            # read out each of the fields
            rows_read = []
            for field in elements:
                if not field_mask[field] : continue
                fld_buf = None
                if obj_buf is not None:
                    if not isinstance(obj_buf, Table) or field not in obj_buf:
                        print("obj_buf for Table", name, 
                              "not formatted correctly. returning new object")
                        obj_buf = None
                    else: fld_buf = obj_buf[field]
                col_dict[field], n_rows_read = self.read_object(name+'/'+field, 
                                                                h5f, 
                                                                start_row=start_row, 
                                                                n_rows=n_rows,
                                                                idx=idx,
                                                                obj_buf=fld_buf,
                                                                obj_buf_start=obj_buf_start,
                                                                verbosity=verbosity)
                if obj_buf is not None and obj_buf_start+n_rows_read > len(obj_buf):
                    obj_buf.resize(obj_buf_start+n_rows_read, do_warn=(verbosity>0))
                rows_read.append(n_rows_read)
            # warn if all columns don't read in the same number of rows
            n_rows_read = rows_read[0]
            for n in rows_read[1:]:
                if n != n_rows_read:
                    print('table', name, 'got strange n_rows_read', n)
                    print(n_rows_read, 'was expected')

            # modify datatype in attrs if a field_mask was used
            attrs = dict(h5f[name].attrs)
            if field_mask is not None:
                selected_fields = []
                for field in elements:
                    if field_mask[field]: selected_fields.append(field)
                attrs['datatype'] =  'table' + '{' + ','.join(selected_fields) + '}'

            # fields have been read out, now return a table
            if obj_buf is None: 
                table = Table(col_dict=col_dict, attrs=attrs)
                # set (write) loc to end of tree
                table.loc = n_rows_read
                return table, n_rows_read
            else:
                # We have read all fields into the object buffer. Run
                # checks: All columns should be the same size. So update
                # table's size as necessary, warn if any mismatches are found
                obj_buf.resize(do_warn=True)
                # set (write) loc to end of tree
                obj_buf.loc = obj_buf_start+n_rows_read
                #check attributes
                if set(obj_buf.attrs.keys()) != set(attrs.keys()):
                    print('warning: attrs mismatch')
                    print('obj_buf.attrs:', obj_buf.attrs)
                    print('h5f['+name+'].attrs:', attrs)
                return obj_buf, n_rows_read

        # VectorOfVectors
        # read out vector of vectors of different size
        if elements.startswith('array'):
            if obj_buf is not None:
                if not isinstance(obj_buf, VectorOfVectors):
                    print("obj_buf for", name, "not a VectorOfVectors. returning new object")
                    obj_buf = None
            if idx is not None:
                print("warning: fancy indexed readout not implemented for vector of vectors, ignoring idx")
                # TODO: implement idx: first pull out all of cumulative length,
                # use it to build an idx for the data_array, then rebuild
                # cumulative length

            # read out cumulative_length
            cumulen_buf = None if obj_buf is None else obj_buf.cumulative_length
            cumulative_length, n_rows_read = self.read_object(name+'/cumulative_length', 
                                                              h5f, 
                                                              start_row=start_row, 
                                                              n_rows=n_rows,
                                                              obj_buf=cumulen_buf,
                                                              obj_buf_start=obj_buf_start,
                                                              verbosity=verbosity)
            # get a view of just what was read out for cleaner code below
            this_cumulen_nda = cumulative_length.nda[obj_buf_start:obj_buf_start+n_rows_read]

            # determine the start_row and n_rows for the flattened_data readout
            da_start = 0
            if start_row > 0 and n_rows_read > 0: 
                # need to read out the cumulen sample -before- the first sample
                # read above in order to get the starting row of the first
                # vector to read out in flattened_data
                da_start = h5f[name+'/cumulative_length'][start_row-1]

                # check limits for values that will be used subsequently
                if this_cumulen_nda[-1] < da_start:
                    print("warning: cumulative_length non-increasing between entries", 
                          start_row, "and", start_row+n_rows_read, "??")
                    print(this_cumulen_nda[-1], da_start, start_row, n_rows_read)

            # determine the number of rows for the flattened_data readout
            da_nrows = this_cumulen_nda[-1] if n_rows_read > 0 else 0

            # Now done with this_cumulen_nda, so we can clean it up to be ready
            # to match the in-memory version of flattened_data. Note: these
            # operations on the view change the original array because they are
            # numpy arrays, not lists.
            #
            # First we need to substract off the in-file offset for the start of
            # read for flattened_data
            this_cumulen_nda -= da_start

            # Then, if we started with a partially-filled buffer, add the
            # appropriate offset for the start of the in-memory flattened
            # data for this read.
            da_buf_start = 0
            if obj_buf_start > 0:
                da_buf_start = cumulative_length.nda[obj_buf_start-1]
                this_cumulen_nda += da_buf_start

            # Now prepare the object buffer if necessary
            da_buf = None 
            if obj_buf is not None:
                da_buf = obj_buf.flattened_data
                # grow da_buf if necessary to hold the data
                dab_size = da_buf_start + da_nrows
                if len(da_buf) < dab_size: da_buf.resize(dab_size)

            # now read
            flattened_data, dummy_rows_read = self.read_object(name+'/flattened_data', 
                                                               h5f, 
                                                               start_row=da_start, 
                                                               n_rows=da_nrows,
                                                               idx=idx,
                                                               obj_buf=da_buf,
                                                               obj_buf_start=da_buf_start,
                                                               verbosity=verbosity)
            if obj_buf is not None: return obj_buf, n_rows_read
            return VectorOfVectors(flattened_data=flattened_data, 
                                   cumulative_length=cumulative_length, 
                                   attrs=h5f[name].attrs), n_rows_read


        # Array
        # FixedSizeArray
        # ArrayOfEqualSizedArrays
        # read out all arrays by slicing
        if 'array' in datatype:
            if obj_buf is not None:
                if not isinstance(obj_buf, Array):
                    print("obj_buf for", name, "not an Array. returning new object")
                    obj_buf = None

            # compute the number of rows to read
            # we culled idx above for start_row and n_rows, now we have to apply
            # the constraint of the length of the dataset
            ds_n_rows = h5f[name].shape[0]
            if idx is not None:
                if len(idx[0]) > 0 and idx[0][-1] >= ds_n_rows:
                    print("warning: idx indexed past the end of the array in the file. Culling...")
                    n_rows_to_read = bisect_left(idx[0], ds_n_rows)
                    idx = (idx[0][:n_rows_to_read],)
                if len(idx[0]) == 0: print("warning: idx empty after culling.")
                n_rows_to_read = len(idx[0])
            else: n_rows_to_read = ds_n_rows - start_row
            if n_rows_to_read > n_rows: n_rows_to_read = n_rows

            # prepare the selection for the read. Use idx if available
            if idx is not None: source_sel = idx
            else: source_sel = np.s_[start_row:start_row+n_rows_to_read]

            # Now read the array
            if obj_buf is not None and n_rows_to_read > 0:
                buf_size = obj_buf_start + n_rows_to_read
                if len(obj_buf) < buf_size: obj_buf.resize(buf_size)
                dest_sel = np.s_[obj_buf_start:buf_size]
                # NOTE: if your script fails on this line, it may be because you
                # have to apply this patch to h5py (or update h5py, if it's
                # fixed): https://github.com/h5py/h5py/issues/1792
                h5f[name].read_direct(obj_buf.nda, source_sel, dest_sel)
            else: 
                if n_rows == 0: 
                    tmp_shape = (0,) + h5f[name].shape[1:]
                    nda = np.empty(tmp_shape, h5f[name].dtype)
                else: nda = h5f[name][source_sel]

            # special handling for bools
            if elements == 'bool': nda = nda.astype(np.bool)

            # Finally, set attributes and return objects
            attrs=h5f[name].attrs
            if obj_buf is None:
                if datatype == 'array': 
                    return Array(nda=nda, attrs=attrs), n_rows_to_read
                if datatype == 'fixedsize_array': 
                    return FixedSizeArray(nda=nda, attrs=attrs), n_rows_to_read
                if datatype == 'array_of_equalsized_arrays': 
                    return ArrayOfEqualSizedArrays(nda=nda, 
                                                   dims=shape, 
                                                   attrs=attrs), n_rows_to_read
            else:
                if set(obj_buf.attrs.keys()) != set(attrs.keys()):
                    print('warning: attrs mismatch')
                    print('obj_buf.attrs:', obj_buf.attrs)
                    print('h5f['+name+'].attrs:', attrs)
                return obj_buf, n_rows_to_read


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
            if n_rows is None or n_rows > obj.cumulative_length.nda.shape[0] - start_row:
                n_rows = obj.cumulative_length.nda.shape[0] - start_row

            # if appending we need to add an appropriate offset to the
            # cumulative lengths as appropriate for the in-file object
            offset = 0
            if append and 'cumulative_length' in group:
                len_cl = len(group['cumulative_length']) 
                if len_cl > 0: offset = group['cumulative_length'][len_cl-1]
            # Add offset to obj.cumulative_length itself to avoid memory allocation. 
            # Then subtract it off after writing!
            obj.cumulative_length.nda += offset
            self.write_object(obj.cumulative_length,
                              'cumulative_length', 
                              lh5_file, 
                              group, 
                              start_row=start_row,
                              n_rows=n_rows,
                              append=append)
            obj.cumulative_length.nda -= offset

            # now write data array. Only write rows with data.
            da_start = 0 if start_row == 0 else obj.cumulative_length.nda[start_row-1]
            da_n_rows = obj.cumulative_length.nda[n_rows-1] - da_start
            self.write_object(obj.flattened_data,
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
            cumulen_buf = None
            return self.read_n_rows(name+'/cumulative_length', h5f)
        
        # read out all arrays by slicing
        if 'array' in datatype:
            # compute the number of rows to read
            return h5f[name].shape[0]

        print('Store: don\'t know how to read datatype', datatype)
        return None


def load_nda(f_list, par_list, lh5_group='', idx_list=None, verbose=True):
    """ Build a dictionary of ndarrays from lh5 data

    Given a list of files, a list of lh5 table parameters, and an optional group
    path, return a numpy array with all values for each parameter.

    Parameters
    ----------
    f_list : str or list of str's
        A list of files. Can contain wildcards
    par_list : list of str's
        A list of parameters to read from each file
    lh5_group : str (optional)
        Optional group path within which to find the specified parameters
    idx_list : list of index arrays
        For fancy-indexed reads. Must be one idx array for each file in f_list
    verbose : bool
        Print info on loaded data

    Returns
    -------
    par_data : dict
        A dictionary of the parameter data keyed by the elements of par_list.
        Each entry contains the data for the specified parameter concatenated
        over all files in f_list
    """
    if isinstance(f_list, str): f_list = [f_list]
    # Expand wildcards
    f_list = [f for f_wc in f_list for f in sorted(glob.glob(os.path.expandvars(f_wc)))]
    if verbose:
        print("loading data for", *f_list)

    sto = Store()
    par_data = {par : [] for par in par_list}
    for f in f_list:
        for par in par_list:
            data, _ = sto.read_object(f'{lh5_group}/{par}', f)
            if not data: continue
            par_data[par].append(data.nda)
    par_data = {par : np.concatenate(par_data[par]) for par in par_list}
    return par_data


def load_dfs(f_list, par_list, lh5_group='', idx_list=None, verbose=True):
    """ Build a pandas dataframe from lh5 data

    Given a list of files (can use wildcards), a list of lh5 columns, and
    optionally the group path, return a pandas DataFrame with all values for
    each parameter.

    Parameters
    ----------
    See load_nda for parameter specification

    Returns
    -------
    df : pandas.DataFrame
        Contains columns for each parameter in par_list, and rows containing all
        data for the associated parameters concatenated over all files in f_list
    """
    return pd.DataFrame( load_nda(f_list, par_list, lh5_group, verbose) )
