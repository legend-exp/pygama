import fnmatch
import glob
import os
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

from .array import Array
from .arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from .fixedsizearray import FixedSizeArray
from .lgdo_utils import *
from .scalar import Scalar
from .struct import Struct
from .table import Table
from .vectorofvectors import VectorOfVectors
from .waveform_table import WaveformTable


class LH5Store:
    """
    DOCME
    """

    def __init__(self, base_path='', keep_open=False):
        """
        Parameters
        ----------
        base_path : str
            directory path to prepend to LH5 files
        keep_open : bool
            whether to keep files open by storing the h5py objects as class
            attributes
        """
        self.base_path = base_path
        self.keep_open = keep_open
        self.files = {}


    def gimme_file(self, lh5_file, mode='r', verbosity=0):
        """
        Returns a h5py file object from the store or creates a new one

        Parameters
        ----------
        lh5_file : str
            LH5 file name
        mode : str, default='r'
            mode in which to open file. See :class:`h5py.File` documentation
        verbosity : bool
            verbosity

        Returns
        -------
        file_obj : h5py.File
        """
        if isinstance(lh5_file, h5py.File): return lh5_file
        if lh5_file in self.files.keys(): return self.files[lh5_file]
        if self.base_path != '': full_path = self.base_path + '/' + lh5_file
        else: full_path = lh5_file
        if mode != 'r':
            directory = os.path.dirname(full_path)
            if directory != '' and not os.path.exists(directory):
                if verbosity > 0: print(f'making path {directory}')
                os.makedirs(directory)
        if mode == 'r' and not os.path.exists(full_path):
            raise FileNotFoundError(f'file {full_path} not found')
        if verbosity > 0 and mode != 'r' and os.path.exists(full_path):
            print(f'opening existing file {full_path} in mode {mode}...')
        h5f = h5py.File(full_path, mode)
        if self.keep_open: self.files[lh5_file] = h5f
        return h5f


    def gimme_group(self, group, base_group, grp_attrs=None, overwrite=False, verbosity=0):
        """
        Returns an existing h5py group from a base group or creates a new one.
        Can also set (or replace) group attributes

        Parameters
        ----------
        group : str
            name of the HDF5 group
        base_group : h5py.File or h5py.Group
            HDF5 group to be used as a base
        grp_attrs : dict, default None
            HDF5 group attributes
        overwrite : bool, default False
            whether overwrite group attributes, ignored is grp_attrs is None
        verbosity : bool
            verbosity
        """
        if not isinstance(group, h5py.Group):
            if group in base_group: group = base_group[group]
            else:
                group = base_group.create_group(group)
                if grp_attrs is not None: group.attrs.update(grp_attrs)
                return group
        if grp_attrs is not None:
            if not overwrite and grp_attrs != group.attrs:
                print('warning: grp_attrs != group.attrs but overwrite not set')
                print('ignoring grp_attrs')
            elif overwrite:
                if verbosity > 0: print(f'overwriting {group}.attrs...')
                group.attrs = {}
                group.attrs.update(grp_attrs)
        return group


    def get_buffer(self, name, lh5_file, size=None, field_mask=None):
        """
        Returns an lh5 object appropriate for use as a pre-allocated buffer
        in a read loop. Sets size to size if object has a size.
        """
        obj, n_rows = self.read_object(name, lh5_file, n_rows=0, field_mask=field_mask)
        if hasattr(obj, 'resize') and size is not None: obj.resize(new_size=size)
        return obj


    def read_object(self, name, lh5_file, start_row=0, n_rows=sys.maxsize, idx=None,
                    field_mask=None, obj_buf=None, obj_buf_start=0, verbosity=0):
        """
        Read LH5 object data from a file

        Parameters
        ----------
        name : str
            Name of the lh5 object to be read (including its group path)
        lh5_file : str or h5py File object, or a list of either
            The file(s) containing the object to be read out. If a list of
            files, array-like object data will be concatenated into the output
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
        field_mask : dict or defaultdict { str : bool } or list/tuple (optional)
            For tables and structs, determines which fields get written out.
            Only applies to immediate fields of the requested objects. If a dict
            is used, a defaultdict will be made with the default set to the
            opposite of the first element in the dict. This way if one specifies
            a few fields at "false", all but those fields will be read out,
            while if one specifies just a few fields as "true", only those
            fields will be read out. If a list is provided, the listed fields
            will be set to "true", while the rest will default to "false".
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
        # Handle list-of-files recursively
        if not isinstance(lh5_file, (str, h5py._hl.files.File)):
            lh5_file = list(lh5_file)
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
        h5f = self.gimme_file(lh5_file, 'r', verbosity=verbosity)
        if not h5f or name not in h5f:
            print('LH5Store:', name, "not in", lh5_file)
            return None, 0

        # make idx a proper tuple if it's not one already
        if not (isinstance(idx, tuple) and len(idx) == 1):
            if idx is not None: idx = (idx,)

        # get the object's datatype
        if 'datatype' not in h5f[name].attrs:
            print('LH5Store:', name, 'in file', lh5_file, 'is missing the datatype attribute')
            return None, 0
        datatype = h5f[name].attrs['datatype']
        datatype, shape, elements = parse_datatype(datatype)

        # check field_mask and make it a default dict
        if datatype == 'struct' or datatype == 'table':
            if field_mask is None: field_mask = defaultdict(lambda : True)
            elif isinstance(field_mask, dict):
                default = True
                if len(field_mask) > 0:
                    default = not field_mask[field_mask.keys[0]]
                field_mask = defaultdict(lambda : default, field_mask)
            elif isinstance(field_mask, (list, tuple)):
                field_mask = defaultdict(lambda : False, { field : True for field in field_mask} )
            elif not isinstance(field_mask, defaultdict):
                print('bad field_mask of type', type(field_mask).__name__)
                return None, 0
        elif field_mask is not None:
            print(f'Warning: datatype {datatype} does not accept a field_mask')


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

        # Table or WaveformTable
        if datatype == 'table':
            col_dict = {}

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
                # if col_dict contains just 3 objects called t0, dt, and values,
                # return a WaveformTable
                if len(col_dict) == 3:
                    if 't0' in col_dict and 'dt' in col_dict and 'values' in col_dict:
                        table = WaveformTable(t0=col_dict['t0'],
                                              dt=col_dict['dt'],
                                              values=col_dict['values'])
                else: table = Table(col_dict=col_dict, attrs=attrs)
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
            # First we need to subtract off the in-file offset for the start of
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
            # (c and Julia store as uint8 so cast to bool)
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


        print('LH5Store: don\'t know how to read datatype', datatype)
        return None


    def write_object(self, obj, name, lh5_file, group='/', start_row=0, n_rows=None,
                     wo_mode='append', write_start=0, verbosity=0):
        """Write an object into an lh5_file

        obj should be a LH5 object. if object is array-like, writes n_rows
        starting from start_row in obj.

        wo_modes:

          - 'write_safe' or 'w': only proceed with writing if the object does
            not already exist in the file
          - 'append' or 'a': append along axis 0 (the first dimension) of
            array-like objects and array-like subfields of structs. Scalar
            objects get overwritten
          - 'overwrite' or 'o': replace data in the file if present, starting
            from write_start. Note: overwriting with write_start = end of
            array is the same as append
          - 'overwrite_file' or 'of': delete file if present prior to writing to
            it. write_start should be 0 (it's ignored)

        """
        if wo_mode == 'write_safe':  wo_mode = 'w'
        if wo_mode == 'append':  wo_mode = 'a'
        if wo_mode == 'overwrite': wo_mode = 'o'
        if wo_mode == 'overwrite_file':
            wo_mode = 'of'
            write_start = 0
        if wo_mode != 'w' and wo_mode != 'a' and wo_mode != 'o' and wo_mode != 'of':
            print(f'Unknown wo_mode {wo_mode}')
            return

        mode = 'w' if wo_mode == 'of' else 'a'
        lh5_file = self.gimme_file(lh5_file, mode=mode, verbosity=verbosity)
        group = self.gimme_group(group, lh5_file, verbosity=verbosity)
        if wo_mode == 'w' and name in group:
            print(f"can't overwrite {name} in wo_mode write_safe")
            return

        # struct or table or waveform table
        if isinstance(obj, Struct):
            group = self.gimme_group(name, group, grp_attrs=obj.attrs, overwrite=(wo_mode=='o'), verbosity=verbosity)
            fields = obj.keys()
            for field in obj.keys():
                self.write_object(obj[field],
                                  field,
                                  lh5_file,
                                  group=group,
                                  start_row=start_row,
                                  n_rows=n_rows,
                                  wo_mode=wo_mode,
                                  write_start=write_start,
                                  verbosity=verbosity)
            return

        # scalars
        elif isinstance(obj, Scalar):
            if verbosity > 0 and name in group:
                print('overwriting {name} in {group}')
            ds = group.create_dataset(name, shape=(), data=obj.value)
            ds.attrs.update(obj.attrs)
            return

        # vector of vectors
        elif isinstance(obj, VectorOfVectors):
            group = self.gimme_group(name, group, grp_attrs=obj.attrs, overwrite=(wo_mode=='o'), verbosity=verbosity)
            if n_rows is None or n_rows > obj.cumulative_length.nda.shape[0] - start_row:
                n_rows = obj.cumulative_length.nda.shape[0] - start_row

            # if appending we need to add an appropriate offset to the
            # cumulative lengths as appropriate for the in-file object
            offset = 0 # declare here because we have to subtract it off at the end
            if (wo_mode == 'a' or wo_mode == 'o') and 'cumulative_length' in group:
                len_cl = len(group['cumulative_length'])
                if wo_mode == 'a': write_start = len_cl
                if len_cl > 0: offset = group['cumulative_length'][write_start-1]
            # Add offset to obj.cumulative_length itself to avoid memory allocation.
            # Then subtract it off after writing!
            obj.cumulative_length.nda += offset
            self.write_object(obj.cumulative_length,
                              'cumulative_length',
                              lh5_file,
                              group=group,
                              start_row=start_row,
                              n_rows=n_rows,
                              wo_mode=wo_mode,
                              write_start=write_start,
                              verbosity=verbosity)
            obj.cumulative_length.nda -= offset

            # now write data array. Only write rows with data.
            da_start = 0 if start_row == 0 else obj.cumulative_length.nda[start_row-1]
            da_n_rows = obj.cumulative_length.nda[n_rows-1] - da_start
            self.write_object(obj.flattened_data,
                              'flattened_data',
                              lh5_file,
                              group=group,
                              start_row=da_start,
                              n_rows=da_n_rows,
                              wo_mode=wo_mode,
                              write_start=offset,
                              verbosity=verbosity)
            return

        # if we get this far, must be one of the Array types
        elif isinstance(obj, Array):
            if n_rows is None or n_rows > obj.nda.shape[0] - start_row:
                n_rows = obj.nda.shape[0] - start_row
            nda = obj.nda[start_row:start_row+n_rows]
            # hack to store bools as uint8 for c / Julia compliance
            if nda.dtype.name == 'bool': nda = nda.astype(np.uint8)
            # need to create dataset from ndarray the first time for speed
            # creating an empty dataset and appending to that is super slow!
            if (wo_mode != 'a' and write_start == 0) or name not in group:
                if verbosity > 0 and wo_mode == 'o' and name in group:
                    print(f'write_object: overwriting {name} in {group}')
                maxshape = list(nda.shape)
                maxshape[0] = None
                maxshape = tuple(maxshape)
                ds = group.create_dataset(name, data=nda, maxshape=maxshape)
                ds.attrs.update(obj.attrs)
                return

            # Now append or overwrite
            ds = group[name]
            old_len = ds.shape[0]
            if wo_mode == 'a': write_start = old_len
            add_len = write_start + nda.shape[0] - old_len
            ds.resize(old_len + add_len, axis=0)
            ds[write_start:] = nda
            return

        else:
            print('LH5Store: do not know how to write', name, 'of type', type(obj).__name__)
            return


    def read_n_rows(self, name, lh5_file):
        """Look up the number of rows in an Array-like object called name
        in lh5_file. Return None if it is a scalar/struct."""
        # this is basically a stripped down version of read_object
        h5f = self.gimme_file(lh5_file, 'r')
        if not h5f or name not in h5f:
            print('LH5Store:', name, "not in", lh5_file)
            return None

        # get the datatype
        if 'datatype' not in h5f[name].attrs:
            print('LH5Store:', name, 'in file', lh5_file, 'is missing the datatype attribute')
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
                    print('table', name, 'got strange n_rows_read', rows_read)
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

        print('LH5Store: don\'t know how to read datatype', datatype)
        return None

def ls(lh5_file, lh5_group=''):
    """Return a list of lh5 groups in the input file and group, similar
    to ls or h5ls. Supports wildcards in group names.

    Parameters
    ----------
    lh5_file : str
        name of file
    lh5_group : str
        group to search

    Returns
    -------
    groups : list of strs
        List of names of groups found
    """

    lh5_st = LH5Store()

    # To use recursively, make lh5_file a h5group instead of a string
    if isinstance(lh5_file, str):
        lh5_file = lh5_st.gimme_file(lh5_file, 'r')

    if lh5_group=='':
        lh5_group='*'

    splitpath = lh5_group.split('/', 1)
    matchingkeys = fnmatch.filter(lh5_file.keys(), splitpath[0])

    # if we gave a group name, go one deeper
    if len(matchingkeys)==1 and matchingkeys[0] == splitpath[0] \
       and isinstance(lh5_file[matchingkeys[0]], h5py.Group):
        splitpath.append('')
    ret = []

    if len(splitpath)==1:
        return matchingkeys
    else:
        ret = []
        for key in matchingkeys:
            ret.extend([key + '/' + path for path in ls(lh5_file[key], splitpath[1])])
        return ret


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
    idx_list : list of index arrays, or a list of such lists
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
    if isinstance(f_list, str):
        f_list = [f_list]
        if idx_list is not None:
            idx_list = [idx_list]
    if idx_list is not None and len(f_list) != len(idx_list):
        print(f"load_nda: f_list len ({len(f_list)}) != idx_list len ({len(idx_list)})!")
        return None

    # Expand wildcards
    f_list = [f for f_wc in f_list for f in sorted(glob.glob(os.path.expandvars(f_wc)))]
    if verbose:
        print("loading data for", *f_list)

    sto = LH5Store()
    par_data = {par : [] for par in par_list}
    for ii, f in enumerate(f_list):
        f = sto.gimme_file(f, 'r')
        for par in par_list:
            if f'{lh5_group}/{par}' not in f:
                print(f'{lh5_group}/{par} not in file {f_list[ii]}')
                return None
            if idx_list is None: data, _ = sto.read_object(f'{lh5_group}/{par}', f)
            else: data, _ = sto.read_object(f'{lh5_group}/{par}', f, idx=idx_list[ii])
            if not data: continue
            par_data[par].append(data.nda)
    par_data = {par : np.concatenate(par_data[par]) for par in par_list}
    return par_data


def load_dfs(f_list, par_list, lh5_group='', idx_list=None, verbose=True):
    """ Build a pandas dataframe from lh5 data

    Given a list of files (can use wildcards), a list of lh5 columns, and
    optionally the group path, return a pandas DataFrame with all values for
    each parameter.

    See :func:`load_nda` for parameter specification

    Returns
    -------
    df : pandas.DataFrame
        Contains columns for each parameter in par_list, and rows containing all
        data for the associated parameters concatenated over all files in f_list
    """
    return pd.DataFrame( load_nda(f_list, par_list, lh5_group=lh5_group, idx_list=idx_list, verbose=verbose) )


class LH5Iterator:
    """
    A class for iterating through one or more LH5 files, one block of entries
    at a time. This also accepts an entry list/mask to enable event selection,
    and a field mask.

    This class can be used either for random access: ::

        lh5_obj, n_rows = lh5_it.read(entry)

    to read the block of entries starting at entry. In case of multiple files
    or the use of an event selection, entry refers to a global event index
    across files and does not count events that are excluded by the selection.

    This can also be used as an iterator: ::

        for lh5_obj, entry, n_rows in LH5Iterator(...):
            # do the thing!

    This is intended for if you are reading a large quantity of data but
    want to limit your memory usage (particularly when reading in waveforms!).
    The lh5_obj that is read by this class is reused in order to avoid
    reallocation of memory; this means that if you want to hold on to data
    between reads, you will have to copy it somewhere!
    """
    def __init__(self, lh5_files, group, base_path='', entry_list=None,
                 entry_mask=None, field_mask=None, buffer_len=3200):
        """
        Parameters
        ----------
        lh5_files : str or list
            File or files to read from. May include wildcards and env vars
        group : str
            HDF5 group to read
        base_path : str
            HDF5 path to prepend
        entry_list : list-like or nested list-like of ints (optional)
            List of entry numbers to read. If a nested list is provided,
            expect one top-level list for each file, containing a list of
            local entries. If a list of ints is provided, use global entries
        entry_mask : array of bools or list of arrays of bools (optional)
            Mask of entries to read. If a list of arrays is provided, expect
            one for each file. Ignore if a selection list is provided...
        field_mask : dict or defaultdict { str : bool } or list/tuple (optional)
            Mask of which fields to read. See read_object for more details.
        buffer_len : int (default 3200)
            Number of entries to read at a time while iterating through files
        """
        self.lh5_st = LH5Store(base_path=base_path, keep_open=True)

        # List of files, with wildcards and env vars expanded
        if isinstance(lh5_files, str): lh5_files = [lh5_files]
        self.lh5_files = [f for f_wc in lh5_files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]
        # Map to last row in each file
        self.file_map = np.array([self.lh5_st.read_n_rows(group, f) for f in self.lh5_files], 'int64').cumsum()
        self.group = group
        self.buffer_len = buffer_len

        self.lh5_buffer = self.lh5_st.get_buffer(self.group, self.lh5_files[0], size=self.buffer_len, field_mask=field_mask) if len(self.lh5_files)>0 else None
        self.n_rows = 0
        self.current_entry = 0

        self.field_mask = field_mask

        # List of entry indices from each file
        self.entry_list = None
        if entry_list is not None:
            entry_list = list(entry_list)
            if isinstance(entry_list[0], int):
                entry_list.sort()
                i_start = 0
                self.entry_list = []
                for f_end in self.file_map:
                    i_stop = bisect_right(entry_list, f_end, lo=i_start)
                    self.entry_list.append(entry_list[i_start:i_stop])
                    i_start = i_stop

            else:
                self.entry_list = [[]]*len(self.file_map)
                for i_file, local_list in enumerate(entry_list):
                    self.entry_list[i_file] = list(local_list)

        elif entry_mask is not None:
            # Convert entry mask into an entry list
            if isinstance(entry_mask, pd.Series):
                entry_mask = entry_mask.values
            if isinstance(entry_mask, np.ndarray):
                self.entry_list = []
                f_start = 0
                for i_file, f_end in enumerate(self.file_map):
                    self.entry_list.append(list(np.nonzero(entry_mask[f_start:f_end])[0]))
                    f_start = f_end
            else:
                self.entry_list = [[]]*len(self.file_map)
                for i_file, local_mask in enumerate(entry_mask):
                    self.entry_list[i_file] = list(np.nonzero(local_mask)[0])

        # Map to last entry of each file
        self.entry_map = self.file_map if self.entry_list is None else \
            np.array([len(elist) for elist in self.entry_list]).cumsum()


    def read(self, entry):
        """Read the next chunk of events, starting at entry. Return the
        lh5 buffer and number of rows read"""
        i_file = np.searchsorted(self.entry_map, entry, 'right')
        local_entry = entry
        if i_file>0: local_entry -= self.entry_map[i_file-1]
        self.n_rows = 0

        while(self.n_rows < self.buffer_len and i_file < len(self.file_map)):
            # Loop through files
            local_idx = self.entry_list[i_file] if self.entry_list is not None else None
            i_local = local_idx[local_entry] if local_idx is not None else local_entry
            self.lh5_buffer, n_rows = self.lh5_st.read_object(self.group, self.lh5_files[i_file], start_row = i_local, n_rows = self.buffer_len - self.n_rows, idx = local_idx, field_mask = self.field_mask, obj_buf = self.lh5_buffer, obj_buf_start = self.n_rows )

            self.n_rows += n_rows
            i_file += 1
            local_entry = 0

        self.current_entry = entry
        return (self.lh5_buffer, self.n_rows)

    def __len__(self):
        """Total number of entries"""
        return self.entry_map[-1] if len(self.entry_map)>0 else 0

    def __iter__(self):
        """Loop through entries in blocks of size buffer_len"""
        entry=0
        while entry < len(self):
            buf, n_rows = self.read(entry)
            yield (buf, entry, n_rows)
            entry += n_rows
