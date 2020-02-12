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
            name = self.__class__.__name__
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


    def write_out_garbage(self, filename, group, code_attrs, lh5_store=None):
        if lh5_store is None: lh5_store = LH5Store()
        size = 0 if len(self.garbage_lensum) == 0 else self.garbage_lensum[-1]
        if size == 0: return

        # set group name according to decoder class name
        if not group.endswith('/'): group += '/'
        group += self.__class__.__name__

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



class LH5Store:
    def __init__(self, base_path='', keep_open=False):
        self.base_path = base_path
        self.keep_open = keep_open
        self.files = {}

    def gimme_file(self, lh5_file):
        if isinstance(lh5_file, h5py.File): return lh5_file
        if lh5_file in self.files.keys(): return self.files[lh5_file]
        full_path = self.base_path + '/' + lh5_file
        h5f = h5py.File(full_path, 'a')
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
        lh5_file = self.gimme_file(lh5_file)
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



class DFBuffer(pd.DataFrame)
    """A fixed-length pandas dataframe with a read/write location and attributes.

    Used in daq-to-raw data conversion, DSP, and parameter calibration as a
    buffer for IO with tables of data. Knows how to write itself out to and
    read itself in from a .lh5 file. Reads and writes are performed in blocks up
    to the specified length.
    """
    def __init__(self, size=1024, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.size = size
        self.loc = 0
        self.attrs = {}


    def is_full(self):
        return self.loc >= self.size


    def clear(self):
        self.loc = 0


    def add_field(self, field_name, field_attrs):
    if field_name in self:
        print('buffer already contains field ' + name)
        return

        # pop the dtype attribute since it's now a property of the field
        dtype = 'double' if 'dtype' not in field_attrs else field_attrs.pop('dtype')

        # handle simple scalar first 
        if 'datatype' not in field_attrs: 
            self[field_name] = np.empty(self.size, dtype=dtype)

        # handle time_series data
        elif field_attrs['datatype'] == 'time_series':
            # need to be able to compute the shape for the buffer
            if 'length' not in field_attrs:
                print('DFBuffer error: time_series must have length attribute.')
                return

            # compute the shape
            if field_attrs['length'] == 'var':
                if 'length_estimate' not in field_attrs:
                    print('DFBuffer error: var-length time_series must have length_estimate attribute.')
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
            print('DFBuffer error: unknown datatype', field_attrs['datatype'])

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
