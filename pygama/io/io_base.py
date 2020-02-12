"""
io_base.py
base class for reading raw data, usually 32-bit data words,
from different `data takers`.
contains methods to save pandas dataframes to files.
subclasses:
     - digitizers.py -- Gretina4M, SIS3302, FlashCam, etc.
     - pollers.py -- MJDPreampDecoder, ISegHVDecoder, etc.
     
Notes on writing new DataTaker objects:

I recommend you save entries to the output file by using 
`self.format_data(locals())` as the last line of `decode_event`.

DataTakers require you declare these before calling `super().__init__()`:
    * `self.digitizer_type`: a string naming the digitizer
    * `self.decoded_values`: the Python lists to convert to HDF5
"""
import sys, os
import numpy as np
import pandas as pd
from abc import ABC
import matplotlib.pyplot as plt
from pprint import pprint
import h5py

# JASON: rename to data decoder
# JASON: add "decoder_ring" for simple packets (Howe-style formatting)
class DataTaker(ABC):
    def __init__(self, user_config=None):
        """
        all DataTakers should count the total number of events, and track
        garbage values separately.  To properly initialize self.garbage_values,
        you should declare self.decoded_values in your DataTaker before calling:
            super().__init__(*args, **kwargs)
        """
        self.total_count = 0
        self.garbage_count = 0 # never leave any data behind
        self.garbage_values = {key:[] for key in self.decoded_values}

        if user_config is not None:
            with open(user_config) as f:
                self.user_config = json.load(f)


    def initialize_buffer(self, table_buffer):
        if not hasattr(self, 'decoded_vals'):
            print('Error: no decoded_values available for setting up table_buffer')
            return
        for key in self.decoded_vals.keys():
            table_buffer.add_field(key, attributes=self.decoded_vals[key])

    def initialize_dataframe(self, dataframe):
        if not hasattr(self, 'decoded_vals'):
            print('Error: no decoded_values available for setting up dataframe')
            return
        for name, attrs in self.decoded_vals.items():
            dataframe.add_field(name, attrs)

    # def set_config(self, config):
        # self.config = config


    def format_data(self, vals, is_garbage=False):
        """
        for every event in the raw DAQ data, we send any local variable with a 
        name in "self.decoded_values" to the output with the line:
            self.format_data(locals())
        """
        self.total_count += 1
        if is_garbage:
            self.garbage_count += 1

        for key in vals:
            if key is not "self" and key in self.decoded_values:
                if is_garbage:
                    self.garbage_values[key].append(vals[key])
                else:
                    if type(vals[key]) == np.ndarray:
                        self.decoded_values[key].append(vals[key].copy())
                    else:
                        self.decoded_values[key].append(vals[key])


    def clear_data(self):
        """ clear out standard objects when we do a write to file """
        self.total_count = 0
        self.garbage_count = 0
        self.decoded_values = {key:[] for key in self.decoded_values}
        self.garbage_values = {key:[] for key in self.garbage_values}


    # === PANDAS HDF5 I/O ======================================================
    def create_df(self, get_garbage=False):
        """
        Base dataframe creation method.
        Classes inheriting from DataTaker (like digitizers or pollers) can
        overload this if necessary for more complicated use cases.
        Try to avoid pickling to 'object' types if possible.
        """
        if not get_garbage:
            raw_values = self.decoded_values
        else:
            raw_values = self.garbage_values
            df = pd.DataFrame.from_dict(self.garbage_values).infer_objects()
            return df

        # main plan is to go with pytables
        pytables_error = False
        if self.h5_format == "table":

            # 'flatten' the values (each wf sample gets a column, h5type = 'table')
            new_cols = []
            for col in sorted(raw_values.keys()):

                # unzip waveforms into an ndarray
                if col == "waveform":

                    wf_lens = [len(wf) for wf in raw_values["waveform"]]
                    wf_dims = set(wf_lens)

                    # error checking
                    if len(wf_dims) > 1:
                        print("ERROR, wfs must all be same size, not ",wf_dims)
                        pytables_error = True
                        break
                    if any(d > self.pytables_col_limit for d in wf_dims):
                        print("ERROR, exceeded pytables_col_limit")
                        print(wf_dims)
                        self.h5_format = "fixed"
                        pytables_error = True
                        break
                        # sys.exit()

                    # create the ndarray and a new dataframe
                    if not pytables_error and len(raw_values["waveform"]) > 0:
                        wfs = np.vstack(raw_values["waveform"])
                        new_cols.append(pd.DataFrame(wfs, dtype='uint16'))

                # everything else is single-valued
                else:
                    if not pytables_error:
                        vals = pd.Series(raw_values[col], name=col)
                        new_cols.append(vals)

            # this is our flattened output (can be chunked with hdf5)
            df = pd.concat(new_cols, axis=1)


        # the backup plan
        if self.h5_format == "fixed" or pytables_error:
            df = pd.DataFrame.from_dict(raw_values)

        if len(df) == 0:
            return None
        else:
            return df


    def save_to_pytables(self, file_name, verbose=False, write_option="a"):
        """ 
        save primary data, garbage data, and metadata to hdf5 
        """
        if verbose:
            print("Writing {}".format(self.decoder_name))

        # we can only fast-append to the file if it's in table format
        append = True if self.h5_format == "table" else False

        hdf_kwargs = {"mode":write_option,
                      "format":self.h5_format,
                      "append":append,
                      "complib":"blosc:snappy", # idk, sounds fast
                      "complevel":2, # compresses raw by ~0.5
                      "data_columns":["ievt"]} # cols for hdf5 fast file indexing

        def check_and_append(file_name, key, df_data):
            try:
                with pd.HDFStore(file_name, 'r') as store:
                    try:
                        extant_df = store.get(key)
                        df_data = pd.concat([extant_df, df_data]).reset_index(drop=True)
                        if verbose:
                            print(key, df_data.shape)
                    except KeyError:
                        pass
            except IOError:
               print("sees that the file is not yet open, which is normal for 1st call??")
            return df_data

        #  ------------- save primary data -------------
        df_data = self.create_df()

        if not append:
            # make a copy of the df already in the file and manually append
            df_data = check_and_append(file_name, self.decoder_name, df_data)

        # write to hdf5 file
        df_data.to_hdf(file_name, key=self.decoder_name, **hdf_kwargs)

        # ------------- save metadata -------------
        if self.df_metadata is not None:

            if self.class_name == self.decoder_name:
                raise ValueError(
                    "Can't write df_metadata if it has the same "
                    "ORCA decoder and class names, you would overwrite the data."
                    "Class: {}\nDecoder: {}".format(self.class_name,
                                                    self.decoder_name))
                sys.exit()

            if not append:
                self.df_metadata = check_and_append(file_name,
                                                    self.class_name,
                                                    self.df_metadata).infer_objects()

            self.df_metadata.to_hdf(file_name, key=self.class_name,
                                    mode='a',
                                    format="fixed")

        # ------------- save garbage data -------------
        if self.garbage_count > 0:
            print("Saving garbage: {} of {} total events"
                  .format(self.garbage_count, self.total_count))

            df_garbage = self.create_df(get_garbage=True)

            # garbage is always in fixed format since the wfs may be different lengths
            hdf_kwargs["format"] = "fixed"
            hdf_kwargs["append"] = False

            df_garbage = check_and_append(file_name,
                                          self.decoder_name+"_Garbage",
                                          df_garbage)#.infer_objects()

            df_garbage.to_hdf(file_name, key=self.decoder_name+"_Garbage",
                              **hdf_kwargs)

        # finally, clear out existing data (relieve memory pressure)
        self.clear_data()


    # === LH5 HDF5 I/O =========================================================
    def save_to_lh5(self, file_name):
        """
        """
        append = os.path.exists(file_name)
        file_mode = "a" if append else "w"
        
        # open the output file
        hf = h5py.File(file_name, file_mode)
        
        if not append:
            
            # create the header, saving everything in attributes (like a dict)
            hf.create_group('header')
            for c in self.file_config:
                #print(c, self.file_config[c])  #test
                hf["/header"].attrs[c] = self.file_config[c]
            hf["/header"].attrs["file_name"] = file_name
            
            # create the main "daqdata" group and put the table def in the attrs
            hf.create_group('daqdata')
            cols = [col for col in self.decoded_values]
            table_def = "table{" + ",".join(cols) + "}"
            hf["/daqdata"].attrs["datatype"] = table_def
            
        # create datasets for each member of self.decoded_values
        for col in self.decoded_values:
            
            # create waveform datasets
            if "waveform" in col:

                # set the overall group name
                wf_group = f"/daqdata/{col}/"

                # read out the waveforms & convert to numpy arrays
                nwfs = len(self.decoded_values[col])
                nsamp = self.file_config['nsamples']
                wf_idxs = np.arange(0, nsamp*nwfs-1, nsamp)
                wfs = np.hstack(self.decoded_values[col])
                
                # TODO: get actual values
                # maybe this should just be in decoded_values instead
                # of the waveform group.  mention it to Oliver
                wf_t0 = np.ones(len(wf_idxs)) 
                wf_dt = np.ones(len(wf_idxs)) * 1e-8

                # TODO: apply compression here, and wf_idxs will vary

                # write first time, declare all groups & attributes
                if not append:
                    
                    # declare the sub-table for the waveform
                    hf.create_group(wf_group)
                    hf[wf_group].attrs["datatype"] = "table{t0,dt,values}"
                    
                    # t0
                    st_t0 = f"{wf_group}/t0"
                    ds_dt = hf.create_dataset(st_t0, data=wf_t0, maxshape=(None,))
                    hf[st_t0].attrs["datatype"] = "array<1>{real}"
                    hf[st_t0].attrs["units"] = "ns"
                    
                    # dt
                    st_dt = f"{wf_group}/dt"
                    ds_dt = hf.create_dataset(st_dt, data=wf_dt, maxshape=(None,))
                    hf[st_dt].attrs["datatype"] = "array<1>{real}"
                    hf[st_dt].attrs["units"] = "ns"
                    
                    # waveform (contained as a paired subgroup)
                    gr_wf = f"{wf_group}/values"
                    hf.create_group(gr_wf)
                    hf[gr_wf].attrs["datatype"] = "array<1>{array<1>{real}}"
                    
                    st_cl = f"{wf_group}/values/cumulative_length"
                    wf_cl = hf.create_dataset(st_cl, data=wf_idxs, maxshape=(None,))
                    hf[st_cl].attrs["datatype"] = "array<1>{real}"
                    
                    st_fl = f"{wf_group}/values/flattened_data"
                    wf_fl = hf.create_dataset(st_fl, data=wfs, maxshape=(None,))
                    hf[st_fl].attrs["datatype"] = "array<1>{real}"
                
                # append
                else:
                    print("appending ...")
                    
                    ds_cl = hf[f"{wf_group}/values/cumulative_length"]
                    ds_cl.resize(ds_cl.shape[0] + wf_idxs.shape[0], axis=0)
                    ds_cl[-wf_idxs.shape[0]:] = wf_idxs
                    
                    ds_fl = hf[f"{wf_group}/values/flattened_data"]
                    ds_fl.resize(ds_fl.shape[0] + wfs.shape[0], axis=0)   
                    ds_fl[-wfs.shape[0]:] = wfs
                    
                    ds_t0 = hf[f"{wf_group}/t0"]
                    ds_t0.resize(ds_t0.shape[0] + wf_t0.shape[0], axis=0)
                    ds_t0[-wf_t0.shape[0]:] = wf_t0
                    
                    ds_dt = hf[f"{wf_group}/dt"]
                    ds_dt.resize(ds_dt.shape[0] + wf_dt.shape[0], axis=0)
                    ds_dt[-wf_dt.shape[0]:] = wf_dt
                    
                    
            # create single-valued datasets
            else:
                
                npa = np.asarray(self.decoded_values[col]) 
                
                # write first time
                if not append:
                    dset = hf.create_dataset(f"/daqdata/{col}", data=npa, maxshape=(None,))
                    # print("first one:", npa.shape[0], col)
                
                    # set default attributes
                    dset.attrs["units"] = "none"
                    dset.attrs["datatype"] = "array<1>{real}"
                
                    # overwrite attributes if they exist
                    if col in self.lh5_spec:
                        if "units" in self.lh5_spec[col]: 
                            dset.attrs["units"] = self.lh5_spec[col]["units"]
                
                # append
                else:
                    dset = hf[f"/daqdata/{col}"]
                    dset.resize(dset.shape[0] + npa.shape[0], axis=0)
                    dset[-npa.shape[0]:] = npa
            
        # write stuff to the file
        hf.flush()
        hf.close()
        
        # finally, clear out existing data (relieve memory pressure)
        self.clear_data()
        

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

        # originally was in gimme_dataset but need to create from nda_data the
        # first time for speed
        if not isinstance(ds, h5py.Dataset):
            if ds not in group:
                # Original version of the next line:
                # dataset = group.create_dataset(ds, shape=(0,), dtype=dtype, maxshape=(None,))
                # This works but is super slow. For some reason, dataset
                # appending is way faster if it's given an appropriately sized
                # array the first time.
                ds = group.create_dataset(ds, data=nda_data, maxshape=(None,))
                if data_attrs is not None: ds.attrs.update(data_attrs)
                return
            else: ds = group[ds]
        if data_attrs is not None: ds.attrs.update(data_attrs)

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
                if not group.endswith('/'): group = group + '/'
                group = group + field
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
