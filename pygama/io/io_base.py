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
import sys
import numpy as np
import pandas as pd
from abc import ABC
import matplotlib.pyplot as plt
from pprint import pprint
from .orca_header import get_object_info
import h5py

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
            with pd.HDFStore(file_name, 'r') as store:
                try:
                    extant_df = store.get(key)
                    df_data = pd.concat([extant_df, df_data]).reset_index(drop=True)
                    if verbose:
                        print(key, df_data.shape)
                except KeyError:
                    pass
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
    def save_to_lh5(self, file_name, verbose=False, append=False):
        """
        TODO: append mode:
        https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py
        TODO: handle units with attributes?  or dimension scales (Ch8 Collette?)
        """
        file_mode = "a" if append else "w"
        hf = h5py.File(file_name, file_mode)
        
        # create the header, saving everything in attributes (like a dict)
        hf.create_group('header')
        for c in self.file_config:
            hf["/header"].attrs[c] = self.file_config[c]
        hf["/header"].attrs["file_name"] = file_name
        
        # create datasets for each member of self.decoded_values
        for col in self.decoded_values:
            
            # decompose waveforms: [dt, t0, cumulative_length[:], flattened_data[:]]
            if "waveform" in col:
                
                wf_group = f"/daqdata/{col}/"
                nwfs = len(self.decoded_values[col])
                nsamp = self.file_config['nsamples']
                wf_idxs = np.arange(0, nsamp*nwfs-1, nsamp)
                wfs = np.hstack(self.decoded_values[col])
                
                wf_t0 = hf.create_dataset(f"{wf_group}/t0", data=(1,))
                wf_dt = hf.create_dataset(f"{wf_group}/dt", data=(1,))
                
                wf_ds = hf.create_dataset(f"{wf_group}/flattened_data", 
                                          data=wfs)
                wf_idx_ds = hf.create_dataset(f"{wf_group}/cumulative_length", 
                                              data=wf_idxs)
                # exit()
                continue
            
            # handle single-valued data
            else:
                npa = np.asarray(self.decoded_values[col]) # dtype is automatic
                dset = hf.create_dataset(f"/daqdata/{col}", data=npa)
                
                # set default attributes
                dset.attrs["units"] = "none"
                dset.attrs["datatype"] = "array<1>{real}"
                
                # overwrite attributes if they exist
                if col in self.lh5_spec:
                    if "units" in self.lh5_spec[col]: 
                        dset.attrs["units"] = self.lh5_spec[col]["units"]
            
        # write stuff to the file
        hf.flush()
        hf.close()