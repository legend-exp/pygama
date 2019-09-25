"""
data_loading.py
base class for reading raw data, usually 32-bit data words,
from different `data takers`.
contains methods to save pandas dataframes to files.
subclasses:
     - digitizers.py -- Gretina4M, SIS3302, FlashCam, etc.
     - pollers.py -- MJDPreampDecoder, ISegHVDecoder, etc.
"""
import sys
import numpy as np
import pandas as pd
from abc import ABC
import matplotlib.pyplot as plt
from pprint import pprint
from .xml_parser import get_object_info


def get_decoders(object_info=None):
    """ Find all the active pygama data takers that inherit from DataLoader.
    This only works if the subclasses have been imported.
    """
    decoders = []
    for sub in DataLoader.__subclasses__():
        for subsub in sub.__subclasses__():
            try:
                decoder = subsub(object_info) # initialize the decoder
                decoders.append(decoder)
            except Exception as e:
                print(e)
                pass
    return decoders


class DataLoader(ABC):
    """
    NOTE:
    all subclasses should save entries into pandas dataframes
    by putting this as the last line of their decode_event:
        self.format_data(locals())
    This sends any variable whose name is a key in
    `self.decoded_values` to the output (create_df) function.
    """
    def __init__(self, df_metadata=None):
        """
        when you initialize this from a derived class with 'super',
        you should have already declared:
        self.decoder_name, self.class_name, and self.decoded_values
        """
        self.total_count = 0
        self.garbage_count = 0 # never leave any data behind
        self.garbage_values = {key:[] for key in self.decoded_values}

        # every DataLoader should set this (affects if we can chunk the output)
        self.h5_format = "table"
        self.pytables_col_limit = 3100

        # need the decoder name and the class name
        if df_metadata is not None:
            self.load_metadata(df_metadata)
        else:
            self.df_metadata = None


    def load_metadata(self, df_metadata):
        """ Load metadata for this data taker """

        # print('trying this', self.class_name)
        # pprint(df_metadata)

        if isinstance(df_metadata, dict):
            self.df_metadata = get_object_info(df_metadata, self.class_name)

        elif isinstance(df_metadata, pd.core.frame.DataFrame):
            self.df_metadata = df_metadata

        elif isinstance(df_metadata, str):
            self.df_metadata = pd.read_hdf(df_metadata, self.class_name)

        else:
            raise TypeError(
                "Wrong DataLoader metadata type:"
                .format(type(df_metadata)))


    def format_data(self, vals, is_garbage=False):
        """
        for every single event in the raw DAQ data, we send
        any variable with a name in "self.decoded_values"
        to the pandas output with:
        self.format_data(locals())
        """
        self.total_count += 1

        if is_garbage:
            self.garbage_count += 1

        for key in vals:
            if key is not "self" and key in self.decoded_values:

                # print(key, vals[key], type(vals[key]))
                # if isinstance(vals[key], np.ndarray):
                    # print(len(vals[key]))

                if is_garbage:
                    self.garbage_values[key].append(vals[key])
                else:
                    # Need to make this fix, otherwise waveform array overwritten at each event
                    if type(vals[key])==np.ndarray:
                        self.decoded_values[key].append(vals[key].copy())
                    else:
                        self.decoded_values[key].append(vals[key])

    def create_df(self, get_garbage=False):
        """
        Base dataframe creation method.
        Classes inheriting from DataLoader (like digitizers or pollers) can
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


    def to_file(self, file_name, verbose=False, write_option="a"):
        """ save primary data, garbage data, and metadata to hdf5 """

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


    def clear_data(self):
        """ clear out standard objects when we do a write to file """
        self.total_count = 0
        self.garbage_count = 0
        self.decoded_values = {key:[] for key in self.decoded_values}
        self.garbage_values = {key:[] for key in self.garbage_values}

