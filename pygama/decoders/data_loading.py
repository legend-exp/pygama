"""
data_loading.py
base class for reading raw data, usually 32-bit data words,
from different `data takers`.
contains methods to save pandas dataframes to files.
subclasses:
     - digitizers.py -- Gretina4M, SIS3302, etc.
     - pollers.py -- MJDPreampDecoder, ISegHVDecoder, etc.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from .xml_parser import get_object_info


class DataLoader(ABC):
    """
    NOTE:
    all subclasses should save entries into pandas dataframes
    by putting this as the last line of their decode_event:
        self.format_data(locals())
    This sends any variable whose name is a key in
    `self.decoded_values` to the output (create_df) function.
    """
    def __init__(self, object_info=None):

        # every DataLoader should write to this
        self.decoded_values = {}

        # every data loader should set this (affects if we can chunk the output)
        self.h5_format = None

        # need the decoder name and the class name
        if object_info is not None:
            self.load_object_info(object_info)

    def load_object_info(self, object_info):
        """ Load metadata for this data taker """

        if isinstance(object_info, dict):
            self.object_info = get_object_info(object_info, self.class_name)

        elif isinstance(object_info, pd.core.frame.DataFrame):
            self.object_info = object_info

        elif isinstance(object_info, str):
            self.object_info = pd.read_hdf(object_info, self.class_name)

        else:
            raise TypeError(
                "DataLoader object_info must be a dict of header values, or a string hdf5 filename.  You passed a {}"
                .format(type(object_info)))

    def format_data(self, vals):
        """
        send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())
        """
        for key in vals:
            if key is not "self" and key in self.decoded_values:
                self.decoded_values[key].append(vals[key])

    def create_df(self, flatten):
        """
        Base dataframe creation method.
        Classes inheriting from DataLoader (like digitizers or pollers) can
        overload this if necessary for more complicated use cases.
        Try to avoid pickling to 'object' types if possible.
        """
        for key in self.decoded_values:
            print("      {} entries: {}".format(key, len(self.decoded_values[key])))

        if not flatten:
            # old faithful (embeds wfs in cells, requires h5type = 'fixed')
            df = pd.DataFrame.from_dict(self.decoded_values)

        else:
            # 'flatten' the values (each wf sample gets a column, h5type = 'table')
            new_cols = []
            for col in sorted(self.decoded_values.keys()):

                # unzip waveforms (needed to use "table" hdf5 output)
                if col == "waveform":
                    wfs = np.vstack(self.decoded_values[col]) # creates an ndarray
                    new_cols.append(pd.DataFrame(wfs, dtype='int16'))

                # everything else is single-valued
                else:
                    vals = pd.Series(self.decoded_values[col], name=col)
                    new_cols.append(vals)

            # this is our flattened output (can be chunked with hdf5)
            df = pd.concat(new_cols, axis=1)

        if len(df) == 0:
            print("Length of DataFrame for {} is 0!".format(self.class_name))
            return None
        # df.set_index("event_number", inplace=True)

        return df

    def to_file(self, file_name, flatten=False):

        # save primary data
        df_data = self.create_df(flatten)
        if df_data is None:
            print("Data is None!")
            return
        # print(df_data.dtypes) # useful to check this

        df_data.to_hdf(
            file_name,
            key=self.decoder_name,
            mode='a',
            format=self.h5_format,
            data_columns=["event_number"]) # can use for hdf5 file indexing

        # save metadata
        if self.object_info is not None:
            # print(self.object_info.dtypes)

            if self.class_name == self.decoder_name:
                raise ValueError(
                    "Class {} has the same ORCA decoder and class names: {}.  Can't write dataframe to file."
                    .format(self.__name__, self.class_name))

            # used fixed type for this (keys have lotsa diff types and i
            # don't want to convert everything to strings)
            self.object_info.to_hdf(file_name, key=self.class_name,
                                    mode='a',
                                    format="fixed")


