"""
Class and methods for loading data from different `data takers`.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from ..processing.header_parser import get_object_info

__all__ = ["get_next_event", "get_decoders"]


class DataLoader(ABC):
    """
    Base class for data taking objects (digitizers or pollers).
    - Digitizers are in digitizers.py
    - Pollers are in pollers.py
    Contains base methods for saving `self.decoded_values` into pandas dataframes.

    Standard practice should be to put this line as the last line of
    decode_event:
    # send any variable with a name in "decoded_values" to the pandas output
    self.format_data(locals())
    """
    def __init__(self, object_info=None):


        # every data loader should set this (affects if we can chunk the output)
        self.h5_format = None

        # need the decoder name and the class name
        if object_info is not None:
            self.load_object_info(object_info)

    def load_object_info(self, object_info):

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
        """ Write any variable with a name matching a key in "decoded_values"
        to the pandas output.

        Standard practice should be to put this line as the last line of
        decode_event:
        # send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())
        """
        for key in vals:
            if key is not "self" and key in self.decoded_values:
                self.decoded_values[key].append(vals[key])

    def create_df(self):
        """
        Base dataframe creation method.
        Classes inheriting from DataLoader (like digitizers or pollers) can
        overload this if necessary for more complicated use cases.
        Try to avoid pickling to 'object' types if possible.
        """
        for key in self.decoded_values:
            print("      {} entries: {}".format(key, len(self.decoded_values[key])))

        # old faithful (embeds wfs in cells, requires h5type = 'fixed')
        # df = pd.DataFrame.from_dict(self.decoded_values)

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

    def to_file(self, file_name):

        df_data = self.create_df()
        if df_data is None:
            print("Data is None!")
            return
        # print(df_data.dtypes) # useful to check this

        df_data.to_hdf(
            file_name,
            key=self.decoder_name,
            mode='a',
            format=self.h5_format,
            data_columns=["event_number"]) # can use hdf5 file indexing 

            # need the decoder name and the class name
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


def get_next_event(f_in):
    """
    Gets the next event, and some basic information about it
    Takes the file pointer as input
    Outputs:
    -event_data: a byte array of the data produced by the card (could be header + data)
    -slot:
    -crate:
    -data_id: This is the identifier for the type of data-taker (i.e. Gretina4M, etc)
    """
    # number of bytes to read in = 8 (2x 32-bit words, 4 bytes each)

    # The read is set up to do two 32-bit integers, rather than bytes or shorts
    # This matches the bitwise arithmetic used elsewhere best, and is easy to implement
    # Using a

    # NCRATES = 10

    try:
        head = np.fromstring(
            f_in.read(4), dtype=np.uint32)  # event header is 8 bytes (2 longs)
    except Exception as e:
        print(e)
        raise Exception("Failed to read in the event orca header.")

    # Assuming we're getting an array of bytes:
    # record_length   = (head[0] + (head[1]<<8) + ((head[2]&0x3)<<16))
    # data_id         = (head[2] >> 2) + (head[3]<<8)
    # slot            = (head[6] & 0x1f)
    # crate           = (head[6]>>5) + head[7]&0x1
    # reserved        = (head[4] + (head[5]<<8))

    # Using an array of uint32
    record_length = int((head[0] & 0x3FFFF))
    data_id = int((head[0] >> 18))
    # slot            =int( (head[1] >> 16) & 0x1f)
    # crate           =int( (head[1] >> 21) & 0xf)
    # reserved        =int( (head[1] &0xFFFF))

    # /* ========== read in the rest of the event data ========== */
    try:
        event_data = f_in.read(record_length * 4 -
                               4)  # record_length is in longs, read gives bytes
    except Exception as e:
        print("  No more data...\n")
        print(e)
        raise EOFError

    # if (crate < 0 or crate > NCRATES or slot  < 0 or slot > 20):
    #     print("ERROR: Illegal VME crate or slot number {} {} (data ID {})".format(crate, slot,data_id))
    #     raise ValueError("Encountered an invalid value of the crate or slot number...")

    # return event_data, slot, crate, data_id
    return event_data, data_id


def get_decoders(object_info):
    """
    Find all the active pygama data takers that inherit from the DataLoader class.
    This only works if the subclasses have been imported.  Is that what we want?
    Also relies on 2-level abstraction, which is dicey
    """
    decoders = []
    for sub in DataLoader.__subclasses__():  # either digitizers or pollers
        for subsub in sub.__subclasses__():
            try:
                decoder = subsub(object_info) # initialize the decoder
                # print("dataloading - name: ",decoder.decoder_name)
                decoders.append(decoder)
            except Exception as e:
                print(e)
                pass
    return decoders
