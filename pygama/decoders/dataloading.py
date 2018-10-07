from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from ..processing._header_parser import get_object_info

__all__ = ["get_next_event", "get_decoders"]

def get_next_event(f_in):
    """
    Gets the next event, and some basic information about it \n
    Takes the file pointer as input \n
    Outputs: \n
        event_data: a byte array of the data produced by the card (could be header + data) \n
        slot: \n
        crate: \n
        data_id: This is the identifier for the type of data-taker (i.e. Gretina4M, etc) \n
    """
    # number of bytes to read in = 8 (2x 32-bit words, 4 bytes each)

    # The read is set up to do two 32-bit integers, rather than bytes or shorts
    # This matches the bitwise arithmetic used elsewhere best, and is easy to implement
    # Using a

    # NCRATES = 10

    try:
        head = np.fromstring(f_in.read(4),dtype=np.uint32)     # event header is 8 bytes (2 longs)
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
    record_length   =int( (head[0] & 0x3FFFF))
    data_id         =int( (head[0] >> 18))
    # slot            =int( (head[1] >> 16) & 0x1f)
    # crate           =int( (head[1] >> 21) & 0xf)
    # reserved        =int( (head[1] &0xFFFF))

    # /* ========== read in the rest of the event data ========== */
    try:
        event_data = f_in.read(record_length*4-4)     # record_length is in longs, read gives bytes
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
        Looks through all the data takers that exist in this DataLoader class and see which ones exist.
        TODO: this only works if the subclasses have been imported.  is that what we want?
              also relies on 2-level abstraction, which is dicy
    """

    decoders = []

    for sub in DataLoader.__subclasses__():
        for subsub in sub.__subclasses__():
            try:
                a = subsub(object_info)
                # n = a.name
                # print("name: ",n)
                decoders.append(a)
            except Exception as e:
                print(e)
                pass

    return decoders


class DataLoader(ABC):
    def __init__(self, object_info=None):
        self.decoded_values = []

        if object_info is not None:
            self.load_object_info(object_info)

        self.hf5_type="fixed"

    def load_object_info(self, object_info):
        if isinstance(object_info, dict):
            self.object_info = get_object_info(object_info, self.class_name)
        elif isinstance(object_info, pd.core.frame.DataFrame):
            self.object_info = object_info
        elif isinstance(object_info, str):
            self.object_info = pd.read_hdf(object_info, self.class_name)
        else:
            raise TypeError("DataLoader object_info must be a dict of header values, or a string hdf5 filename.  You passed a {}".format( type(object_info)  ))

    @abstractmethod
    def decode_event(self,event_data_bytes, event_number, header_dict):
        pass

    # @abstractmethod
    # def decode_header(self):
    #     pass

    def create_df(self):
        '''
        allows us to overload for more complicated use cases
        '''
        df = pd.DataFrame.from_dict(self.decoded_values)
        if len(df) == 0: return None
        df.set_index("event_number", inplace=True)
        return df

    def to_file(self, file_name):
        df_data = self.create_df()
        if df_data is None: return
        df_data.to_hdf(file_name, key=self.decoder_name, mode='a', format=self.hf5_type, data_columns=df_data.columns.tolist())

        if self.object_info is not None:
            if self.class_name == self.decoder_name:
                raise ValueError("Class {} has the same ORCA decoder and class names: {}.  Can't write dataframe to file.".format(self.__name__, self.class_name))
            self.object_info.to_hdf(file_name, key=self.class_name, mode='a')
