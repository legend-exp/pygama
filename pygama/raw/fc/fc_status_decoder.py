import os
import numpy as np
from ..data_decoder.py import *

class FCStatusDecoder(DataDecoder):
    """
    decode FlashCam digitizer status data.
    """
    def __init__(self, *args, **kwargs):
        """
        """
        self.decoded_values = {
            'status': { # 0: Errors occured, 1: no errors
              'dtype': 'int32',
            },
            'statustime': { # fc250 seconds, microseconds, dummy, startsec startusec
              'dtype': 'float32',
              'units': 's',
            },
            'cputime': { # CPU seconds, microseconds, dummy, startsec startusec
              'dtype': 'float64',
              'units': 's',
            },
            'startoffset': { # fc250 seconds, microseconds, dummy, startsec startusec
              'dtype': 'float32',
              'units': 's',
            },
            'cards': { # Total number of cards (number of status data to follow)
              'dtype': 'int32',
            },
            'size': { # Size of each status data
              'dtype': 'int32',
            },
            'environment': { # FC card-wise environment status
              # Array contents:
              # [0-4] Temps in mDeg
              # [5-10] Voltages in mV
              # 11 main current in mA
              # 12 humidity in o/oo
              # [13-14] Temps from adc cards in mDeg
              # FIXME: change to a table?
              'dtype': 'uint32',
              'datatype': 'array_of_equalsized_arrays<1,1>{real}',
              'length': 16,
            },
            'totalerrors': { # FC card-wise list DAQ errors during data taking
              'dtype': 'uint32',
            },
            'enverrors': {
              'dtype': 'uint32',
            },
            'ctierrors': {
              'dtype': 'uint32',
            },
            'linkerrors': {
              'dtype': 'uint32',
            },
            'othererrors': {
              'dtype': 'uint32',
              'datatype': 'array_of_equalsized_arrays<1,1>{real}',
              'length': 5,
            },
        }
        super().__init__(*args, **kwargs)


    def decode_packet(self, fcio, lh5_table, packet_id, verbosity=0):
        """
        access FC status (temp., log, ...)
        """
        # aliases for brevity
        tbl = lh5_table
        ii = tbl.loc

        # status -- 0: Errors occured, 1: no errors
        tbl['status'].nda[ii] = fcio.status

        # times
        tbl['statustime'].nda[ii] = fcio.statustime[0]+fcio.statustime[1]/1e6
        tbl['cputime'].nda[ii]    = fcio.statustime[2]+fcio.statustime[3]/1e6
        tbl['startoffset'].nda[ii]= fcio.statustime[5]+fcio.statustime[6]/1e6

        # Total number of cards (number of status data to follow)
        tbl['cards'].nda[ii] = fcio.cards

        # Size of each status data
        tbl['size'].nda[ii] = fcio.size

        # FC card-wise environment status (temp., volt., hum., ...)
        tbl['environment'].nda[ii][:] = fcio.environment

        # FC card-wise list DAQ errors during data taking
        tbl['totalerrors'].nda[ii] = fcio.totalerrors
        tbl['linkerrors'].nda[ii] = fcio.linkerrors
        tbl['ctierrors'].nda[ii] = fcio.ctierrors
        tbl['enverrors'].nda[ii] = fcio.enverrors
        tbl['othererrors'].nda[ii][:] = fcio.othererrors

        tbl.push_row()

        # sizeof(fcio_status): (3 + 10 + 256*(10 + 9 + 16 + 4 + 256))*4
        return 302132

