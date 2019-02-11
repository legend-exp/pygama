import numpy as np
import pandas as pd
import sys

from .data_loading import DataLoader


class Poller(DataLoader):
    """ handle any data taker that does not record waveforms """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_settings(self, settings):
        """ apply user settings specific to this card and run """
        pass


class MJDPreampDecoder(Poller):
    """
    members:
    - decode_event (decodes raw data from an MJDPreAmp object)
    - format_data (returns a dict)
    - get_detectors_for_preamp (returns a dict)
    """

    def __init__(self, *args, **kwargs):

        self.decoder_name = 'ORMJDPreAmpDecoderForAdc'
        self.class_name = 'MJDPreAmp'

        # store an entry for every event -- this is what we convert to pandas
        self.decoded_values = {
            "adc": [],
            "enabled": [],
            "timestamp": [],
            "device_id": [],
            "event_number": []
        }
        super().__init__(*args, **kwargs) # also initializes the garbage df

        self.event_header_length = -1
        self.h5_format = 'fixed'


    def decode_event(self,
                     event_data_bytes,
                     event_number,
                     header_dict,
                     verbose=False):
        """ see README for the 32-bit data word diagram """

        event_data_uint = np.fromstring(event_data_bytes, dtype=np.uint32)
        event_data_float = np.fromstring(event_data_bytes, dtype=np.float32)

        device_id = (event_data_uint[0] & 0xFFF)
        timestamp = event_data_uint[1]
        enabled = np.zeros(16)
        adc = np.zeros(16)

        try:
            detector_names = self.get_detectors_for_preamp(
                header_dict, device_id)
        except KeyError:
            return None

        for i, val in enumerate(enabled):
            enabled[i] = (event_data_uint[2] >> (i) & 0x1)

            if (verbose):
                if (enabled[i] != 0):
                    print("Channel %d is enabled" % (i))
                else:
                    print("Channel %d is disabled" % (i))

        for i, val in enumerate(adc):
            adc[i] = event_data_float[3 + i]

        if (verbose):
            print(adc)

        # send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())

    def get_detectors_for_preamp(self, header_dict, an_ID):
        """
        Returns a dictionary that goes:
        dict[preamp_id] = ["detectorName1", "detectorName2"...
        e.g: d[5] = ['P12345A','P12346B','', ... 'B8765']
        """

        for preampNum in header_dict["ObjectInfo"]["AuxHw"]:

            preamp_ID = preampNum["MJDPreAmp"]["preampID"]
            if (preamp_ID == an_ID):
                channel_names = [
                    preampNum["MJDPreAmp"]["detectorName0"],
                    preampNum["MJDPreAmp"]["detectorName1"],
                    preampNum["MJDPreAmp"]["detectorName2"],
                    preampNum["MJDPreAmp"]["detectorName3"],
                    preampNum["MJDPreAmp"]["detectorName4"],
                    "+12V",  #preampNum["MJDPreAmp"]["detectorName5"],
                    "-12V",  #preampNum["MJDPreAmp"]["detectorName6"],
                    "Temp Chip 1",  #preampNum["MJDPreAmp"]["detectorName7"],
                    preampNum["MJDPreAmp"]["detectorName8"],
                    preampNum["MJDPreAmp"]["detectorName9"],
                    preampNum["MJDPreAmp"]["detectorName10"],
                    preampNum["MJDPreAmp"]["detectorName11"],
                    preampNum["MJDPreAmp"]["detectorName12"],
                    "+24V",  #preampNum["MJDPreAmp"]["detectorName13"],
                    "-24V",  #preampNum["MJDPreAmp"]["detectorName14"],
                    "Temp Chip 2"  #preampNum["MJDPreAmp"]["detectorName15"]
                ]
        return channel_names


class ISegHVDecoder(Poller):
    """ iSeg HV Card """

    def __init__(self, *args, **kwargs):

        self.decoder_name = 'ORiSegHVCardDecoderForHV'
        self.class_name = 'ORiSegHVCardDecoder'

        # store an entry for every event -- this is what we convert to pandas
        self.decoded_values = {
            "timestamp": [],
            "voltage": [],
            "current": [],
            "enabled": [],
            "crate": [],
            "card": [],
            "event_number": []
        }
        super().__init__(*args, **kwargs) # also initializes the garbage df

        self.event_header_length = -1
        self.h5_format = 'fixed'


    def decode_event(self,
                     event_data_bytes,
                     event_number,
                     header_dict,
                     verbose=False):
        """ see README for the 32-bit data word diagram """
        event_data_int = np.fromstring(event_data_bytes, dtype=np.uint32)
        event_data_float = np.fromstring(event_data_bytes, dtype=np.float32)

        crate = (event_data_int[0] >> 20) & 0xF
        card = (event_data_int[0] >> 16) & 0xF

        enabled = np.zeros(8)  #enabled channels
        voltage = np.zeros(8)
        current = np.zeros(8)
        timestamp = event_data_int[3]

        for i, j in enumerate(enabled):
            enabled[i] = (event_data_int[1] >> (4 * i) & 0xF)

            if (verbose):
                if (enabled[i] != 0):
                    print("Channel %d is enabled" % (i))
                else:
                    print("Channel %d is disabled" % (i))

        for i, j in enumerate(voltage):
            voltage[i] = event_data_float[4 + (2 * i)]
            current[i] = event_data_float[5 + (2 * i)]

        if (verbose):
            print("HV voltages: ", voltage)
            print("HV currents: ", current)

        # send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())
