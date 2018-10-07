import numpy as np
import pandas as pd
import sys

from .dataloading import DataLoader

__all__ = ['MJDPreampDecoder', 'ISegHVDecoder']

class Poller(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode_event(self,event_data_bytes, event_number, header_dict):
        pass


# Polled devices

class MJDPreampDecoder(Poller):
    def __init__(self, *args, **kwargs):
        self.decoder_name = 'ORMJDPreAmpDecoderForAdc' #
        self.class_name = 'MJDPreAmp'

        super().__init__(*args, **kwargs)
        self.event_header_length = -1

        return

    def decode_event(self, event_data_bytes, event_number, header_dict, verbose=False):
        """
            Decodes the data from a MJDPreamp Object.
            Returns:
                adc_val     : A list of floating point voltage values for each channel
                timestamp   : An integer unix timestamp
                enabled     : A list of 0 or 1 values indicating which channels are enabled

            Data Format:
            0 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
                                       ^^^^ ^^^^ ^^^^- device id
            1 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  unix time of measurement
            2 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  enabled adc mask
            3 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  adc chan 0 encoded as a float
            4 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  adc chan 1 encoded as a float
            ....
            ....
            18 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  adc chan 15 encoded as a float

        """

        event_data_uint = np.fromstring(event_data_bytes,dtype=np.uint32)
        event_data_float = np.fromstring(event_data_bytes,dtype=np.float32)

        device_id = (event_data_uint[0]&0xFFF)
        timestamp = event_data_uint[1]
        enabled = np.zeros(16)
        adc_val = np.zeros(16)

        try:
            detector_names = self.get_detectors_for_preamp(header_dict,device_id)
        except KeyError:
            return None

        for i,val in enumerate(enabled):
            enabled[i] = (event_data_uint[2]>>(i) & 0x1)

            if(verbose):
                if(enabled[i] != 0):
                    print("Channel %d is enabled" % (i))
                else:
                    print("Channel %d is disabled" % (i))

        for i,val in enumerate(adc_val):
            adc_val[i] = event_data_float[3+i]

        if(verbose):
            print(adc_val)

        data_dict = self.format_data(adc_val, timestamp, enabled, device_id, detector_names, event_number)
        for d in data_dict:
            self.decoded_values.append(d)

        return data_dict

    def format_data(self, adc_val, timestamp, enabled, device_id, detector_names, event_number):
        """
        Format the values that we get from this card into a pandas-friendly format.
        """
        data = []

        # print(detector_names)

        for i,enabled_val in enumerate(enabled):
            d = {
                "adc" : adc_val[i-1],
                "enabled" : enabled_val,
                "timestamp" : timestamp,
                "name" : detector_names[i-1],
                "channel" : i-1,
                "device_id" : device_id,
                "event_number" : event_number
            }
            data.append(d)
        return data

    def get_detectors_for_preamp(self,header_dict,an_ID):
        """
            Returns a dictionary that goes:
                dict[preamp_id] = ["detectorName1", "detectorName2"...

                e.g: d[5] = ['P12345A','P12346B','', ... 'B8765']
        """

        for preampNum in header_dict["ObjectInfo"]["AuxHw"]:

            preamp_ID = preampNum["MJDPreAmp"]["preampID"]
            if(preamp_ID == an_ID):
                # channel_names =
                #     "0" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName0"],
                #     "1" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName1"],
                #     "2" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName2"],
                #     "3" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName3"],
                #     "4" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName4"],
                #     "5" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName5"],
                #     "6" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName6"],
                #     "7" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName7"],
                #     "8" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName8"],
                #     "9" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName9"],
                #     "10" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName10"],
                #     "11" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName11"],
                #     "12" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName12"],
                #     "13" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName13"],
                #     "14" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName14"],
                #     "15" : headerDict["ObjectInfo"]["AuxHw"][preampNum]["MJDPreAmp"]["detectorName15"],
                # }

                channel_names = [
                    preampNum["MJDPreAmp"]["detectorName0"],
                    preampNum["MJDPreAmp"]["detectorName1"],
                    preampNum["MJDPreAmp"]["detectorName2"],
                    preampNum["MJDPreAmp"]["detectorName3"],
                    preampNum["MJDPreAmp"]["detectorName4"],
                    "+12V",#preampNum["MJDPreAmp"]["detectorName5"],
                    "-12V",#preampNum["MJDPreAmp"]["detectorName6"],
                    "Temp Chip 1",#preampNum["MJDPreAmp"]["detectorName7"],
                    preampNum["MJDPreAmp"]["detectorName8"],
                    preampNum["MJDPreAmp"]["detectorName9"],
                    preampNum["MJDPreAmp"]["detectorName10"],
                    preampNum["MJDPreAmp"]["detectorName11"],
                    preampNum["MJDPreAmp"]["detectorName12"],
                    "+24V",#preampNum["MJDPreAmp"]["detectorName13"],
                    "-24V",#preampNum["MJDPreAmp"]["detectorName14"],
                    "Temp Chip 2"#preampNum["MJDPreAmp"]["detectorName15"]
                ]
        return channel_names

class ISegHVDecoder(Poller):
    def __init__(self, *args, **kwargs):
        self.decoder_name = 'ORiSegHVCardDecoderForHV'
        self.class_name = 'ORiSegHVCard_placeholder' #what should this be?

        super().__init__(*args, **kwargs)
        self.event_header_length = -1


        return

    def decode_event(self,event_data_bytes, event_number, header_dict, verbose=False):
        """
            Decodes an iSeg HV Card event

            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
            ^^^^ ^^^^ ^^^^ ^^----------------------- Data ID (from header)
            -----------------^^ ^^^^ ^^^^ ^^^^ ^^^^- length
        0   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
            ----------^^^^-------------------------- Crate number
            ---------------^^^^--------------------- Card number

        1    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx -ON Mask
        2    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx -Spare
        3    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  time in seconds since Jan 1, 1970
        4    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Voltage encoded as a float (chan 0)
        5    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Current encoded as a float (chan 0)
        6    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Voltage encoded as a float (chan 1)
        7    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Current encoded as a float (chan 1)
        8    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Voltage encoded as a float (chan 2)
        9    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Current encoded as a float (chan 2)
        10   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Voltage encoded as a float (chan 3)
        11   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Current encoded as a float (chan 3)
        12   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Voltage encoded as a float (chan 4)
        13   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Current encoded as a float (chan 4)
        14   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Voltage encoded as a float (chan 5)
        15   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Current encoded as a float (chan 5)
        16   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Voltage encoded as a float (chan 6)
        17   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Current encoded as a float (chan 6)
        18   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Voltage encoded as a float (chan 7)
        19   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  actual Current encoded as a float (chan 7)
        """

        event_data_int = np.fromstring(event_data_bytes,dtype=np.uint32)
        event_data_float = np.fromstring(event_data_bytes,dtype=np.float32)

        # print(event_data_int)
        # print(event_data_float)

        crate   = (event_data_int[0]>>20)&0xF
        card    = (event_data_int[0]>>16)&0xF


        enabled = np.zeros(8)     #enabled channels
        voltage = np.zeros(8)
        current = np.zeros(8)
        timestamp = event_data_int[3]

        for i,j in enumerate(enabled):
            enabled[i] = (event_data_int[1]>>(4*i) & 0xF)

            if(verbose):
                if(enabled[i] != 0):
                    print("Channel %d is enabled" % (i))
                else:
                    print("Channel %d is disabled" % (i))

        for i,j in enumerate(voltage):
            voltage[i] = event_data_float[4+(2*i)]
            current[i] = event_data_float[5+(2*i)]

        if(verbose):
            print("HV voltages: ",voltage)
            print("HV currents: ",current)

        # self.values["channel"] = channel

        data_dict = self.format_data(timestamp, voltage, current, enabled, crate, card, event_number)
        self.decoded_values.append(data_dict)

        return data_dict

    def format_data(self, timestamp, voltage, current, enabled, crate, card, event_number):
        """
        Format the values that we get from this card into a pandas-friendly format.
        """

        data = {
            "timestamp" : timestamp,
            "voltage" : voltage,
            "current" : current,
            "enabled" : enabled,
            "crate" : crate,
            "card" : card,
            "event_number" : event_number
        }

        return data
