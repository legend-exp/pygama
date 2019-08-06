import numpy as np  
import pandas as pd
import sys
from scipy import signal
import itertools
import array
from pprint import pprint

from .data_loading import DataLoader
from .waveform import Waveform

class SIS3316File:
    """ 
    A parser file for the sis3316 able to decode header and events.
    The inputs are files produced by the llamaDAQ sis3316 readout program
    magic bytes: "LArU"
    """

    def __init__(self, file_binary, verbosity=0):
        self.f_in = file_binary
        self.verbose = verbosity
        flag = self.parse_fileheader()
        # chConf = self.parse_channelConfigs()
        
        #idx, nrx = self.__read_chunk_header()
        #print("head1: ID: {}, nr: {}".format(idx, nrx))
        #flag = True
        #while flag:
        #    fi, ci, da = self.read_next_event(chConf)
        #    if da is None:
        #        flag = False
        
    def parse_fileheader(self):
        """
        parses the fileheader of a SIS3316 file.
        The fileheader takes the first 4 32-bit words and includes:
        Magic bytes ("LArU")
        version of the llamaDAQ program producing the output file
        nr of open channels
        length in bytes of channel configurations (information for every channel, following the file header)
        returns true, if magic bytes match, false if not
        """
        self.f_in.seek(0)    #should be there anyhow, but re-set if not
        header = self.f_in.read(16)
        evt_data_32 = np.fromstring(header, dtype=np.uint32)
        evt_data_16 = np.fromstring(header, dtype=np.uint16)
        
        #line0: magic bytes
        magic = evt_data_32[0]
        #print(hex(magic))
        if magic == 0x5572414c:
            if self.verbose > 0:
                print ("Read in file as SIS3316, magic bytes correct.")
        else:
            print ("ERROR: Magic bytes not matching for SIS3316 file!")
            return False
        
        self.version_major = evt_data_16[4]
        self.version_minor = evt_data_16[3]
        self.version_patch = evt_data_16[2]
        self.length_econf = evt_data_16[5]
        self.number_chOpen = evt_data_32[3]
        
        if self.verbose > 0:
            print ("File version: {}.{}.{}".format(self.version_major, self.version_minor, self.version_patch))
            print ("{} channels open, each config {} bytes long".format(self.number_chOpen, self.length_econf))
            
        return True
       
        
    def parse_channelConfigs(self):
        """
        Reads the metadata from the beginning of the file (the "channel configuration" part, directly after the file header).
        Creates a dictionary of the metadata for each FADC/channel combination, which is returned
        
        structure of channelConfigs:
        FADCindex      channelIndex
        A ------------- x ----------- metadata for FADC A channel x
                      | y ----------- metadata for FADC A channel y
                      | z ----------- metadata for FADC A channel z
                      
        B ------------- k ----------- metadata for FADC B channel k
                      | l ----------- metadata for FADC B channel l
        ...
                      
        """
        self.f_in.seek(16)    #should be after file header anyhow, but re-set if not
        
        channelConfigs = {}
        
        for i in range(0, self.number_chOpen):
            if self.verbose > 1:
                print("reading in channel config {}".format(i))
                
            channel = self.f_in.read(68)
            ch_dpf = channel[16:32]
            evt_data_32 = np.fromstring(channel, dtype=np.uint32)
            evt_data_dpf = np.fromstring(ch_dpf, dtype=np.float64)
            
            fadcIndex = evt_data_32[0]
            channelIndex = evt_data_32[1]
            
            if fadcIndex in channelConfigs:
                #print("pre-existing fadc")
                pass
            else:
                #print("new fadc #{}".format(fadcIndex))
                channelConfigs[fadcIndex] = {}
    
            if channelIndex in channelConfigs[fadcIndex]:
                print("ERROR: Duplicate channel configuration in file: FADCID: {}, ChannelID: {}".format(fadcIndex, channelIndex))
            else:
                channelConfigs[fadcIndex][channelIndex] = {}
                
            channelConfigs[fadcIndex][channelIndex]["14BitFlag"] = evt_data_32[2] & 0x00000001
            if evt_data_32[2] & 0x00000002 == 0:
                print("WARNING: Channel in configuration marked as non-open!")
            channelConfigs[fadcIndex][channelIndex]["ADCOffset"] = evt_data_32[3]
            channelConfigs[fadcIndex][channelIndex]["SampleFreq"] = evt_data_dpf[0]     #64 bit float
            channelConfigs[fadcIndex][channelIndex]["Gain"] = evt_data_dpf[1]
            channelConfigs[fadcIndex][channelIndex]["FormatBits"] = evt_data_32[8]
            channelConfigs[fadcIndex][channelIndex]["SampleLength"] = evt_data_32[9]
            channelConfigs[fadcIndex][channelIndex]["MAWBufferLength"] = evt_data_32[10]
            channelConfigs[fadcIndex][channelIndex]["EventLength"] = evt_data_32[11]
            channelConfigs[fadcIndex][channelIndex]["EventHeaderLength"] = evt_data_32[12]
            channelConfigs[fadcIndex][channelIndex]["Accum6Offset"] = evt_data_32[13]
            channelConfigs[fadcIndex][channelIndex]["Accum2Offset"] = evt_data_32[14]
            channelConfigs[fadcIndex][channelIndex]["MAW3Offset"] = evt_data_32[15]
            channelConfigs[fadcIndex][channelIndex]["SampleOffset"] = evt_data_32[16]
            
            
        
        if self.verbose > 0:
            pprint(channelConfigs)
            
        #self.chConfigs = channelConfigs
        
        self.currentEventIndex=-1   #points to header of next chunk, not to event
        self.currentChunkIndex=0    #points to first chunk of file
            
        return channelConfigs
        
        
        
    def __read_chunk_header(self):
        """
        reads the header of the chunk
        The current file position has to point to the FADCID line of the requested chunk header
        file position afterwards is at first event of chunk
        returns FADCID, nr of events in chunk
        """
        header = self.f_in.read(8)
        if len(header) < 8:
            raise BinaryReadException(8, len(header))
        
        header_data_32 = np.fromstring(header, dtype=np.uint32)
        
        self.currentEventIndex=0   #points to first event of chunk
        
        return header_data_32[0], header_data_32[1]
    
    
    def __read_next_event(self, fadcID, channelConfigs, nrEventsPerChunk):
        """
        reads a chunk of data containing event data (event header and samples)
        file pointer has to be on beginning of event, NOT on chunk header
        Returns the channelID of the Event, as well as the chunk of data
        """
        
        if self.currentEventIndex == -1:
            print("ERROR: pointing at chunk header!")
            return None
            
        if self.verbose > 1:
            print("Reading chunk #{} event #{}".format(self.currentChunkIndex, self.currentEventIndex))
        
        position = self.f_in.tell()     #save position of the event header's 1st byte
        data1 = self.f_in.read(4)       #read the first (32 bit) word of the event's header: channelID & format bits
        if len(data1) < 4:
            raise BinaryReadException(4, len(data1))
        self.f_in.seek(position)        #go back to 1st position of event header
        
        header_data_32 = np.fromstring(data1, dtype=np.uint32)
        channelID = (header_data_32[0] >> 4) & 0x00000fff
        if self.verbose > 1:
            print("Event is from FADC #{}, channel #{}".format(fadcID, channelID))
        eventLength8 = channelConfigs[fadcID][channelID]["EventLength"]
        eventLength8 *= 4 #EventLength is in 32 bit
        
        data = self.f_in.read(eventLength8)
        if len(data) < eventLength8:
            raise BinaryReadException(eventLength8, len(data))
    
        self.currentEventIndex += 1     #move to next event
        if(self.currentEventIndex == nrEventsPerChunk): #if this event was the last of the chunk
            self.currentEventIndex = -1     #points to header of next chunk, not to event
            self.currentChunkIndex += 1     #points to next chunk of file
    
    
        return channelID, data
    
    def read_next_event(self, channelConfigs):
        """
        This should be the main method to call when parsing the file for events.
        returns the current FADC index, the channel ID and the binary data of the event if a valid event is found
        Returns -1, -1, None when after the last event in the file.
        automatically goes to the next event in the file, calling __read_chunk_header and __read_next_event
        when appropriate.
        returns FADCindex, channelIndex, binaryEventData
        """
        
        # have to extract channel index from binary, since we need the length of the event, which can change between channels
        
        if self.currentEventIndex == -1:     #points to header of next chunk, not to event
            try:
                self.currentFADC, self.currentChunkSize = self.__read_chunk_header()
            except BinaryReadException as e:
                #print("  No more data...\n")
                return -1,-1,None
                
        try:
            channelID, binary_data = self.__read_next_event(self.currentFADC, channelConfigs, self.currentChunkSize)
        except BinaryReadException as e:
            #print("  No more data...\n")
            return -1,-1,None
            
        return self.currentFADC, channelID, binary_data
        
    
        #ToDo !!!!!!!!!!!
    
    
    
class BinaryReadException(Exception):

    def __init__(self, requestedNrOfBytes, gotNrOfBytes):
        self.reqNOB = requestedNrOfBytes
        self.gotNOB = gotNrOfBytes
    
    def printMessage(self):
        print("Exception: tried to read {} bytes, got {} bytes".format(self.reqNOB, self.gotNOB))
    
    
    




