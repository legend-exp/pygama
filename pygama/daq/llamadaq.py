import sys
import numpy as np  
import pandas as pd
from pprint import pprint

from .io_base import DataTaker
#from .waveform import Waveform

class llama_3316:
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
 
        if header_data_32[1] == 0:
            if self.verbose > 1:
                print("Warning: having a chunk with 0 events")
        
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
            self.currentChunkSize = 0
            try:
                while self.currentChunkSize == 0:   #apparently needed, as there can be 0-size chunks in the file
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
        
    
    
class BinaryReadException(Exception):

    def __init__(self, requestedNrOfBytes, gotNrOfBytes):
        self.reqNOB = requestedNrOfBytes
        self.gotNOB = gotNrOfBytes
    
    def printMessage(self):
        print("Exception: tried to read {} bytes, got {} bytes".format(self.reqNOB, self.gotNOB))
    


class LLAMAStruck3316(DataTaker):
    """ 
    decode Struck 3316 digitizer data
    
    TODO:
    handle per-channel data (gain, ...)
    most metadata of Struck header (energy, ...)
    """
    def __init__(self, metadata=None, *args, **kwargs):
        self.decoder_name = 'SIS3316Decoder'
        self.class_name = 'SIS3316'

        # store an entry for every event
        self.decoded_values = {
            "packet_id": [],
            "ievt": [],
            "energy_first": [],
            "energy": [],
            "timestamp": [],
            "peakhigh_index": [],
            "peakhigh_value": [],
            "information": [],
            "accumulator1": [],
            "accumulator2": [],
            "accumulator3": [],
            "accumulator4": [],
            "accumulator5": [],
            "accumulator6": [],
            "accumulator7": [],
            "accumulator8": [],
            "mawMax": [],
            "maw_before": [],
            "maw_after": [],
            "fadcID": [],
            "channel": [],
            "waveform": [],
        }

        self.config_names = []  #TODO at some point we want the metainfo here
        self.file_config = {}
        if metadata is not None: 
            self.file_config = self.readMetadata(metadata)
            print("We have {} adcs and {} samples per WF.".format(self.file_config["nadcs"],self.file_config["nsamples"]))

        super().__init__(*args, **kwargs) # also initializes the garbage df (whatever that means...)

        # self.event_header_length = 1 #?
        self.sample_period = 0  # ns, I will set this later, according to header info
        self.gain = 0           
        self.h5_format = "table"	#was table
        #self.n_blsamp = 2000
        self.ievt = 0       #event number
        self.ievt_gbg = 0      #garbage event number
        self.window = False
        self.df_metadata = metadata #seems that was passed to superclass before, try now like this
        self.pytables_col_limit = 3000

    def readMetadata(self, meta):
        nsamples = -1
        totChan = 0
        configs = {}
        adcOff = {}
        for fadc in meta:
            adcOff[fadc] = {}
            for channel in meta[fadc]:
                if nsamples == -1:
                    # FIXME everything is fixed to 1st existing channel.
                    nsamples = meta[fadc][channel]["SampleLength"]
                    configs["14BitFlag"] = meta[fadc][channel]["14BitFlag"]
                    #configs["ADCOffset"] = meta[fadc][channel]["ADCOffset"]
                    configs["FormatBits"] = meta[fadc][channel]["FormatBits"]
                    configs["Gain"] = meta[fadc][channel]["Gain"]
                    configs["SampleFreq"] = meta[fadc][channel]["SampleFreq"]
                    configs["SampleOffset"] = meta[fadc][channel]["SampleOffset"]
                    adcOff[fadc][channel] = meta[fadc][channel]["ADCOffset"]
                elif nsamples != meta[fadc][channel]["SampleLength"]:
                    print("samples not uniform!!!")
                totChan += 1
        configs["nadcs"] = totChan
        configs["nsamples"] = nsamples
        return configs
        
    def initialize(self, sample_period, gain):
        """
        sets certain global values from a run, like:
        sample_period: time difference btw 2 samples in ns
        gain: multiply the integer sample value with the gain to get the voltage in V
        Method has to be called before the actual decoding work starts !
        """
        self.sample_period = sample_period
        self.gain = gain
        
        
    def decode_event(self, event_data_bytes, packet_id, header_dict, fadcIndex, 
                     channelIndex, verbose=False):
        """
        see the llamaDAQ documentation for data word diagrams
        """
        
        if self.sample_period == 0:
            print("ERROR: Sample period not set; use initialize() before using decode_event() on SIS3316Decoder")
            raise Exception ("Sample period not set")
        
        # parse the raw event data into numpy arrays of 16 and 32 bit ints
        evt_data_32 = np.fromstring(event_data_bytes, dtype=np.uint32)
        evt_data_16 = np.fromstring(event_data_bytes, dtype=np.uint16)
        
        # e sti gran binaries non ce li metti
        timestamp = ((evt_data_32[0] & 0xffff0000) << 16) + evt_data_32[1]
        format_bits = (evt_data_32[0]) & 0x0000000f
        offset = 2
        if format_bits & 0x1:
            peakhigh_value = evt_data_16[4]
            peakhigh_index = evt_data_16[5]
            information = (evt_data_32[offset+1] >> 24) & 0xff
            accumulator1 = evt_data_32[offset+2]
            accumulator2 = evt_data_32[offset+3]
            accumulator3 = evt_data_32[offset+4]
            accumulator4 = evt_data_32[offset+5]
            accumulator5 = evt_data_32[offset+6]
            accumulator6 = evt_data_32[offset+7]
            offset += 7
        else:
            peakhigh_value = 0
            peakhigh_index = 0  
            information = 0
            accumulator1 = accumulator2 = accumulator3 = accumulator4 = accumulator5 = accumulator6 = 0
            pass
        if format_bits & 0x2:
            accumulator7 = evt_data_32[offset+0]
            accumulator8 = evt_data_32[offset+1]
            offset += 2
        else:
            accumulator7 = accumulator8 = 0
            pass
        if format_bits & 0x4:
            mawMax = evt_data_32[offset+0]
            maw_before = evt_data_32[offset+1]
            maw_after = evt_data_32[offset+2]
            offset += 3
        else:
            mawMax = maw_before = maw_after = 0
            pass
        if format_bits & 0x8:
            energy_first = evt_data_32[offset+0]
            energy = evt_data_32[offset+1]
            offset += 2
        else:
            energy_first = energy = 0
            pass
        wf_length_32 = (evt_data_32[offset+0]) & 0x03ffffff
        offset += 1 #now the offset points to the wf data
        fadcID = fadcIndex
        channel = channelIndex
        
        
        # compute expected and actual array dimensions
        wf_length16 = 2 * wf_length_32
        header_length16 = offset * 2
        expected_wf_length = len(evt_data_16) - header_length16

        # error check: waveform size must match expectations
        if wf_length16 != expected_wf_length:
            print(len(evt_data_16), header_length16)
            print("ERROR: Waveform size %d doesn't match expected size %d." %
                  (wf_length16, expected_wf_length))
            exit()

        # indexes of stuff (all referring to the 16 bit array)
        i_wf_start = header_length16
        i_wf_stop = i_wf_start + wf_length16

        # handle the waveform(s)
        if wf_length_32 > 0:
            wf_data = evt_data_16[i_wf_start:i_wf_stop]

        if len(wf_data) != expected_wf_length:
            print("ERROR: event %d, we expected %d WF samples and only got %d" %
                  (ievt, expected_wf_length, len(wf_data)))
            exit()

        # final raw wf array
        waveform = wf_data

        # if the wf is too big for pytables, we can window it
        if self.window:
            wf = Waveform(wf_data, self.sample_period, self.decoder_name)
            win_wf, win_ts = wf.window_waveform(self.win_type,
                                                self.n_samp,
                                                self.n_blsamp,
                                                test=False)
            # ts_lo, ts_hi = win_ts[0], win_ts[-1]  # FIXME: what does this mean?

            waveform = win_wf # modify final wf array

            if wf.is_garbage:
                ievt = self.ievt_gbg
                self.ievt_gbg += 1
                self.format_data(locals(), wf.is_garbage)
                return

        if len(waveform) > self.pytables_col_limit and self.h5_format == "table":
            print("WARNING: too many columns for tables output,\n",
                  "         reverting to saving as fixed hdf5 ...")
            self.h5_format = "fixed"

        # set the event number (searchable HDF5 column)
        ievt = self.ievt
        self.ievt += 1

        # send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())


def process_llama_3316(daq_filename, raw_filename, run, n_max, config, verbose):
    """
    convert llama DAQ data to pygama "raw" lh5

    Mario's implementation for the Struck SIS3316 digitizer.
    Requires the llamaDAQ program for producing compatible input files.
    """
    ROW_LIMIT = 5e4

    start = time.time()
    f_in = open(daq_filename.encode('utf-8'), "rb")
    if f_in == None:
        print("Couldn't find the file %s" % daq_filename)
        sys.exit(0)

    #file = llama_3316(f_in,2) #test

    verbosity = 1 if verbose else 0     # 2 is for debug
    sisfile = llama_3316(f_in, verbosity)

    # figure out the total size
    SEEK_END = 2
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)  # rewind
    file_size_MB = file_size / 1e6
    print("Total file size: {:.3f} MB".format(file_size_MB))

    header_dict = sisfile.parse_channelConfigs()    # parse the header dict after manipulating position in file

    # run = get_run_number(header_dict)
    print("Run number: {}".format(run))

    pprint(header_dict)

    #see pygama/pygama/io/decoders/io_base.py
    decoders = []
    #decoders.append(LLAMAStruck3316(metadata=pd.DataFrame.from_dict(header_dict)))   #we just have that one
    decoders.append(LLAMAStruck3316(metadata=header_dict))  #we just have that one
                    # fix: saving metadata using io_bases ctor
                    # have to convert to dataframe here in order to avoid
                    # passing to xml_header.get_object_info in io_base.load_metadata
    channelOne = list(list(header_dict.values())[0].values())[0]
    decoders[0].initialize(1000./channelOne["SampleFreq"], channelOne["Gain"])
        # FIXME: gain set according to first found channel, but gain can change!

    print("pygama will run this fancy decoder: SIS3316Decoder")

    # pass in specific decoder options (windowing, multisampling, etc.)
    #for d in decoders:
    #    d.apply_config(config) #no longer used (why?)

    # ------------ scan over raw data starts here -----------------
    # more code duplication

    print("Beginning Tier 0 processing ...")

    packet_id = 0  # number of events decoded
    row_id = 0      #index of written rows, FIXME maybe gets unused
    unrecognized_data_ids = []

    # header is already skipped by llama_3316,

    def toFile(digitizer, filename_raw, rowID, verbose):
        numb = str(rowID).zfill(4)
        filename_mod = filename_raw + "." + numb
        print("redirecting output file to packetfile "+filename_mod)
        digitizer.save_to_pytables(filename_mod, verbose)


    # start scanning
    while (packet_id < n_max and f_in.tell() < file_size):
        packet_id += 1

        if verbose and packet_id % 1000 == 0:
            update_progress(float(f_in.tell()) / file_size)

        # write periodically to the output file instead of writing all at once
        if packet_id % ROW_LIMIT == 0:
            for d in decoders:
                d.save_to_lh5(raw_filename)
            row_id += 1

        try:
            fadcID, channelID, event_data = sisfile.read_next_event(header_dict)
        except Exception as e:
            print("Failed to get the next event ... Exception:",e)
            break
        if event_data is None:
            break

        decoder = decoders[0]       #well, ...
        # sends data to the pandas dataframe
        decoder.decode_event(event_data, packet_id, header_dict, fadcID, channelID)

    print("done.  last packet ID:", packet_id)
    f_in.close()

    # final write to file
    for d in decoders:
        d.save_to_lh5(raw_filename)

    if verbose:
        update_progress(1)

    if len(unrecognized_data_ids) > 0:
        print("WARNING, Found the following unknown data IDs:")
        for id in unrecognized_data_ids:
            print("  {}".format(id))
        print("hopefully they weren't important!\n")

    # ---------  summary ------------

    print("Wrote: Tier 1 File:\n    {}\nFILE INFO:".format(raw_filename))
    with pd.HDFStore(raw_filename,'r') as store:
        print(store.keys())
    #    # print(store.info())



