import os
import numpy as np
from pprint import pprint

from ..utils import *
from .io_base import DataDecoder
from . import lh5


class FlashCamEventDecoder(DataDecoder):
    """ 
    decode FlashCam digitizer event data.
    """
    def __init__(self, *args, **kwargs):
        """
        """
        # these are read for every event (decode_event)
        self.decoded_values = {
            'packet_id': { # packet index in file
               'dtype': 'uint32',
             },
            'ievt': { # index of event
              'dtype': 'int32',
            },
            'timestamp': { # time since beginning of file
              'dtype': 'float32',
              'units': 's',
            },
            'numtraces': { # number of triggered adc channels
              'dtype': 'int32',
            },
            'tracelist': { # list of triggered adc channels
              'dtype': 'int16',
              'datatype': 'array<1>{array<1>{real}}', # vector of vectors
              'length_guess': 16,
            },
            'baseline': { # fpga baseline
              'dtype': 'uint16',
            },
            'energy': {  # fpga energy
              'dtype': 'uint16',
            },
            'channel': { # right now, index of the trigger (trace)
              'dtype': 'uint32',
            },
            'wf_max': { # ultra-simple np.max energy estimation
              'dtype': 'uint16',
            },
            'wf_std': { # ultra-simple np.std noise estimation
              'dtype': 'float32',
            },
            'waveform': { # digitizer data
              'dtype': 'uint16',
              'datatype': 'waveform', 
              'length': 65532, # max value. override this before initializing buffers to save RAM
              'sample_period': 16, # override if a different clock rate is used
              'sample_period_units': 'ns',
              't0_units': 'ns',
            },
        }

        # these are read for every file (get_file_config)
        # FIXME: push into a file header object?
        self.config_names = [
            'nsamples', # samples per channel
            'nadcs', # number of adc channels
            'ntriggers', # number of triggertraces
            'telid', # id of telescope
            'adcbits', # bit range of the adc channels
            'sumlength', # length of the fpga integrator
            'blprecision', # precision of the fpga baseline
            'mastercards', # number of attached mastercards
            'triggercards', # number of attached triggercards
            'adccards', # number of attached fadccards
            'gps', # gps mode (0: not used, 1: external pps and 10MHz)
        ]
    
        super().__init__(*args, **kwargs)
        self.skipped_channels = {}
        
        
    def get_file_config(self, fcio):
        """
        access FCIOConfig members once when each file is opened
        """
        self.file_config = {c:getattr(fcio, c) for c in self.config_names}
        self.decoded_values['waveform']['length'] = self.file_config['nsamples']


    def decode_packet(self, fcio, lh5_tables, packet_id, verbose=False):
        """
        access FCIOEvent members for each event in the raw file
        """

        ievt      = fcio.eventnumber # the eventnumber since the beginning of the file
        timestamp = fcio.eventtime   # the time since the beginning of the file in seconds
        eventsamples = fcio.nsamples   # number of sample per trace
        numtraces = fcio.numtraces   # number of triggered adcs
        tracelist = fcio.tracelist   # list of triggered adcs
        traces    = fcio.traces      # the full traces for the event: (nadcs, nsamples)
        baselines = fcio.baseline    # the fpga baseline values for each channel in LSB
        energies  = fcio.daqenergy   # the fpga energy values for each channel in LSB

        # all channels are read out simultaneously for each event
        for iwf in tracelist:
            tb = lh5_tables
            if not isinstance(tb, lh5.Table): 
                if iwf not in lh5_tables:
                    #print('FlashCamEventDecoder Warning: no table buffer for channel', iwf)
                    if iwf not in self.skipped_channels: 
                        self.skipped_channels[iwf] = 0
                    self.skipped_channels[iwf] += 1
                    continue
                tb = lh5_tables[iwf]
            if eventsamples != tb['waveform']['values'].nda.shape[1]:
                print('FlashCamEventDecoder Warning: event wf length was',
                      eventsamples, 'when',
                      self.decoded_values['waveform']['length'], 'were expected')
            ii = tb.loc
            tb['channel'].nda[ii] = iwf 
            tb['packet_id'].nda[ii] = packet_id
            tb['ievt'].nda[ii] =  ievt
            tb['timestamp'].nda[ii] =  timestamp
            tb['numtraces'].nda[ii] =  numtraces
            tb['tracelist'].set_vector(ii, tracelist)
            tb['baseline'].nda[ii] = baselines[iwf]
            tb['energy'].nda[ii] = energies[iwf]
            waveform = traces[iwf]
            tb['wf_max'].nda[ii] = np.amax(waveform)
            tb['wf_std'].nda[ii] = np.std(waveform)
            tb['waveform']['values'].nda[ii][:] = waveform
            tb.push_row()

        return 36*4 + numtraces*(1 + eventsamples + 2)*2


class FlashCamStatusDecoder(DataDecoder):
    """ 
    decode FlashCam digitizer status data.
    """
    def __init__(self, *args, **kwargs):

        self.decoded_values = {
            'status': { # 0: Errors occured, 1: no errors
              'dtype': 'int32',
            },
            'statustime': { # fc250 seconds, microseconds, dummy, startsec startusec 
              'dtype': 'float32',
              'units': 's',
            },
            'cputime': { # CPU seconds, microseconds, dummy, startsec startusec 
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

        # these are read for every file (get_file_config)
        self.config_names = [
            'nsamples', # samples per channel
            'nadcs', # number of adc channels
            'ntriggers', # number of triggertraces
            'telid', # id of telescope
            'adcbits', # bit range of the adc channels
            'sumlength', # length of the fpga integrator
            'blprecision', # precision of the fpga baseline
            'mastercards', # number of attached mastercards
            'triggercards', # number of attached triggercards
            'adccards', # number of attached fadccards
            'gps', # gps mode (0: not used, 1: external pps and 10MHz)
        ]
    
        super().__init__(*args, **kwargs)
        
        
    def get_file_config(self, fcio):
        """
        access FCIOConfig members once when each file is opened
        """
        self.file_config = {c:getattr(fcio, c) for c in self.config_names}


    def decode_packet(self, fcio, lh5_table, packet_id, verbose=False):
        """
        access FC status (temp., log, ...)
        """
        # aliases for brevity
        tb = lh5_table
        ii = tb.loc

        # status -- 0: Errors occured, 1: no errors
        tb['status'].nda[ii] = fcio.status 

        # times
        tb['statustime'].nda[ii] = fcio.statustime[0]+fcio.statustime[1]/1e6
        tb['cputime'].nda[ii] = fcio.statustime[2]+fcio.statustime[3]/1e6

        # Total number of cards (number of status data to follow)
        tb['cards'].nda[ii] = fcio.cards 

        # Size of each status data
        tb['size'].nda[ii] = fcio.size 

        # FC card-wise environment status (temp., volt., hum., ...)
        tb['environment'].nda[ii][:] = fcio.environment 

        # FC card-wise list DAQ errors during data taking
        tb['totalerrors'].nda[ii] = fcio.totalerrors 
        tb['linkerrors'].nda[ii] = fcio.linkerrors
        tb['ctierrors'].nda[ii] = fcio.ctierrors
        tb['enverrors'].nda[ii] = fcio.enverrors
        tb['othererrors'].nda[ii][:] = fcio.othererrors

        tb.push_row()

        # sizeof(fcio_status)
        return 302132 


def process_flashcam(daq_file, raw_files, n_max, config, verbose, buffer_size=8092, chans=None, f_out = ''):
    """
    decode FlashCam data, using the fcutils package to handle file access,
    and the FlashCam DataTaker to save the results and write to output.
    
    `raw_files` can be a string, or a dict with a label for each file:
      `{'geds':'filename_geds.lh5', 'muvt':'filename_muvt.lh5}`
    """
    import fcutils
    
    if isinstance(raw_files, str):
        single_output = True
        f_out = raw_files
    elif len(raw_files) == 1:
        single_output = True
        f_out = raw_files['']
    else:
        single_output = False
        
    fcio = fcutils.fcio(daq_file)
    
    # set up event decoder
    event_decoder = FlashCamEventDecoder()
    event_decoder.get_file_config(fcio)

    # parse daq_to_raw config 
    event_tbs = {}
    tb_grp_file_list = [] # map LH5 table to group name and file name
    
    if 'daq_to_raw' in config and 'ch_groups' in config['daq_to_raw']:
        ch_groups = config['daq_to_raw']['ch_groups']
        for group, attrs in ch_groups.items():
            ch_range = attrs['ch_range']
            try:
               subsystem = attrs['sysn']
            except:
               subsystem='default'
            
            if not single_output:
                if subsystem not in raw_files.keys():
                    print('Error, no output file found for subsystem:', subsystem)

            # each subsystem can output a table per channel, or a combined table
            tb_per_ch = False
            if 'tb_per_ch' in attrs:
                tb_per_ch = (attrs['tb_per_ch'].lower() == "true")

            if tb_per_ch:
                try:
                   cr = range(ch_range[0], ch_range[1]+1)
                except:
                   cr = [0] 
                for ch in cr:
                    
                    tb = lh5.Table(buffer_size)
                    event_tbs[ch] = tb
                    event_decoder.initialize_lh5_table(tb)
                    
                    if '{ch:' in group: ch_group = group.format(ch=ch)
                    ch_group = ch_group + '/raw'
                    # out_file = raw_files[subsystem].format_map(attrs)
                    out_file = f_out if single_output else raw_files[subsystem]
                    tb_grp_file_list.append( (tb, ch_group, out_file) )
       

            # one table for all channels in the range
            else: 
                tb = lh5.Table(buffer_size)
                event_decoder.initialize_lh5_table(tb)
                group = group + '/raw'
                # out_file = raw_file.format_map(attrs)
                out_file = f_out if single_output else raw_files[subsystem]
                tb_grp_file_list.append( (tb, group, out_file) )
    else:
        print('Config not found.  Single-table mode')
        tb = lh5.Table(buffer_size)
        event_decoder.initialize_lh5_table(tb)
        tb_grp_file_list.append( (tb, 'raw', f_out) )
        event_tbs = tb
    
    if verbose:
        print('Output group : output file')
        for a, b, c in tb_grp_file_list:
            print(b, ':', c.split('/')[-1])
        
    # set up status decoder (this is 'auxs' output)
    status_decoder = FlashCamStatusDecoder()
    status_decoder.get_file_config(fcio)
    status_tb = lh5.Table(buffer_size)
    status_decoder.initialize_lh5_table(status_tb)
    try:
      status_filename = f_out if single_output else raw_files['auxs']
    except:
      status_filename = "stat"
    # TODO: add overwrite capability
    lh5_store = lh5.Store()
    
    # loop over raw data packets
    i_debug = 0
    packet_id = 0
    rc = 1
    bytes_processed = 0
    file_size = os.path.getsize(daq_file)
    
    while rc and packet_id < n_max:
        rc = fcio.get_record()
        
        # Skip non-interesting records
        # FIXME: push to a buffer of skipped packets?
        if rc == 0 or rc == 1 or rc == 2 or rc == 5: continue

        packet_id += 1

        if verbose and packet_id % 1000 == 0:
            # FIXME: is cast to float necessary?
            pct_done = bytes_processed / file_size
            if n_max < np.inf and n_max > 0: pct_done = packet_id / n_max
            update_progress(pct_done)

        # Status record
        if rc == 4: 
            bytes_processed += status_decoder.decode_packet(fcio, status_tb, packet_id)
            if status_tb.is_full():
                lh5_store.write_object(status_tb, 'stat', status_filename, n_rows=status_tb.size)
                status_tb.clear()

        # Event or SparseEvent record
        if rc == 3 or rc == 6: 
            for tb, group, filename in tb_grp_file_list:
                if tb.size - tb.loc < fcio.numtraces: # might overflow
                    lh5_store.write_object(tb, group, filename, n_rows=tb.loc)
                    tb.clear()
            
            bytes_processed += event_decoder.decode_packet(fcio, event_tbs, packet_id)

            # i_debug += 1
            # if i_debug == 10:
            #    print("breaking early")
            #    break # debug, deleteme

    # end of loop, write to file once more
    for tb, group, filename in tb_grp_file_list:
        if tb.loc != 0:
            lh5_store.write_object(tb, group, filename, n_rows=tb.loc)
            tb.clear()
    if status_tb.loc != 0:
        lh5_store.write_object(status_tb, 'stat', status_filename,
                               n_rows=status_tb.loc)
        status_tb.clear()

    if verbose:
        update_progress(1)
        print(packet_id, 'packets decoded')

    if len(event_decoder.skipped_channels) > 0:
        print("Warning - daq_to_raw skipped some channels in file")
        if verbose: 
            for ch, n in event_decoder.skipped_channels.items():
                print("  ch", ch, ":", n, "hits")

    return bytes_processed
