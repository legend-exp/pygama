import numpy as np
from pygama import lh5
from .io_base import DataDecoder
import ROOT

class MGDODecoder(DataDecoder):
    """
    Convert ROOT files containing TTrees of MGTEvents into LH5 files. This will
    write to file one LH5 group for each channel, called g#### where #### is
    the channel number, with a table called raw containing:
      waveforms (use LH5 wf specification)
      auxwaveforms (if included)
      energy
      channel
      timestamp
      index
    These are the parameters included in MGVDigitizerData; to include additional
    parameters, a decoder inheriting this one must be created for the digitizer
    data class used. auxwaveforms will be automatically included if it appears
    in the file; in this case it is assumed to appear for all events. Note that
    the waveforms assume fixed waveform length; because the gretina digitizers
    have some variation in waveform lengths when multi-sampling is applied,
    this may cause problems when decoding some MJD data...
    """

    def __init__(self, buffer_length, *args, **kwargs):
        self.decoded_values = {
            'channel' : { 'dtype': 'uint32' },
            'energy' : { 'dtype': 'uint32', 'units': 'adc'},
            'timestamp' : { 'dtype': 'uint64', 'units': 'clk' },
            'index' : { 'dtype': 'uint64' },
            'waveform' : {
                'dtype': 'uint16',
                'datatype': 'waveform',
                'length': None, #override before initializing Table
                'sample_period': None, #override
                'sample_period_units': 'ns',
                'units': 'adc'
            }
        }
        #optional entry that will be added if applicable to decoded_values
        self.auxwf_values = {
            'dtype': 'uint16',
            'datatype': 'waveform',
            'length': None, #override before initializing Table
            'sample_period': None, #override
            'sample_period_units': 'ns',
            'units': 'adc'
        }
        # size of table buffer
        self.table_len = buffer_length
        
        super().__init__(*args, **kwargs)

    def get_table(self, dd, wf, auxwf):
        """ Get the table for this channel. If need be, initialize it too"""
        #Set wf and auxwf values
        self.decoded_values['waveform']['length'] = wf.GetLength()
        self.decoded_values['waveform']['sample_period'] = wf.GetSamplingPeriod()
        if auxwf:
            if 'auxwaveform' not in self.decoded_values:
                self.decoded_values['auxwaveform'] = self.auxwf_values
            self.decoded_values['auxwaveform']['length'] = auxwf.GetLength()
            self.decoded_values['auxwaveform']['sample_period'] = auxwf.GetSamplingPeriod()
        else:
            self.decoded_values.pop('auxwaveform', None)
                
        #initialize the table
        tb = lh5.Table(self.table_len)
        self.initialize_lh5_table(tb)
        return tb
            
    def read_waveform(self, tb, dd, wf, auxwf):
        """
        Fill LH5 table from waveform and digitizer data
        """
        i_chan = tb.loc
        # Fill basic digitizer data
        tb['channel'].nda[i_chan] = dd.GetID()
        tb['energy'].nda[i_chan] = dd.GetEnergy()
        tb['timestamp'].nda[i_chan] = dd.GetTimeStamp()
        tb['index'].nda[i_chan] = dd.GetIndex()
        
        # fill waveform
        tb['waveform']['t0'].nda[i_chan] = wf.GetTOffset()
        wf_dat = np.asarray(wf.GetVectorData())
        wf_tb = tb['waveform']['values'].nda[i_chan, :]
        if len(wf_dat)<len(wf_tb):
            wf_tb[:len(wf_dat)] = wf_dat[:]
            wf_tb[len(wf_dat):] = 0
        else:
            wf_tb[:] = wf_dat[:len(wf_tb)]
            
        # fill auxwaveform, if applicable
        if auxwf:
            tb['auxwaveform']['t0'].nda[i_chan] = auxwf.GetTOffset()
            auxwf_dat = np.asarray(auxwf.GetVectorData())
            auxwf_tb = tb['auxwaveform']['values'].nda[i_chan, :]
            if len(auxwf_dat)<len(auxwf_tb):
                auxwf_tb[:len(auxwf_dat)] = auxwf_dat[:]
                auxwf_tb[len(auxwf_dat):] = 0
            else:
                auxwf_tb[:] = auxwf_dat[:len(auxwf_tb)]

def process_ttree(root_files, raw_file=None, n_max=None, config=None, verbose=False, buffer_size=1024, chans=None, tree_name='MGTree'):
    # Load up the tree (or trees)
    ch = ROOT.TChain(tree_name)
    if isinstance(root_files, str):
        ch.Add(root_files)
    else:
        for root_file in raw_files:
            ch.Add(raw_file)

    dec = MGDODecoder(buffer_size)
    lh5_st = lh5.Store()
    if not raw_file:
        raw_file = root_files.replace('.root', '.lh5')
    
    tables = {} # map from detector channel to output table
    n_tot = 0 # total waveforms
    # loop through MGTEvents in ttree
    for event in ch:
        # loop through waveforms in event
        for i_wf in range(event.event.GetNWaveforms()):
            # Get digitizer data, waveform and auxwaveform (if applicable)
            dd = event.event.GetDigitizerData(i_wf)
            wf = event.event.GetWaveform(i_wf)
            auxwf = event.event.GetAuxWaveform(i_wf) if event.event.GetAuxWaveformArrayStatus() else None

            # Get the output table for this channel
            tb = tables.get(dd.GetID(), None)
            if not tb:
                if verbose:
                    print('Create table for channel', dd.GetID())
                tb = dec.get_table(dd, wf, auxwf)
                tables[dd.GetID()]=tb
            i_chan = tb.loc
            
            dec.read_waveform(tb, dd, wf, auxwf)

            # write table if it is full
            tb.push_row()
            if tb.is_full():
                lh5_st.write_object( tb, 'g{:04d}/raw'.format(dd.GetID()), raw_file, n_rows = tb.loc )
                tb.clear()
                
            n_tot += 1
            
        # check if we have hit n_wf limit. Note that we always include all WFs in an event, which can result in including a few extra waveforms
        if n_max and n_tot >= n_max:
            break

    # Fill remaining events for each table
    for channel, tb in tables.items():
        if verbose:
            print('Wrote to', 'g{:04d}/raw'.format(channel), 'in', raw_file)
        lh5_st.write_object( tb, 'g{:04d}/raw'.format(channel), raw_file, n_rows = tb.loc)
        tb.clear()
    
