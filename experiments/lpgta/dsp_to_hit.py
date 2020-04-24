#!/usr/bin/env python3
import argparse

from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.io import lh5

from pygama.dsp.units import *
from pygama.dsp.processors import *

def main():
    """
    Clone of pygama/apps/raw_to_dsp.py.  Intended for quick prototyping of dsp_to_hit
    processors.  Heavy lifting with many input/output files should be moved to a
    more specialized processing app, with raw_to_dsp and dsp_to_hit both moved to
    functions in pygama.io.
    """
    parser = argparse.ArgumentParser(description=
    """Process a 'pygama DSP LH5' file and produce a 'pygama HIT LH5' file.""")
    parser.add_argument('file', help="Input (dsp) LH5 file.")
    parser.add_argument('-o', '--output',
                        help="Name of output file. By default, output to ./t2_[input file name].")
    
    parser.add_argument('-g', '--group', default='',
                        help="Name of group in LH5 file. By default process all base groups. Supports wildcards.")
    args = parser.parse_args()

    
    # import h5py
    # f = h5py.File('/Users/wisecg/Data/LPGTA/raw/geds/cal/LPGTA_r0018_20200302T184433Z_cal_geds_raw.lh5')
    # # print(f['g024/raw'].keys())
    # # ['baseline', 'channel', 'energy', 'ievt', 'numtraces', 'packet_id', \
    # #  'timestamp', 'tracelist', 'waveform', 'wf_max', 'wf_std']
    # def print_attrs(name, obj):
    #     print(name)
    #     for key, val in obj.attrs.items():
    #         print("    attr: %s  val: %s" % (key, val))
    # # f = h5py.File(f,'r')
    # f.visititems(print_attrs)
    
    # exit()
    
    lh5_in = lh5.Store()
    groups = lh5_in.ls(args.file, args.group)
    
    out = args.output if args.output is not None else './d2h_test.lh5'
    print('output file:', out)
    
    for group in groups[:1]:
        print(group)
    
        print("Processing: " + args.file + '/' + group)
        
        #data = lh5_in.read_object(args.group, args.file, 0, args.chunk)
        
        data = lh5_in.read_object(group+'/raw', args.file)
        
        # print(type(data))#, data.keys())
        # print(data.keys())
        
        wf_in = data['waveform']['values'].nda
        dt = data['waveform']['dt'].nda[0] * unit_parser.parse_unit(data['waveform']['dt'].attrs['units'])
        # print(wf_in.shape)
        
        ene_in = data['energy'].nda
        # print(ene_in.shape)
        # print(ene_in.dtype)
        # exit()
        
        n_block = 8
        verbose = 1
        
        proc = ProcessingChain(block_width=n_block, clock_unit=dt, verbosity=verbose)

        # proc.add_input_buffer("wf", wf_in, dtype='float32')
        
        proc.add_input_buffer("ene_in", ene_in, dtype='uint16')
        
        proc.add_processor(energy_cal, "ene_in")
        
        
        
        
        
    



if __name__=='__main__':
    main()