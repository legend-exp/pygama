#!/usr/bin/env python3
import sys
import numpy as np
import argparse

from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.transforms import *
from pygama.dsp.units import *

from pygama.io import io_base as io


parser = argparse.ArgumentParser(description=
"""Process a tier 1 LH5 file and produce a tier 2 LH5 file. This entails running
sequence of DSP transforms to calculate un-calibrated parameters.""")

parser.add_argument('file', help="Input (tier 1) LH5 file.")
parser.add_argument('-o', '--output',
                    help="Name of output file. By default, output to ./t2_[input file name].")

parser.add_argument('-v', '--verbose', default=2, type=int,
                    help="Verbosity level: 0=silent, 1=basic warnings, 2=verbose output, 3=debug. Default is 2.")

parser.add_argument('-b', '--block', default=8, type=int,
                    help="Number of waveforms to process simultaneously. Default is 8")

parser.add_argument('-c', '--chunk', default=256, type=int,
                    help="Number of waveforms to read from disk at a time. Default is 256. THIS IS NOT IMPLEMENTED YET!")

parser.add_argument('-g', '--group', default='daqdata',
                    help="Name of group in LH5 file. Default is daqdata.")

args = parser.parse_args()

lh5 = io.LH5Store()
#data = lh5.read_object(args.group, args.file, 0, args.chunk)
data = lh5.read_object(args.group, args.file)

wf_in = data['waveform']['values'].nda
dt = data['waveform']['dt'].nda[0] * unit_parser.parse_unit(data['waveform']['dt'].attrs['units'])

# Set up processing chain
proc = ProcessingChain(block_width=args.block, clock_unit=dt, verbosity=args.verbose)
proc.add_input_buffer("wf", wf_in, dtype='float32')

proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
proc.add_processor(pole_zero, "wf_blsub", 70*us, "wf_pz")
proc.add_processor(trap_filter, "wf_pz", 10*us, 5*us, "wf_trap")
proc.add_processor(np.amax, "wf_trap", 1, "trapmax", signature='(n),()->()', types=['fi->f'])
proc.add_processor(np.divide, "trapmax", 10*us, "trapE")
proc.add_processor(avg_current, "wf_pz", 10, "curr")
proc.add_processor(np.amax, "curr", 1, "A_10", signature='(n),()->()', types=['fi->f'])
proc.add_processor(np.divide, "A_10", "trapE", "AoE")

# Set up the LH5 output
lh5_out = io.LH5Table(size=proc.__buffer_len__)
lh5_out.add_field("trapE", io.LH5Array(proc.get_output_buffer("trapE"), attrs={"units":"ADC"}))
lh5_out.add_field("bl", io.LH5Array(proc.get_output_buffer("bl"), attrs={"units":"ADC"}))
lh5_out.add_field("bl_sig", io.LH5Array(proc.get_output_buffer("bl_sig"), attrs={"units":"ADC"}))
lh5_out.add_field("A", io.LH5Array(proc.get_output_buffer("A_10"), attrs={"units":"ADC"}))
lh5_out.add_field("AoE", io.LH5Array(proc.get_output_buffer("AoE"), attrs={"units":"ADC"}))

print("Processing:\n",proc)
proc.execute()

out = args.output
if out is None:
    out = 't2_'+args.file[args.file.rfind('/')+1:].replace('t1_', '')
print("Writing to: "+out)
lh5.write_object(lh5_out, "data", out)
