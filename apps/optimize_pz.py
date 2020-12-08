#!/usr/bin/env python3
import sys, json
import numpy as np
import argparse

from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.processors import *
from pygama.dsp.units import *
from pygama.utils import update_progress
from pygama.io import lh5

usage = """
Find the pole-zero constant that will result in the flattest tail for
the PZ waveform. This will measure the slope of the falling tail of each
waveform. The slope and baseline will be used to calculate a decay tail
for the tail. Finally, a distribution of the decay tails in the run
will be maximized. The maximum will be published to the metadata dir (TODO)
to be used in signal processing.
"""

parser = argparse.ArgumentParser(description=usage)

arg = parser.add_argument
arg('file', help="Input (tier 1) LH5 file.")
arg('-o', '--output', default='pz_consts.json',
    help="Name of output file. By default, output to ./pz_consts.json")

arg('-v', '--verbose', default=2, type=int,
    help="Verbosity level: 0=silent, 1=basic warnings, 2=verbose output, 3=debug. Default is 2.")

arg('-c', '--channel', default='',
    help="Name of group in LH5 file. By default process all base groups. Supports wildcards.")
arg('-r', '--range', default='50-500', type=str,
    help="Range in us for time constant, as a hyphen-separated pair of numbers. Default is '50-300")
arg('-b', '--BL_samples', default='0:1000', type=str,
    help="Samples to use for baseline subtraction as a slice. Default is '0:1000'")
arg('-t', '--Tail_samples', default='3000:', type=str,
    help="Samples to use for linear fit of falling tail as a slice. Default is '3000:'")

arg('-N', '--nentries', default=None, type=int,
    help="Number of waveforms to process for each channel. By default use all.")
arg('-B', '--block', default=16, type=int,
    help="Number of waveforms to process simultaneously. Default is 8")

arg('-C', '--chunk', default=3200, type=int,
    help="Number of waveforms to read from disk at a time. Default is 256.")

args = parser.parse_args()

lh5_st = lh5.Store()
chans = lh5_st.ls(args.file, args.channel)

rc_range = tuple([ round(float(tc),1) for tc in args.range.split('-') ])
if len(rc_range)!=2:
    print("Range must have exactly two values")
n_bins = int((rc_range[1]-rc_range[0])/0.1)

rc_const_lib = {}

np.seterr(all='ignore')

for chan_name in chans:
    group = chan_name + '/raw'
    print("Processing: " + args.file + '/' + group)
    
    tot_n_rows = lh5_st.read_n_rows(group, args.file)
    if args.nentries is not None:
        tot_n_rows = min(tot_n_rows, args.nentries)
    lh5_in, n_rows_read = lh5_st.read_object(group, args.file, 0, args.chunk)
    wf_in = lh5_in['waveform']['values'].nda
    dt = lh5_in['waveform']['dt'].nda[0] * unit_parser.parse_unit(lh5_in['waveform']['dt'].attrs['units'])

    # Set up processing chain
    proc = ProcessingChain(block_width=args.block, clock_unit=dt, verbosity=args.verbose)
    proc.add_input_buffer("wf", wf_in, dtype='float32')

    # measure baseline, then window and baseline subtract
    proc.add_processor(mean_stdev, "wf[{}]".format(args.BL_samples), "bl", "bl_sig")
    proc.add_processor(np.subtract, "wf[{}]".format(args.Tail_samples), "bl", "wf_blsub")

    # RC constant. Linear fit of log of falling tail.
    proc.add_processor(np.log, "wf_blsub", "tail_log")
    proc.add_processor(linear_fit, "tail_log", "tail_b", "tail_m")
    proc.add_processor(np.divide, -1, "tail_m", "tail_rc")

    # Get tail_rc output buffer
    tail_rc = proc.get_output_buffer("tail_rc", unit=us)

    # Process and create histogram
    rc_hist, bins = np.histogram([], n_bins, rc_range)
    for start_row in range(0, tot_n_rows, args.chunk):
        if args.verbose > 0:
            update_progress(start_row/tot_n_rows)
        lh5_in, n_rows = lh5_st.read_object(group, args.file, start_row=start_row, obj_buf=lh5_in)
        proc.execute(0, n_rows)
        rc_hist += np.histogram(tail_rc, n_bins, rc_range)[0]

    if args.verbose > 0: update_progress(1)

    # Get mode of hist and record it
    rc_const = bins[np.argmax(rc_hist)]
    if args.verbose > 0: print("Optimal pole-zero constant is", rc_const, "us")
    rc_const_lib[chan_name] = {'pz_const':"{:.1f}*us".format(rc_const)}

with open(args.output, 'w') as f:
    json.dump(rc_const_lib, f, indent=2, sort_keys=True)
