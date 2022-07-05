#!/usr/bin/env python3
import sys, json
import numpy as np
import argparse

from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.processors import *
from pygama.dsp.units import *

from pygama.io import lh5


parser = argparse.ArgumentParser(description=
"""Process a tier 1 LH5 file and produce a tier 2 LH5 file. This entails running
sequence of DSP transforms to calculate un-calibrated parameters.""")

arg = parser.add_argument
arg('file', help="Input (tier 1) LH5 file.")
arg('-o', '--output',
                    help="Name of output file. By default, output to ./t2_[input file name].")

arg('-v', '--verbose', default=2, type=int,
                    help="Verbosity level: 0=silent, 1=basic warnings, 2=verbose output, 3=debug. Default is 2.")

arg('-b', '--block', default=8, type=int,
    help="Number of waveforms to process simultaneously. Default is 8")

arg('-c', '--chunk', default=256, type=int,
    help="Number of waveforms to read from disk at a time. Default is 256. THIS IS NOT IMPLEMENTED YET!")

arg('-g', '--group', default='',
    help="Name of group in LH5 file. By default process all base groups. Supports wildcards.")
defaultconfig = os.path.dirname(os.path.realpath(__path__)) + 'dsp_config.json'
arg('-r', '--recreate', action='store_const', const=0, dest='writemode',
    help="Overwrite file if it already exists. Default option. Multually exclusive with --update and --append")
arg('-u', '--update', action='store_const', const=1, dest='writemode',
    help="Update existing file with new values. Useful with the --outpar option. Mutually exclusive with --recreate and --append THIS IS NOT IMPLEMENTED YET!")
arg('-a', '--append', action='store_const', const=1, dest='writemode',
    help="Append values to existing file. Mutually exclusive with --recreate and --update THIS IS NOT IMPLEMENTED YET!")
args = parser.parse_args()

lh5_in = lh5.Store()
groups = lh5_in.ls(args.file, args.group)
out = args.output
if out is None:
    out = 't2_'+args.file[args.file.rfind('/')+1:].replace('t1_', '')

for group in groups:
    print("Processing: " + args.file + '/' + group)
    #data = lh5_in.read_object(args.group, args.file, 0, args.chunk)
    data = lh5_in.read_object(group, args.file)

    wf_in = data['waveform']['values'].nda
    chan_in = data['channel'].nda
    dt = data['waveform']['dt'].nda[0] * unit_parser.parse_unit(data['waveform']['dt'].attrs['units'])

    # Set up processing chain
    proc = ProcessingChain(block_width=args.block, clock_unit=dt, verbosity=args.verbose)

    proc.add_input_buffer("wf", wf_in, dtype='float32')
    proc.add_input_buffer("chan", chan_in)

    # Get tail constants from a hard-coded dict. This is a temporary stop-gap
    # until we have analysis parameters databasing. This is also a test of the
    # param_lookup processor. This would become a necessary processor
    # for mixed-channel processing!
    chan2PZ = {24:242.4, 25:138.9, 26:345.6, 27:142.6, 28:130.3, 29:219.8, 31:395.2, 32:504.3, 33:373.5, 34:408.5, 35:385.5}
    for key, val in chan2PZ.items(): chan2PZ[key] = convert(val, us, dt)
    pz_lookup = param_lookup(chan2PZ, convert(150, us, dt), 'f')
    proc.add_processor(pz_lookup, "chan", "pz_const")

    # Basic Filters
    proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
    proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
    proc.add_processor(pole_zero, "wf_blsub", "pz_const", "wf_pz")
    proc.add_processor(trap_norm, "wf_pz", 10*us, 5*us, "wf_trap")
    proc.add_processor(asymTrapFilter, "wf_pz", 0.05*us, 2*us, 4*us, "wf_atrap")

    # Timepoint calculation
    proc.add_processor(np.argmax, "wf_blsub", 1, "t_max", signature='(n),()->()', types=['fi->i'])
    proc.add_processor(time_point_frac, "wf_blsub", 0.95, "t_max", "tp_95")
    proc.add_processor(time_point_frac, "wf_blsub", 0.8, "t_max", "tp_80")
    proc.add_processor(time_point_frac, "wf_blsub", 0.5, "t_max", "tp_50")
    proc.add_processor(time_point_frac, "wf_blsub", 0.2, "t_max", "tp_20")
    proc.add_processor(time_point_frac, "wf_blsub", 0.05, "t_max", "tp_05")
    proc.add_processor(time_point_thresh, "wf_atrap", 0, "t_max", "tp_0")

    # Energy calculation
    proc.add_processor(np.amax, "wf_trap", 1, "trapEmax", signature='(n),()->()', types=['fi->f'])
    proc.add_processor(fixed_time_pickoff, "wf_trap", "tp_0+(5*us+9*us)", "trapEftp")
    proc.add_processor(trap_pickoff, "wf_pz", 1.5*us, 0, "tp_0", "ct_corr")

    # Current calculation
    proc.add_processor(avg_current, "wf_pz", 10, "curr(len(wf_pz)-10, f)")
    proc.add_processor(np.amax, "curr", 1, "curr_amp", signature='(n),()->()', types=['fi->f'])
    proc.add_processor(np.divide, "curr_amp", "trapEftp", "aoe")

    # DCR calculation: use slope using 1000 samples apart and averaging 200
    # samples, with the start 1.5 us offset from t0
    proc.add_processor(trap_pickoff, "wf_pz", 200, 1000, "tp_0+1.5*us", "dcr_unnorm")
    proc.add_processor(np.divide, "dcr_unnorm", "trapEftp", "dcr")

    # Tail slope. Basically the same as DCR, except with no PZ correction
    proc.add_processor(linear_fit, "wf_blsub[3000:]", "wf_b", "wf_m")
    proc.add_processor(np.divide, "-wf_b", "wf_m", "tail_rc")

    # Set up the LH5 output
    lh5_out = lh5.Table(size=proc._buffer_len)
    lh5_out.add_field("trapEmax", lh5.Array(proc.get_output_buffer("trapEmax"), attrs={"units":"ADC"}))
    lh5_out.add_field("trapEftp", lh5.Array(proc.get_output_buffer("trapEftp"), attrs={"units":"ADC"}))
    lh5_out.add_field("ct_corr", lh5.Array(proc.get_output_buffer("ct_corr"), attrs={"units":"ADC*ns"}))
    lh5_out.add_field("bl", lh5.Array(proc.get_output_buffer("bl"), attrs={"units":"ADC"}))
    lh5_out.add_field("bl_sig", lh5.Array(proc.get_output_buffer("bl_sig"), attrs={"units":"ADC"}))
    lh5_out.add_field("A", lh5.Array(proc.get_output_buffer("curr_amp"), attrs={"units":"ADC"}))
    lh5_out.add_field("AoE", lh5.Array(proc.get_output_buffer("aoe"), attrs={"units":"ADC"}))
    lh5_out.add_field("dcr", lh5.Array(proc.get_output_buffer("dcr"), attrs={"units":"ADC"}))

    lh5_out.add_field("tp_max", lh5.Array(proc.get_output_buffer("tp_95", unit=us), attrs={"units":"us"}))
    lh5_out.add_field("tp_95", lh5.Array(proc.get_output_buffer("tp_95", unit=us), attrs={"units":"us"}))
    lh5_out.add_field("tp_80", lh5.Array(proc.get_output_buffer("tp_80", unit=us), attrs={"units":"us"}))
    lh5_out.add_field("tp_50", lh5.Array(proc.get_output_buffer("tp_50", unit=us), attrs={"units":"us"}))
    lh5_out.add_field("tp_20", lh5.Array(proc.get_output_buffer("tp_20", unit=us), attrs={"units":"us"}))
    lh5_out.add_field("tp_05", lh5.Array(proc.get_output_buffer("tp_05", unit=us), attrs={"units":"us"}))
    lh5_out.add_field("tp_0", lh5.Array(proc.get_output_buffer("tp_0", unit=us), attrs={"units":"us"}))
    lh5_out.add_field("tail_rc", lh5.Array(proc.get_output_buffer("tail_rc", unit=us), attrs={"units":"us"}))

    proc.execute()

    groupname = group[:group.rfind('/')+1]+"dsp"
    print("Writing to: " + out + "/" + groupname)
    lh5_in.write_object(lh5_out, groupname, out)
