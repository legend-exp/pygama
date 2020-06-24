import os
import json
import h5py
import time
import numpy as np
from collections import OrderedDict
from pprint import pprint

from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.units import *
from pygama.io import lh5


def raw_to_dsp(f_raw, f_dsp, dsp_config, lh5_tables=None, verbose=False,
               n_max=np.inf, overwrite=True, buffer_len=8):
    """
    Convert raw LH5 files with waveforms to dsp files.

    Uses the ProcessingChain class.
    The list of processors is specifed via a JSON file.
    To preserve the ordering, we read in using an OrderedDict.
    """
    t_start = time.time()

    if not isinstance(dsp_config, OrderedDict):
        print('Error, dsp_config must be an OrderedDict')
        exit()

    raw_store = lh5.Store()
    lh5_file = raw_store.gimme_file(f_raw, 'r')


    # if no group is specified, assume we want to decode every table in the file
    if lh5_tables is None:
        lh5_tables = []
        lh5_tables_temp = raw_store.ls(f_raw)

        # sometimes 'raw' is nested, e.g g024/raw
        for tb in lh5_tables_temp:
            if "raw" not in tb:
                tbname = raw_store.ls(lh5_file[tb])[0]
                if "raw" in tbname:
                    tb = tb +'/'+ tbname # g024 + /raw
            lh5_tables.append(tb)
    #make sure every group points to waveforms, if not, remove the group
    for tb in lh5_tables:
        if 'raw' not in tb:
            lh5_tables.remove(tb)

    # set up DSP for each table
    chains = []

    for tb in lh5_tables:
        print('Processing table: ', tb)

        # load primary table
        data_raw = raw_store.read_object(tb, f_raw, start_row=0, n_rows=n_max)

        # load waveform info
        if "waveform" not in data_raw.keys():
            print(f"waveform data not found in table: {tb}.  skipping ...")
            continue
        wf_in = data_raw["waveform"]["values"].nda
        wf_units = data_raw['waveform']['dt'].attrs['units']
        dt = data_raw['waveform']['dt'].nda[0] * unit_parser.parse_unit(wf_units)

        # set up DSP for this table (JIT-compiles functions the first time)
        pc = ProcessingChain(block_width=buffer_len, clock_unit=dt, verbosity=1)
        pc.add_input_buffer('wf', wf_in, dtype='float32')
        pc.set_processor_list(dsp_config)

        # set up LH5 output table
        tb_out = lh5.Table(size = pc._buffer_len)
        cols = dsp_config['outputs']
        # cols = pc.get_column_names() # TODO: should add this method
        for col in cols:
            lh5_arr = lh5.Array(pc.get_output_buffer(col),
                                attrs={'units':dsp_config['outputs'][col]})
            tb_out.add_field(col, lh5_arr)

        # get names of non-wf columns to copy over
        copy_cols = [col for col in data_raw.keys() if col != 'waveform']
        for col in copy_cols:
            print('copying col:', col)
            lh5_arr = lh5.Array(data_raw[col].nda)
            tb_out.add_field(col, lh5_arr)

        chains.append((tb, tb_out, pc))


    # run DSP.  TODO: parallelize this
    print('Writing to output file:', f_dsp)
    for tb, tb_out, pc in chains:
        print(f'Processing table: {tb} ...')
        pc.execute()

        print(f'Done.  Writing to file ...')
        raw_store.write_object(tb_out, tb, f_dsp)

    t_elap = (time.time() - t_start) / 60
    print(f'Done processing.  Time elapsed: {t_elap:.2f} min.')

# def raw_to_dsp(lh5_in, json_file, buffer_len=8):
#     """
#     Will add doc string when I have time
#     """
#
#     with open(json_file) as f:
#         config = json.load(f)
#     paths = config["paths"]
#     options = config["options"]
#
#     wf_in = lh5_in["waveform"]["values"].nda
#     dt = lh5_in['waveform']['dt'].nda[0] * unit_parser.parse_unit(lh5_in['waveform']['dt'].attrs['units'])
#
#
#     proc = ProcessingChain(block_width=buffer_len, clock_unit=dt, verbosity=3) #NOTE Need to add verbosity input to function
#     proc.add_input_buffer("wf", wf_in, dtype='float32')
#     proc.set_processor_list(json_file)
#     # proc.get_column_names() loop over
#
#     lh5_out = lh5.Table(size=proc._buffer_len)
#
#     for output in config["outputs"]:
#
#         lh5_out.add_field(output, lh5.Array(proc.get_output_buffer(output),
#                                                attrs={"units":config["outputs"][output]}))
#
#     return lh5_out, proc
#
#
#
#
#
# def example():
#
#     lh5_in = lh5.Store()
#     #data = lh5_in.read_object(args.group, args.file, 0, args.chunk)
#     data = lh5_in.read_object(args.group, args.file)
#
#     wf_in = data['waveform']['values'].nda
#     dt = data['waveform']['dt'].nda[0] * unit_parser.parse_unit(data['waveform']['dt'].attrs['units'])
#
#     # Set up processing chain
#     proc = ProcessingChain(block_width=args.block, clock_unit=dt, verbosity=args.verbose)
#     proc.add_input_buffer("wf", wf_in, dtype='float32')
#
#     # Basic Filters
#     proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
#     proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
#     proc.add_processor(pole_zero, "wf_blsub", 70*us, "wf_pz")
#     proc.add_processor(trap_norm, "wf_pz", 10*us, 5*us, "wf_trap")
#     proc.add_processor(asymTrapFilter, "wf_pz", 0.05*us, 2*us, 4*us, "wf_atrap")
#
#     # Timepoint calculation
#     proc.add_processor(np.argmax, "wf_blsub", 1, "t_max", signature='(n),()->()', types=['fi->i'])
#     proc.add_processor(time_point_frac, "wf_blsub", 0.95, "t_max", "tp_95")
#     proc.add_processor(time_point_frac, "wf_blsub", 0.8, "t_max", "tp_80")
#     proc.add_processor(time_point_frac, "wf_blsub", 0.5, "t_max", "tp_50")
#     proc.add_processor(time_point_frac, "wf_blsub", 0.2, "t_max", "tp_20")
#     proc.add_processor(time_point_frac, "wf_blsub", 0.05, "t_max", "tp_05")
#     proc.add_processor(time_point_thresh, "wf_atrap[0:1200]", 0, "tp_0")
#
#     # Energy calculation
#     proc.add_processor(np.amax, "wf_trap", 1, "trapEmax", signature='(n),()->()', types=['fi->f'])
#     proc.add_processor(fixed_time_pickoff, "wf_trap", "tp_0", 5*us+9*us, "trapEftp")
#     proc.add_processor(trap_pickoff, "wf_pz", 1.5*us, 0, "tp_0", "ct_corr")
#
#     # Current calculation
#     proc.add_processor(avg_current, "wf_pz", 10, "curr")
#     proc.add_processor(np.amax, "curr", 1, "curr_amp", signature='(n),()->()', types=['fi->f'])
#     proc.add_processor(np.divide, "curr_amp", "trapEftp", "aoe")
#
#     # Set up the LH5 output
#     lh5_out = lh5.Table(size=proc._buffer_len)
#     lh5_out.add_field("trapEmax", lh5.Array(proc.get_output_buffer("trapEmax"), attrs={"units":"ADC"}))
#     lh5_out.add_field("trapEftp", lh5.Array(proc.get_output_buffer("trapEftp"), attrs={"units":"ADC"}))
#     lh5_out.add_field("ct_corr", lh5.Array(proc.get_output_buffer("ct_corr"), attrs={"units":"ADC*ns"}))
#     lh5_out.add_field("bl", lh5.Array(proc.get_output_buffer("bl"), attrs={"units":"ADC"}))
#     lh5_out.add_field("bl_sig", lh5.Array(proc.get_output_buffer("bl_sig"), attrs={"units":"ADC"}))
#     lh5_out.add_field("A", lh5.Array(proc.get_output_buffer("curr_amp"), attrs={"units":"ADC"}))
#     lh5_out.add_field("AoE", lh5.Array(proc.get_output_buffer("aoe"), attrs={"units":"ADC"}))
#
#     lh5_out.add_field("tp_max", lh5.Array(proc.get_output_buffer("tp_95"), attrs={"units":"ticks"}))
#     lh5_out.add_field("tp_95", lh5.Array(proc.get_output_buffer("tp_95"), attrs={"units":"ticks"}))
#     lh5_out.add_field("tp_80", lh5.Array(proc.get_output_buffer("tp_80"), attrs={"units":"ticks"}))
#     lh5_out.add_field("tp_50", lh5.Array(proc.get_output_buffer("tp_50"), attrs={"units":"ticks"}))
#     lh5_out.add_field("tp_20", lh5.Array(proc.get_output_buffer("tp_20"), attrs={"units":"ticks"}))
#     lh5_out.add_field("tp_05", lh5.Array(proc.get_output_buffer("tp_05"), attrs={"units":"ticks"}))
#     lh5_out.add_field("tp_0", lh5.Array(proc.get_output_buffer("tp_0"), attrs={"units":"ticks"}))
#
#     print("Processing:\n",proc)
#     proc.execute()
#
#     out = args.output
#     if out is None:
#         out = 't2_'+args.file[args.file.rfind('/')+1:].replace('t1_', '')
#     print("Writing to: "+out)
#     lh5_in.write_object(lh5_out, "data", out)
#
#     t_start = time.time()
#
#     if not isinstance(dsp_config, OrderedDict):
#         print('Error, dsp_config must be an OrderedDict')
#         exit()
#
#     raw_store = lh5.Store()
#
#     # if no group is specified, assume we want to decode every table in the file
#     if lh5_tables is None:
#         lh5_tables = raw_store.ls(f_raw)
#
#     # set up DSP for each table
#     chains = []
#     for tb in lh5_tables:
#         print('Processing table: ', tb)
#
#         # load primary table
#         data_raw = raw_store.read_object(tb, f_raw, start_row=0, n_rows=n_max)
#
#         # load waveform info
#         if "waveform" not in data_raw.keys():
#             print(f"waveform data not found in table: {tb}.  skipping ...")
#             continue
#         wf_in = data_raw["waveform"]["values"].nda
#         wf_units = data_raw['waveform']['dt'].attrs['units']
#         dt = data_raw['waveform']['dt'].nda[0] * unit_parser.parse_unit(wf_units)
#
#         # set up DSP for this table (JIT-compiles functions the first time)
#         pc = ProcessingChain(block_width=buffer_len, clock_unit=dt, verbosity=0)
#         pc.add_input_buffer('wf', wf_in, dtype='float32')
#         pc.set_processor_list(dsp_config)
#
#         # set up LH5 output table
#         tb_out = lh5.Table(size = pc._buffer_len)
#         cols = dsp_config['outputs']
#         # cols = pc.get_column_names() # TODO: should add this method
#         for col in cols:
#             lh5_arr = lh5.Array(pc.get_output_buffer(col),
#                                 attrs={'units':dsp_config['outputs'][col]})
#             tb_out.add_field(col, lh5_arr)
#
#         chains.append((tb, tb_out, pc))
#
#
#     # run DSP.  TODO: parallelize this
#     print('Writing to output file:', f_dsp)
#     for tb, tb_out, pc in chains:
#         print(f'Processing table: {tb} ...')
#         # pc.execute()
#         # print(f'Done.  Writing to file ...')
#         # raw_store.write_object(tb_out, tb, f_dsp)
#
#     t_elap = (time.time() - t_start) / 60
#     print(f'Done processing.  Time elapsed: {t_elap:.2f} min.')


if __name__=="__main__":
    example()
