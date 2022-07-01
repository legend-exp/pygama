#!/usr/bin/env python3
import argparse

import pandas as pd

import pygama.lgdo.lh5_store as lh5
from pygama.evt.build_evt import build_evt

# EXAMPLE USAGE:
# ./build_tcm.py /global/project/projectdirs/legend/data/lngs/l200_commissioning/orca_data/pygama/raw/cal/p00/r001/L60_p0_r1_20220606T015344Z_f0.lh5 -o ~/tcm_test/L60_p0_r1_20220606T015344Z_f0_tcm.lh5

def main():
    doc = """Demonstrate usage of the `build_tcm` function, to build a time
    coincidence map and organize data from many channels in a single cycle file,
    into an event-like structure where coincindent events are time-ordered and
    grouped together into sub-events."""

    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('input', type=str, help='input file name (required)')
    arg('-o', '--output', type=str, help='output file name')
    args = par.parse_args()

    # set i/o
    f_in = args.input
    f_out = None if not args.output else args.output

    # run routines
    gen_tcm(f_in, f_out)
    show_tcm(f_out)


def gen_tcm(f_in, f_out):
    """ run build_evt with the special hardware TCM config on a raw file. """

    # test case -- create a hardware TCM for ORCA FlashCam data.
    # test file:
    # /global/project/projectdirs/legend/data/lngs/l200_commissioning/orca_data/pygama/raw/cal/p00/r001/L60_p0_r1_20220606T015344Z_f0.lh5
    copy_cols = ['eventnumber', 'channel']

    # this makes a 'hardware TCM' which just uses the event counter from FlashCam instead of a timestamp.
    builder_config = {
    'ts_unit' : 1,       # give the conversion of timestamps to seconds
    'ch_col' : 'channel',   # name of column with channel ID (should be int)
    'ts_col' : 'eventnumber', # name of column with timestamps
    'coin_window' : 0.5,   # length of coincidence window in seconds
    'data_cols' : copy_cols # columns to copy over into output table
    }

    # run build_evt
    build_evt(f_in, f_out, builder_config=builder_config, copy_cols=copy_cols)


def show_tcm(f_tcm):
    """ read back the TCM we just made """

    par_list = [
        'eventnumber', 'channel', 'tcm_sec',
           'tcm_dt', 'ix_evt', 'ix_hit']

    dfs = lh5.load_dfs(f_tcm, par_list, 'events')

    pd.set_option('display.max_rows', 200)
    print(dfs[:200])



if __name__=='__main__':
    main()
