#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
import pandas as pd
from pprint import pprint
from collections import OrderedDict

from pygama import DataGroup
from pygama.io.daq_to_raw import daq_to_raw
from pygama.io.raw_to_dsp import raw_to_dsp
from pygama.io.ch_group import *


def main():
    doc="""
    LPGTA data processing routine.
    You may need to set these environment variables, check your LPGTA.json file.
      * $LPGTA_DATA : base data directory
      * $LEGEND_META : the legend-metadata repository

    By default, we move output files into place in the LH5 directory tree, but
    there is also an option to leave it in place in CWD.

    TODO: parallelize, submit processing jobs
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    # declare datagroup
    arg('--dg', action=st, help='load datagroup')
    arg('--q', type=str, help='datagroup query (AND-ed with the date query)')
    arg('--date', type=str, help='date query (AND-ed with the datagroup query)')

    # routines
    arg('--d2r', action=st, help='run daq_to_raw')
    arg('--r2d', action=st, help='run raw_to_dsp')

    # options
    arg('-o', '--over', action=st, help='overwrite existing files')
    arg('-n', '--nwfs', type=int, help='limit num. waveforms')
    arg('-v', '--verbose', action=st, help='verbose mode')
    arg('-l', '--bl', type=int, default=3200, help='buffer length for chunked reads for DSP')
    arg('-w', '--bw', type=int, default=8, help='buffer width for DSP')
    arg('-c', '--cwd', action=st, help="save output to current directory")
    args = par.parse_args()

    # print out the arguments for log files
    if args.verbose:
        print('Arguments:', args)

    # -- set options --
    nwfs = args.nwfs if args.nwfs is not None else np.inf

    print('Processing settings:'
          '\n$LPGTA_DATA =', os.environ.get('LPGTA_DATA'),
          '\n$LEGEND_META =', os.environ.get('LEGEND_META'),
          f'\n  overwrite? {args.over}'
          f'\n  limit wfs? {nwfs}')

    # -- run routines --
    query = "YYYYmmdd == '" + args.date + "'" if args.date else args.q
    if args.date and args.q:
        query += ' and ' + args.q
    if args.dg or args.d2r or args.r2d: dg = load_datagroup(query)
    if args.d2r: d2r(dg, args.over, nwfs, args.verbose, args.cwd)
    if args.r2d: r2d(dg, args.over, nwfs, args.verbose, args.cwd, args.bl, args.bw)


def load_datagroup(query=None):
    """
    """
    dg = DataGroup('LPGTA.json')
    dg.load_df('LPGTA_fileDB.h5')

    # NOTE: for now, we have to edit this line to choose which files to process
    # process one big cal file (64 GB)
    #query = "run==18 and YYYYmmdd == '20200302' and hhmmss == '184529'"
    if query is not None: dg.fileDB.query(query, inplace=True)

    print('files to process:')
    print(dg.fileDB)

    # can add other filters here
    #dg.fileDB = dg.fileDB[:2]

    return dg


def cmap_to_ch_groups(cmap, sys_per_ged=False):
    """
    convert LEGEND-style json channel map in file $LEGEND_META/cmap into
    pygama ch_groups dictionary parsable by daq_to_raw

    each ged channel gets its own group

    all spm channels are in the spms groups

    all other channeles go in auxs for now

    if sys_per_ged is True, every ged channel gets output to its own file
    """
    path = os.environ.get('LEGEND_META') + '/hardware/channelmaps/' + cmap
    if not os.path.exists(path):
        print("could not find channel map", cmap)
        return {}
    with open(path) as f:
        cmap = json.load(f)

    ch_groups = {}

    for record in cmap.values():
        adc = None
        if 'trace' in record: adc = int(record['trace'])
        elif 'adc' in record: adc = int(record['adc'])
        if adc == None:
            print("cmap", cmap, "- Couldn't find adc channel in record...")
            print(record)

        if record['type'] == 'ged':
            group = f'g{adc:0>3d}'
            ch_groups[group] = {}
            if sys_per_ged: ch_groups[group]['system'] = group
            else: ch_groups[group]['system'] = 'geds'
            ch_groups[group]['ch_list'] = [ adc ]

        elif record['type'] == 'spm':
            if 'spms' not in ch_groups:
                ch_groups['spms'] = {}
                ch_groups['spms']['system'] = 'spms'
                ch_groups['spms']['ch_list'] = []
            ch_groups['spms']['ch_list'].append(adc)

        elif record['type'] == 'none': continue

        else:
            print("cmap", cmap, "- Unknown record type", record['type'], ": ADC = ", adc)
            print(record)

    return ch_groups


def d2r(dg, overwrite=False, nwfs=None, vrb=False, cwd=False):
    """
    run daq_to_raw on the current DataGroup
    """
    # print(dg.fileDB)
    # print(dg.fileDB.columns)

    # subs = ['geds'] # TODO: ignore other datastreams
    # chans = ['g035', 'g042'] # TODO: select a subset of detectors

    print(f'Processing {dg.fileDB.shape[0]} files ...')

    for i, row in dg.fileDB.iterrows():

        # set up I/O paths
        f_daq = f"{dg.daq_dir}/{row['daq_dir']}/{row['daq_file']}"
        f_raw = f"{dg.lh5_dir}/{row['raw_path']}/{row['raw_file']}"
        if cwd: f_raw = f"{row['raw_file']}"

        subrun = row['cycle'] if 'cycle' in row else None
        systems = dg.subsystems

        # load cmap if there is one
        if 'cmap' in row:
            cmap = row['cmap']
            if ('daq_to_raw' in dg.config and
                'ch_groups' in dg.config['daq_to_raw'] and
                'FlashCamEventDecoder' in dg.config['daq_to_raw']['ch_groups']):
                # we are going to overwrite the "default" in config with
                # the cmap, so give a nice warning
                print('overwriting default ch_groups in config file with channel map info in', cmap)
            else:
                if 'daq_to_raw' not in dg.config:
                    dg.config['daq_to_raw'] = {}
                if 'ch_groups' not in dg.config['daq_to_raw']:
                    dg.config['daq_to_raw']['ch_groups'] = {}
            ch_groups = cmap_to_ch_groups(cmap)
            dg.config['daq_to_raw']['ch_groups']['FlashCamEventDecoder'] = ch_groups
            systems = get_list_of('system', ch_groups)
            systems.append('auxs') # for FC status info

        daq_to_raw(f_daq, f_raw, config=dg.config, systems=systems, verbose=vrb,
                   n_max=nwfs, overwrite=overwrite, subrun=subrun)#, chans=chans)


def r2d(dg, overwrite=False, nwfs=None, vrb=False, cwd=False, buffer_len=3200, block_width=8):
    """
    """
    # print(dg.fileDB)
    # print(dg.fileDB.columns)

    with open(f'{dg.experiment}_dsp.json') as f:
        dsp_config = json.load(f, object_pairs_hook=OrderedDict)

    for i, row in dg.fileDB.iterrows():

        f_raw = f"{dg.lh5_dir}/{row['raw_path']}/{row['raw_file']}"
        f_dsp = f"{dg.lh5_dir}/{row['dsp_path']}/{row['dsp_file']}"
        if cwd: f_dsp = f"{row['dsp_file']}"

        if "sysn" in f_raw:
            tmp = {'sysn' : 'geds'} # hack for lpgta
            f_raw = f_raw.format_map(tmp)
            f_dsp = f_dsp.format_map(tmp)

        raw_to_dsp(f_raw, f_dsp, dsp_config, n_max=nwfs, verbose=vrb,
                   overwrite=overwrite, buffer_len=buffer_len, block_width=block_width)


if __name__=="__main__":
    main()
