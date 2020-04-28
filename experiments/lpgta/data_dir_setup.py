#!/usr/bin/env python3
"""
Utility to create data directories following the LEGEND convention.

Requires:
    a metadata directory with a config file for this experiment
        export LEGEND_META=/some/other/directory
    an env variable with the experiment name (LPGTA, L200A, etc.):
        export LPGTA_DATA=/some/directory

Source:
https://docs.legend-exp.org/index.php/apps/files/?dir=/LEGEND%20Documents/Technical%20Documents/Analysis&fileid=9140#pdfviewer
"""
import os, json
from pprint import pprint
from pathlib import Path

test_mode = False
if test_mode:
    print('Test mode, not creating directories')
else:
    print('Creating directories ...')

# define base data directory and load primary config file
datadir = os.path.expandvars('$LPGTA_DATA')
metadir = os.path.expandvars('$LEGEND_META/analysis/LPGTA/')
with open(metadir + "LPGTA.json") as f:
    config = json.load(f)
    
# get a list of run types we want to include
run_types = config['run_types']

# get a list of subsystems (data takers)
subsystems = []
ch_groups = config['daq_to_raw']['ch_groups']
for grp in ch_groups:
    if 'g{' in grp:
        clo, chi = ch_groups[grp]['ch_range']
        for ch in range(clo, chi+1): # inclusive
            subsystems.append(f'g{ch:0>3d}')
    else:
        subsystems.append(grp)
         
# create hit-level directories
for hdir in config['hit_dirs']:
    for sysn in subsystems:
        for rt in run_types:
            fpath = f'{datadir}/{hdir}/{sysn}/{rt}'
            if not test_mode:
                Path(fpath).mkdir(parents=True, exist_ok=True)
            print(fpath)
    
# create event-level directories
for edir in config['event_dirs']:
    for rt in run_types:
        fpath = f'{datadir}/{edir}/{rt}'
        if not test_mode:
            Path(fpath).mkdir(parents=True, exist_ok=True)
        print(fpath)
