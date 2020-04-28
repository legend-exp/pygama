#!/usr/bin/env python3
"""
Define data storage directories for a LEGEND experiment (LPGTA, L200A, etc.)
Requires an env variable pointing to the base data directory, and another 
pointing to the legend-metadata directory.

$LPGTA_DATA
+-- LPGTA | L200A
    |-- raw | dsp | hit
    |   +-- gNNN | spms | pmts | auxs | pNNN
    |       +-- phy | cal | trn | lac | tst
    |           + cycle files
    |-- evt|tmap
    |   +-- phy | cal | trn | lac | tst
    |       + run files
    
$LEGEND_META
+-- legend_metadata
    |-- analysis
    |   +-- LDQTA | LPGTA | L200A
    |       + LPGTA.json (main config)
    |       + runDB.json (run index)
    |       +-- ged-dsp | ged-ene | ged-aoe
    |           + run files
    |       +-- chmap
    |           + channel map files
    |-- hardware
    |   +-- detectors
    |       + attribute files
    |   +-- materials
    
Source:
https://docs.legend-exp.org/index.php/apps/files/?dir=/LEGEND%20Documents/Technical%20Documents/Analysis&fileid=9140#pdfviewer
"""
import os, json
from pprint import pprint
import itertools
from pathlib import Path

datadir = os.environ.get('LPGTA_DATA')
metadir = os.environ.get('LEGEND_META') + '/analysis/LPGTA'

with open(f'{metadir}/LPGTA.json') as f:
    config = json.load(f)
    
run_types = config['run_types']
hit_dirs = config['hit_dirs']
event_dirs = config['event_dirs']

# get the active ge detectors from the main config file
ch_groups = []
for key, val in config['daq_to_raw']['ch_groups'].items():
    if 'g{' in key:
        clo, chi = val['ch_range']
        geds = [f'g{ch:0>3d}' for ch in range(clo, chi+1)]
        ch_groups.extend(geds)
    else:
        ch_groups.append(key)

# create directories for hit-level data
for base, rtype, chan in itertools.product(hit_dirs, run_types, ch_groups):
    dirname = f'{datadir}/{base}/{rtype}/{chan}'
    Path(dirname).mkdir(parents=True, exist_ok=True)
    print(dirname)
        
# create directories for event-level data
for dir in config['event_dirs']:
    dirname = f'{datadir}/{dir}'
    Path(dirname).mkdir(parents=True, exist_ok=True)
    print(dirname)
