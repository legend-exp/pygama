#!/usr/bin/env python3
import os, json
from pprint import pprint
"""
Simple utiltity to create required data folders for LPGTA.json
Requires a base folder and environment variable:
    export LPGTADIR=/some/directory

Source:
https://docs.legend-exp.org/index.php/apps/files/?dir=/LEGEND%20Documents/Technical%20Documents/Analysis&fileid=9140#pdfviewer
"""
with open('LPGTA.json') as f:
    config = json.load(f)
    
tiers = ['raw','dsp','hit']

# datatypes = 

ch_groups = config['daq_to_raw']['ch_groups']

# filename_info_mods

folder_names = []
for group in ch_groups:
    
    if 'g{' in group:
        ge_folders = [f'g{ch{
        folder_name.extend([f'g{ch:0>3d}' for ch in 
    
    ch_range = ch_groups['ch_range']
    
    # separate ged's gNNN and pulsers pNNN by detector
    # folder_name = 
    
        