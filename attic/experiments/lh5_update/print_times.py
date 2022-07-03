#!/usr/bin/env python3
import os
import time
import h5py
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

import pygama.analysis.histograms as pgh

def main():
    """
    """
    hf = h5py.File("/lfs/l1/legend/users/kermaidy/pygama_test/char_data/tier1/char_data-V00048B-th_HS2_lat_psa-run0001-210128T143746_tier1.lh5")
    
    # General info
    print(hf.keys())
    print(hf['raw'].keys())
    print(hf['stat'].keys())
    ievt = hf['/raw/ievt']

    # Statustime fields
    statustime  = hf['/stat/statustime'];
    cputime     = hf['/stat/cputime'];
    startoffset = hf['/stat/startoffset'];

    # Timestamp fields
    runtime           = hf['/raw/runtime']
    timestamp         = hf['/raw/timestamp']
    timestamp_pps     = hf['/raw/ts_pps']
    timestamp_ticks   = hf['/raw/ts_ticks']
    timestamp_maxticks= hf['/raw/ts_maxticks']

    # Timeoffset fields
    to_master_sec = hf['/raw/to_master_sec']
    to_mu_sec     = hf['/raw/to_mu_sec']
    to_mu_usec    = hf['/raw/to_mu_usec']
    to_start_sec  = hf['/raw/to_start_sec']
    to_start_usec = hf['/raw/to_start_usec']
    to_abs_mu_usec= hf['/raw/to_abs_mu_usec']
    to_dt_mu_usec = hf['/raw/to_dt_mu_usec']

    # Deadregion fields
    dr_start_pps  = hf['/raw/dr_start_pps']
    dr_start_ticks= hf['/raw/dr_start_ticks']
    dr_stop_pps   = hf['/raw/dr_stop_pps']
    dr_stop_ticks = hf['/raw/dr_stop_ticks']
    dr_maxticks   = hf['/raw/dr_maxticks']
    deadtime      = hf['/raw/deadtime']

    stop=10 # Number of events to parse

    k=0
    print(" ")
    print("Statustime fields: ")
    for st,ct,so in zip(statustime,cputime,startoffset):
      print("  ",st,"s - ",ct,"s - ",so,"s - ",st-so,"s")
      k=k+1
      if (k>stop): break

    k=0
    print(" ")
    print("Timestamp fields: ")
    for evt,ts_pps,ts_ticks,ts_maxticks,rt,ts in zip(ievt,timestamp_pps,timestamp_ticks,timestamp_maxticks,runtime,timestamp):
      print("  ",evt,ts_pps,"s - ",ts_ticks," - ",ts_maxticks," - ",rt,"s - ",ts,"s")
      k=k+1
      if (k>stop): break

    k=0
    print(" ")
    print("Timeoffset fields: ")
    for evt,to0,to1,to2,to3,to4,to5,to6 in zip(ievt,to_master_sec,to_mu_sec,to_mu_usec,to_start_sec,to_start_usec,to_abs_mu_usec,to_dt_mu_usec):
      print("  ",evt,to0,"s - ",to1,"s",to2,"us - ",to3,"s",to4,"us - ",to5,"us - ",to6,"us")
      k=k+1
      if (k>stop): break

    k=0
    print(" ")
    print("Deadregion fields: ")
    for evt,rt,dr0,dr1,dr2,dr3,dr4,dr5 in zip(ievt,runtime,dr_start_pps,dr_start_ticks,dr_stop_pps,dr_stop_ticks,dr_maxticks,deadtime):
      print("  ",evt,dr0,"s",dr1," - ",dr2,"s",dr3," - ",dr4," - ",dr5,"s",dr5/rt*100.,"%")
      k=k+1
      if (k>stop): break

    print(" ")
    
    hf.close()

if __name__=="__main__":
    main()
