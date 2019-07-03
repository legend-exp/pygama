#!/usr/bin/env python3

#%matplotlib inline
# %load_ext snakeviz
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 7] # make a bigger default figure
plt.rcParams['font.size'] = 18

import sys
import json
import numpy as np
import pandas as pd
from pprint import pprint
from pygama import DataSet
# from pygama.utils import set_plot_style
# set_plot_style('root')
# load file
db_file = "testDB.json"
with open(db_file) as f:
    testDB = json.load(f)

print("-- Top-level information -- ")
for key in testDB:
    if not isinstance(testDB[key], dict):
        print(key, ":", testDB[key])

print("-- Data set definitions -- ")
pprint(testDB["ds"])

try:
    xrun = int(sys.argv[1])
except:
    print("You have to give a run number as argument!")
    exit(0)
ds = DataSet(run=xrun, md=db_file, v=True) # can also use a list of run number

# print some of the DataSet attributes
print("raw dir : ", ds.raw_dir)
print("tier dir : ", ds.tier_dir)
print("t1 file prefix :", ds.t1pre)
print("t2 file prefix :", ds.t2pre)
print("current run list :", ds.runs)
print("current file paths :")
pprint(ds.paths)

print("IF YOUR t0_path IS EMPTY, CROSS-CHECK $DATADIR AND FILE NAMING")

"""
Show waveforms from the Tier 1 file.
NOTE: pygama.DataSet has a convenience function "get_t1df" but is undeveloped.
If there are too many waveforms, we have to use a lot of memory.
For now, let's show an example of accessing the file directly with pandas.read_hdf.
"""
# t1df = ds.get_t1df() # not ready for use yet

def getWFEvent(index, wfs, channels):
    current = [0] * len(channels)
    i = 0
    chIDs = t1df["channel"].values
    wfList = []
    while True:
        try:
            indexx = channels.index(chIDs[i])
        except ValueError as e:
            continue
        if current[indexx] == index:
            wfList.append(wfs[i])
        current[indexx] += 1
        i += 1
        if len(wfList) == len(channels):
            return wfList

# remind ourselves where the file is stored using DataSet
# we know it's run 204 already, but use DataSet's members anyway
# pprint(ds.paths)
run = ds.runs[0]
t1_file = ds.paths[run]["t1_path"]

# remind ourselves the name of the HDF5 group key using the DB.
# pprint(testDB['build_options'])
t1_key = testDB['build_options']['conf1']['tier0_options']['digitizer']

# load a small dataframe
t1df = pd.read_hdf(t1_file, stop=200, key=t1_key)
t1df.reset_index(inplace=True) # required step -- until we fix pygama 'append' bug

print("Tier 1 DataFrame columns:")
print(t1df.columns)

# scrub the non-wf columns and create a 2d numpy array
icols = []
for idx, col in enumerate(t1df.columns):
    if isinstance(col, int):
        icols.append(col)
wfs = t1df[icols].values

#prepare the x values for plotting
#np.arange: start, stop, step
ts = np.arange(0, len(wfs[0]), 1)


# i don't get any interactive plotting to work here on this shitty 
# server, so I just use pdf files.
import matplotlib
matplotlib.use('pdf')
f = plt.figure()

# rendering depends on options in ~/.config/matplotlib/matplotlibrc
# one waveform
plt.plot(ts, wfs[0])


# 50 waveforms
#for row in wfs[:10]:
#    plt.plot(ts, row)

#plt.show()

f.savefig("temp.pdf")

import os
os.system("evince temp.pdf &")

print("plotting now, using pdf...")
print("Use next for next WF or exit for closing the program")
index = 0

while True:
    user = input("WF viewer> ")
    if user == "next":
        plt.clf()   #have to clear the figure before the next plot appears
        wfList = getWFEvent(index, wfs, [0, 1, 2])
        for i, ch in enumerate(wfList):
            plt.plot(ts, wfList[i])
        f.savefig("temp.pdf") #overwrite; evince should be intelligent 
                                #enough to re-print
        index += 1
    elif user == "exit" or user == "quit" or user == ".q":
        break


#input("Press Enter to continue...")











