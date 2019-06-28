#!/usr/bin/env python3

#%matplotlib inline
# %load_ext snakeviz
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 7] # make a bigger default figure
plt.rcParams['font.size'] = 18

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

#-- data set ---

# can declare the DataSet either by "ds" number, or run numbers.

# ds = DataSet(0, 3, md=db_file, v=True) # can use a range of datasets

ds = DataSet(run=111, md=db_file, v=True) # can also use a list of run numbers

#somehow, the program doesn't find a file named testSCARF137, but
# 2019-3-18-BackgroundRun204 seems to be ok ...

# print some of the DataSet attributes
print("raw dir : ", ds.raw_dir)
print("tier dir : ", ds.tier_dir)
print("t1 file prefix :", ds.t1pre)
print("t2 file prefix :", ds.t2pre)
print("current run list :", ds.runs)
print("current file paths :")
pprint(ds.paths)

print("IF YOUR t0_path IS EMPTY, CROSS-CHECK $DATADIR AND FILE NAMING")










