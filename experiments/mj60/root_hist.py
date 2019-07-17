import root_numpy
import ROOT
import numpy as np
import pandas as pd
import h5py
import sys
import json
import os

# this code is to make a root histogram out of the t2 data from pygama.
# this is necessary to use the MultiPeakFitter.
# you will need to change things below in terms of titling things and the inputting t2 file of interest.

with open("runDB.json") as f:
    runDB = json.load(f)
tier_dir = os.path.expandvars(runDB["tier_dir"])
meta_dir = os.path.expandvars(runDB["meta_dir"])

df = pd.read_hdf('{}/t2_run280-329.h5'.format(tier_dir))

xlo, xhi, xpb = 0, 10000, 1.0
nbins = int((xhi-xlo)/xpb)

hist, bins = np.histogram(np.array(df['e_ftp']), nbins, (xlo, xhi))

hist = np.array(hist, dtype=np.float)

run_280_329_hist = ROOT.TH1F("run_280_329_hist", "", nbins, xlo, xhi)
root_numpy.array2hist(hist, run_280_329_hist, errors=None)
run_280_329_hist.Draw()

out_hist_file = ROOT.TFile.Open("{}/run_280_329_hist.root".format(meta_dir), "RECREATE")
out_hist_file.cd()
run_280_329_hist.Write()
out_hist_file.Close()

