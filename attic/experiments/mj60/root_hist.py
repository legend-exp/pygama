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

if(len(sys.argv) != 3):
    print('Usage: root_hist.py [run number] [output root file name]')
    sys.exit()

with open("runDB.json") as f:
    runDB = json.load(f)
tier_dir = os.path.expandvars(runDB["tier_dir"])
meta_dir = os.path.expandvars(runDB["meta_dir"])

df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))

xlo, xhi, xpb = 0, 10000, 0.5
nbins = int((xhi-xlo)/xpb)

hist, bins = np.histogram(np.array(df['e_ftp']), nbins, (xlo, xhi))

# The MultiPeakFitter normalizes to one's peak of choice. For Ba I am normalizing to the 356.013 keV peak, which is between 820, and 860 ADC. The binning is 0.5 ADC, hence the factor in the division that sums the histogram counts from entry 1640 (820/0.5) to 1720 (860/0.5). 
hist = np.array(hist, dtype=np.float)/(xpb*sum(hist[1640:1720]))

root_hist = ROOT.TH1F("root_hist", "", nbins, xlo, xhi)
root_numpy.array2hist(hist, root_hist, errors=None)
root_hist.Draw()

out_hist_file = ROOT.TFile.Open("{}/{}".format(meta_dir,sys.argv[2]), "RECREATE")
out_hist_file.cd()
root_hist.Write()
out_hist_file.Close()

