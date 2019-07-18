import ROOT
import numpy as np
import json, os, sys

def main():

    get_calibration()

def get_calibration():

    if(len(sys.argv) != 2):
        print('Usage: fitresults.py [fit results root file]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    # input and get the fit results of interest
    fitresultsfile = "{}/{}".format(meta_dir,sys.argv[1])
    fitresultsname = "hmcResults"
    infile = ROOT.TFile.Open(fitresultsfile, "READ")
    results = infile.Get(fitresultsname)
    infile.Close()

    # print out calibration parameter
    A = 1/(results.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[1])
    B = -(results.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[0])/(results.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[1])
    print("E=A*e_ftp+B")
    print("A, B = {}, {}".format(A,B))


if __name__ == '__main__':
        main()
