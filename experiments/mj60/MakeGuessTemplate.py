import ROOT
import numpy as np
import json, os
import sys


if(len(sys.argv) != 3):
    print('Usage: MakeGuessTemplate.py [input root histogram] [output root template]')
    sys.exit()

with open("runDB.json") as f:
    runDB = json.load(f)
meta_dir = os.path.expandvars(runDB["meta_dir"])


def main():

    make_template()
    draw_template()

def make_template():

    fout = ROOT.TFile("{}/{}".format(meta_dir,sys.argv[2]), "RECREATE")
    fitter = ROOT.GATMultiPeakFitter("fit_template")

    # add regions and peaks
    x, x[0] = ROOT.vector('double')(1), 351.932
    fitter.AddRegion(345, 358, x)
    x, x[0] = ROOT.vector('double')(1), 583.187
    fitter.AddRegion(570, 595, x)

    # set initial peak amplitudes. This is best done by eye
    x, x[0], x[1] = ROOT.vector('double')(2), 0.34, 0.26 
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kAmp, ROOT.GATMultiPeakFitter.kFree, x)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kAmp, 0, 0.01, 192420)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kAmp, 1, 0.01, 192420)

    # set the functions to use for each of the peakshape parameters, as a function of energy    
    x, x[0], x[1] = ROOT.vector('double')(2), 0, 1.000
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kMu, ROOT.GATMultiPeakFitter.kLinear, x)
    fitter.LimitPar(ROOT.GATMultiPeakFitter.kMu, 0, -5, 5)
    fitter.LimitPar(ROOT.GATMultiPeakFitter.kMu, 1, 0.1, 5)
 
    x, x[0], x[1], x[2] = ROOT.vector('double')(3), 0.09, 0.04, 0.00095
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kSig, ROOT.GATMultiPeakFitter.kRootQuad, x)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kSig, 0, -30, 30)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kSig, 1, -30, 30)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kSig, 2, -30, 30)
    
    x, x[0] = ROOT.vector('double')(1), 0.3
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kFt, ROOT.GATMultiPeakFitter.kConst, x)
    fitter.LimitPar(ROOT.GATMultiPeakFitter.kFt, 0,  0.0, 5.0)

    x, x[0], x[1] = ROOT.vector('double')(2), 0, 0.0004
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kTau, ROOT.GATMultiPeakFitter.kLinear, x)

    # effectively no high tail
    x, x[0] = ROOT.vector('double')(1), 0
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kFht, ROOT.GATMultiPeakFitter.kConst, x)
    fitter.FixPar(ROOT.GATMultiPeakFitter.kFht, 0, 0)    

    x, x[0] = ROOT.vector('double')(1), 0.5
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kTauHT, ROOT.GATMultiPeakFitter.kConst, x)
    fitter.FixPar(ROOT.GATMultiPeakFitter.kTauHT, 0, 0.5)
    
    x, x[0], x[1], x[2] = ROOT.vector('double')(3), 750, 0.18, -0.88
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kHs, ROOT.GATMultiPeakFitter.kStepHeightFun, x)
    fitter.LimitPar(ROOT.GATMultiPeakFitter.kHs, 0, -1000, 100000)
    fitter.LimitPar(ROOT.GATMultiPeakFitter.kHs, 1, -1000, 100)
    fitter.FixPar(ROOT.GATMultiPeakFitter.kHs, 2, -0.88)

    # background parameters; similar to amplitude, this is best done by eye
    fitter.SetBGPars(0, 0.0519,-0.0004, 0)
    fitter.LimitBGPar(0, 0, 0.000001, 10000)
    fitter.LimitBGPar(0, 1, -40, 0)
    fitter.LimitBGPar(0, 2, -10, 0)

    fitter.SetBGPars(1, 0.0202,-0.0001, 0)
    fitter.LimitBGPar(1, 0, 0.000001, 4000)
    fitter.FixBGPar(1, 1,  0)
    fitter.FixBGPar(1, 2, 0)

    fitter.Write()
    fout.Close()
    #ROOT.gSystem.Exit(0)


def draw_template():

    # set up I/O files and templates
    histfile = "{}/{}".format(meta_dir,sys.argv[1])
    histname = "root_hist"
    templatefile = "{}/{}".format(meta_dir,sys.argv[2])
    templatename = "fit_template"

    # identify what peak will be used for normalization
    Enorm = 583.187 #true energy of peak
    Elow_norm, Ehigh_norm = 1415, 1450 #uncalibrated range of ADC values in which the peak should be found

    # open histogram
    infile = ROOT.TFile.Open(histfile, "READ")
    hist = infile.Get(histname)
    hist.SetDirectory(0)
    infile.Close()
    hist.Draw()

    # get fit template
    infile = ROOT.TFile.Open(templatefile, "READ")
    fitter = infile.Get(templatename)
    infile.Close()
    fitter.SetHists(hist)

    # scale energy / amplitudes
    hist.GetXaxis().SetRangeUser(Elow_norm, Ehigh_norm)
    maxbin = hist.GetMaximumBin()
    scaleenergy = hist.GetBinCenter(maxbin)/Enorm
    scaleamps = hist.Integral()*hist.GetBinWidth(maxbin)
    hist.GetXaxis().SetRange()
    # scale mu, sig, and tau pars
    fitter.ScaleEnergy(scaleenergy)
    # scale all amp and BG pars
    fitter.ScaleAmps(scaleamps)

    print("scaleenergy = {}".format(scaleenergy))
    print("scaleamps = {}".format(scaleamps))

    fitter.DrawRegions()
  

if __name__ == '__main__':
        main()

