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
    x, x[0] = ROOT.vector('double')(1), 80.998
    fitter.AddRegion(75, 85, x)
    x, x[0] = ROOT.vector('double')(1), 276.399
    fitter.AddRegion(270, 282, x)
    x, x[0] = ROOT.vector('double')(1), 302.851
    fitter.AddRegion(295, 310, x)
    x, x[0] = ROOT.vector('double')(1), 356.013
    fitter.AddRegion(345, 364, x)
    x, x[0] = ROOT.vector('double')(1), 383.849
    fitter.AddRegion(376, 391, x)

    # set initial peak amplitudes. This is best done by eye
    x, x[0], x[1], x[2], x[3], x[4] = ROOT.vector('double')(5), 0.290, 0.65, 1.10, 3.25, 0.265
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kAmp, ROOT.GATMultiPeakFitter.kFree, x)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kAmp, 0, 0.05, 0.15)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kAmp, 1, 0.10, 0.20)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kAmp, 2, 0.30, 0.40)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kAmp, 3, 1.00, 1.20)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kAmp, 4, 0.10, 0.20)

    # set the functions to use for each of the peakshape parameters, as a function of energy    
    x, x[0], x[1] = ROOT.vector('double')(2), 0, 1.000
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kMu, ROOT.GATMultiPeakFitter.kLinear, x)
    fitter.LimitPar(ROOT.GATMultiPeakFitter.kMu, 0, -5, 5)
    fitter.LimitPar(ROOT.GATMultiPeakFitter.kMu, 1, 0.1, 10)
 
    x, x[0], x[1], x[2] = ROOT.vector('double')(3), 1.2, 0.0093, 0.00083
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kSig, ROOT.GATMultiPeakFitter.kRootQuad, x)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kSig, 0, 0.001, 2)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kSig, 1, 0.0001, 2)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kSig, 2, 0.000001, 2)
    
    x, x[0] = ROOT.vector('double')(1), 0.7
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kFt, ROOT.GATMultiPeakFitter.kConst, x)
    fitter.LimitPar(ROOT.GATMultiPeakFitter.kFt, 0,  0.1, 1)

    x, x[0], x[1] = ROOT.vector('double')(2), 0, .50
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kTau, ROOT.GATMultiPeakFitter.kLinear, x)

    # effectively no high tail
    x, x[0] = ROOT.vector('double')(1), 0.1
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kFht, ROOT.GATMultiPeakFitter.kConst, x)
    #fitter.FixPar(ROOT.GATMultiPeakFitter.kFht, 0, 0)    

    x, x[0] = ROOT.vector('double')(1), 0.5
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kTauHT, ROOT.GATMultiPeakFitter.kConst, x)
    #fitter.FixPar(ROOT.GATMultiPeakFitter.kTauHT, 0, 0.5)
    
    x, x[0], x[1], x[2] = ROOT.vector('double')(3), 0, 0.000003, -0.88
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kHs, ROOT.GATMultiPeakFitter.kStepHeightFun, x)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kHs, 0, -100, 100)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kHs, 1, -10, 10)
    #fitter.FixPar(ROOT.GATMultiPeakFitter.kHs, 2, -0.88*2.357)

    # background parameters; similar to amplitude, this is best done by eye
    fitter.SetBGPars(0, 0.00819,.000050011, 0.00001001)
    #fitter.LimitBGPar(0, 0, 0.00082, 0.01819)
    #fitter.LimitBGPar(0, 1, 0, 0.1)
    #fitter.FixBGPar(0, 2, 0)

    fitter.SetBGPars(1, 0.00452, -0.000005, 0)
    #fitter.LimitBGPar(1, 0, 0.00452, 10000)
    #fitter.LimitBGPar(1, 1, -10, 0)
    fitter.FixBGPar(1, 2, 0)

    fitter.SetBGPars(2, 0.00355, -0.000002, 0)
    #fitter.LimitBGPar(2, 0, 0.00355, 10000)
    #fitter.LimitBGPar(2, 1,  -10, 0)
    fitter.FixBGPar(2, 2, 0)

    fitter.SetBGPars(3, 0.000202, 0.0000102, 0)
    #fitter.LimitBGPar(3, 0, 0.0000102, 0.001)
    #fitter.FixBGPar(3, 1,  0)
    fitter.FixBGPar(3, 2, 0)

    fitter.SetBGPars(4, 0.000862,0, 0)
    #fitter.LimitBGPar(4, 0, 0.0000102, 0.001)
    #fitter.LimitBGPar(4, 1,  0, 0)
    fitter.LimitBGPar(4, 2, 0, 0)


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
    Enorm = 356.013 #true energy of peak
    Elow_norm, Ehigh_norm = 820, 860 #uncalibrated range of ADC values in which the peak should be found

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

