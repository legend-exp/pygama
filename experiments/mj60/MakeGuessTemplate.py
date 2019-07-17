import ROOT
import numpy as np
import json, os

def main():

    make_template()
    draw_template()

def make_template():

    with open("runDB.json") as f:
        runDB = json.load(f)
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    fout = ROOT.TFile("{}/run_280_329_template.root".format(meta_dir), "RECREATE")
    fitter = ROOT.GATMultiPeakFitter("run_280_329_template")

    # add regions and peaks
    x, x[0] = ROOT.vector('double')(1), 583.187
    fitter.AddRegion(570, 595, x)
    x, x[0] = ROOT.vector('double')(1), 727.330
    fitter.AddRegion(718, 737, x)
    x, x[0] = ROOT.vector('double')(1), 860.557
    fitter.AddRegion(850, 870, x)
    x, x[0] = ROOT.vector('double')(1), 2614.511
    fitter.AddRegion(2580, 2650, x)

    # set initial peak amplitudes. This is best done by eye
    x, x[0], x[1], x[2], x[3] = ROOT.vector('double')(4), 0.71, 0.21, 0.135, 0.9
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kAmp, ROOT.GATMultiPeakFitter.kFree, x)
    #fitter.LimitPar(ROOT.GATMultiPeakFitter.kAmp, 0,  0.0, 5)   

    # set the functions to use for each of the peakshape parameters, as a function of energy
    x, x[0], x[1] = ROOT.vector('double')(2), 0.3, 1.000
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kMu, ROOT.GATMultiPeakFitter.kLinear, x)
    x, x[0], x[1], x[2] = ROOT.vector('double')(3), 0.04, 0.03, 0.00095
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kSig, ROOT.GATMultiPeakFitter.kRootQuad, x)
    x, x[0] = ROOT.vector('double')(1), 0.3
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kFt, ROOT.GATMultiPeakFitter.kConst, x)
    x, x[0], x[1] = ROOT.vector('double')(2), 0.0, 0.0004
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kTau, ROOT.GATMultiPeakFitter.kLinear, x)
    fitter.LimitPar(ROOT.GATMultiPeakFitter.kFt, 0,  0.0, 1.0)

    # effectively no high tail
    x, x[0] = ROOT.vector('double')(1), 0.1
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kFht, ROOT.GATMultiPeakFitter.kConst, x)
    x, x[0] = ROOT.vector('double')(1), 0.5
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kTauHT, ROOT.GATMultiPeakFitter.kConst, x)
    fitter.FixPar(ROOT.GATMultiPeakFitter.kFht, 0, 0)
    fitter.FixPar(ROOT.GATMultiPeakFitter.kTauHT, 0, 0.5)
    x, x[0], x[1], x[2] = ROOT.vector('double')(3), 750, 0.03, -0.88
    fitter.SetParFunction(ROOT.GATMultiPeakFitter.kHs, ROOT.GATMultiPeakFitter.kStepHeightFun, x)
    fitter.FixPar(ROOT.GATMultiPeakFitter.kHs, 2, -0.88)

    # background parameters; similar to amplitude, this is best done by eye
    fitter.SetBGPars(0, 0.0792,-0.0003, 0)
    fitter.SetBGPars(1, 0.0532, -0.0001, 0)
    fitter.FixBGPar(1, 2, 0)
    fitter.SetBGPars(2, 0.0409, -0.0001, 0)
    fitter.FixBGPar(2, 1, 0)
    fitter.FixBGPar(2, 2, 0)
    fitter.SetBGPars(3, 0.0009, 0, 0)
    fitter.FixBGPar(3, 1, 0)
    fitter.FixBGPar(3, 2, 0)

    fitter.Write()
    fout.Close()
    #ROOT.gSystem.Exit(0)


def draw_template():

    with open("runDB.json") as f:
        runDB = json.load(f)
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    # set up I/O files and templates
    histfile = "{}/run_280_329_hist.root".format(meta_dir)
    histname = "run_280_329_hist"
    templatefile = "{}/run_280_329_template.root".format(meta_dir)
    templatename = "run_280_329_template"

    # identify what peak will be used for normalization
    Enorm = 2614.511 #true energy of peak
    Elow_norm, Ehigh_norm = 6350, 6500 #uncalibrated range of ADC values in which the peak should be found

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
    fitter.ScaleEnergy(scaleenergy)
    fitter.ScaleAmps(scaleamps)

    fitter.DrawRegions()


if __name__ == '__main__':
        main()

