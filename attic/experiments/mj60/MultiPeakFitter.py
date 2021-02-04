import ROOT
import numpy as np
import json, os, sys

with open("runDB.json") as f:
    runDB = json.load(f)
meta_dir = os.path.expandvars(runDB["meta_dir"])

if(len(sys.argv) != 4):
    print('Usage: MultiPeakFitter.py [input root histogram file] [input template guesses file] [output fit results file]')
    sys.exit()

def main():

    do_fit()

def do_fit():

    # set up I/O files and templates
    histfile = "{}/{}".format(meta_dir,sys.argv[1])
    histname = "root_hist"
    templatefile = "{}/{}".format(meta_dir,sys.argv[2])
    templatename = "fit_template"
    outputfile = "{}/{}".format(meta_dir,sys.argv[3])

    # identify what peak will be used for normalization
    Enorm = 356.013 #true energy of peak
    Elow_norm, Ehigh_norm = 820, 860 #uncalibrated range of ADC values in which the peak should be found

    # open histogram
    infile = ROOT.TFile.Open(histfile, "READ")
    hist = infile.Get(histname)
    hist.SetDirectory(0)
    infile.Close()
    #hist.Draw()

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

    # set output file
    fout = ROOT.TFile(outputfile, "RECREATE")
    hist.Write()
    fitter.Write("initialFitter")

    # perform HMC burn-in to refine parameter guess 
    hmc = ROOT.GATHybridMonteCarlo()
    hmc.SetNLLFunc(fitter.SetPoissonLLFCN())
    hmc.SetOutputFile(fout)
    hmc.SetRecordPaths()

    x = ROOT.vector('double')()
    for i in range(fitter.NPar()):
        x.push_back(fitter.Parameters()[i])

    hmc.SetParameters(x)

    # Set the HMC parameters. Step size is leapfrog step size. Step length is number of leapfrog steps for each MCMC step. NSteps is number of MCMC steps. Adapt step size and parameter scales will automatically adjust step size and individual parameter scales for each step
    hmc.SetStepSize(0.01)
    hmc.SetStepLength(100)
    hmc.SetNSteps(200)
    hmc.SetAdaptStepSize()
    hmc.SetAdaptParScales()
    hmc.SetLimits(fitter.GetParLimits())

    # output the random seed to be used, this might be useful for troubleshooting
    print('Random Seed: {}'.format(hmc.GetCurrentSeed()))
    hmc.DoMCMC(1)
    fitter.SetParameters(hmc.GetLikeliestPars().data())
    fitter.Write("hmcResults")

    # perform a minuit fit using the most likely parameters found during the HMC as the initial parameters. Use minos error estimation
    fitter.FitToHists(0,1,0,1)
    fitter.Write("results")
    hist.Draw()
    hist.GetXaxis().SetTitle("ADC")
    hist.GetYaxis().SetTitle("Counts")
    fitter.DrawRegions()

    A = 1/(fitter.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[1])
    B = -(fitter.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[0])/(fitter.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[1])
    mu0 = fitter.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[0]
    mu1 = fitter.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[1]
    mu2 = fitter.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[2]
    print(A)
    print(B)
    print(mu0)
    print(mu1)
    print(mu2)

    print('If the fit failed, try it a few more times. The steps in the fit are based on choosing a random seed, and this can often lead to it failing one time and succeeding another. If the fit continues to fail, look into limiting some of your parameters, such as your background terms. Sometimes the fit can try to make the background negative, and this can cause it to fail. If you continue to have issues, you can look into the hmcResults to figure out where the fit is going wrong.')

if __name__ == '__main__':
        main()

