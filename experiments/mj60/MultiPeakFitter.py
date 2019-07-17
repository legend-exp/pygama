import ROOT
import numpy as np
import json, os

def main():

    do_fit()

def do_fit():

    with open("runDB.json") as f:
        runDB = json.load(f)
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    # set up I/O files and templates
    histfile = "{}/run_280_329_hist.root".format(meta_dir)
    histname = "run_280_329_hist"
    templatefile = "{}/run_280_329_template.root".format(meta_dir)
    templatename = "run_280_329_template"
    outputfile = "{}/run_280_329_fit_results.root".format(meta_dir)

    # identify what peak will be used for normalization
    Enorm = 2614.511 #true energy of peak
    Elow_norm, Ehigh_norm = 6350, 6500 #uncalibrated range of ADC values in which the peak should be found

    # open histogram
    infile = ROOT.TFile.Open(histfile, "READ")
    hist = infile.Get(histname)
    hist.SetDirectory(0)
    infile.Close()

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
    hmc.SetStepSize(0.02)
    hmc.SetStepLength(50)
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

    print('If the fit failed, try it a few more times. The steps in the fit are based on choosing a random seed, and this can often lead to it failing one time and succeeding another.')


if __name__ == '__main__':
        main()

