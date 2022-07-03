import ROOT
import time
import numpy as np
import pandas as pd
import json
import sys
import os
import pygama.analysis.histograms as pgh

# this code fits background peaks to the Response Function of a germanium detector.
# the code is very sensitive to guesses.
# if an error is returned, your guesses are likely the culprit.
# the guesses are inputs into the peak_fit function defined.

if(len(sys.argv) != 2):
    print('Usage: root_fit_peaks.py [run number]')
    sys.exit()

def peak_fit(real_pk_energy=2614.51, E_type='e_ftp', E_low=6360, E_high=6490, xpb=1.0, mu=6420, sigma=1.05, hstep=0.001, htail=0.02, tau=5, bg0=1, a=4000, plot_name="test.pdf"):
    """
    E_low and E_high are chosen so as to fit a peak by windowing out others.
    xpb is the binning size.
    E-type can be chosen to fit uncalibrated energy, or calibrated energy.
    """

    with open("runDB.json") as f:
        runDB = json.load(f)
    meta_dir = os.path.expandvars(runDB["meta_dir"])
    tier_dir = os.path.expandvars(runDB["tier_dir"])

    if E_type == 'e_cal':
        df =  pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[1]), key="df")
        hist, bins, var = pgh.get_hist(df['e_cal'], range=(E_low,E_high), dx=xpb)
    if E_type == 'e_ftp':
        df = pd.read_hdf("{}/t2_run{}.h5".format(tier_dir,sys.argv[1]))
        hist, bins, var = pgh.get_hist(df['e_ftp'], range=(E_low,E_high), dx=xpb)

    hist = np.append(hist,0)

    guesses = np.array([mu, sigma, hstep, htail, tau, bg0, a])
    hist = np.array(hist, dtype=np.float)
    bins = np.array(bins, dtype=np.float)
    count_errors = hist**.5
    x_errors = xpb/np.sqrt(12) + np.zeros(len(bins))

    graph = ROOT.TGraph(len(hist),bins,hist)
    graphe = ROOT.TGraphErrors(len(hist),np.array(bins, dtype=np.float),np.array(hist, dtype=np.float), np.array(x_errors, dtype=np.float),np.array(hist**0.5, dtype = np.float))
    func = ROOT.TF1("func","(1-[3])*[6] * (1. / [1] / sqrt(2 * pi)) * exp(-(x - [0])**2 / (2. * [1]**2)) + [5] + [6] * [2] * erfc((x - [0]) / ([1] * sqrt(2))) + [6] * [3] * erfc((x - [0]) / ([1] * sqrt(2)) + [1] / ([4] * sqrt(2))) * exp((x - [0]) / [4]) / ((2 * [4] * exp(-([1] / (sqrt(2) * [4]))**2)))" ,2540,2680)
    func.SetParameters(np.array(guesses, dtype=np.float))
    func.SetParName(0,"mu")
    func.SetParName(1,"sigma")
    func.SetParName(2, "hstep")
    func.SetParName(3, "htail")
    func.SetParName(4, "tau")
    func.SetParName(5, "bg0")
    func.SetParName(6, "a")
    graphe.Fit("func","")
    reduced_chi_squared = func.GetChisquare()/func.GetNDF()
    print("Reduced Chi-Square = {}".format(reduced_chi_squared))
    #print(func.GetParameter(0))

    c1 = ROOT.TCanvas("c1","c1",600,400)
    graphe.Draw()
    graphe.GetXaxis().SetLimits(E_low,E_high)
    func.Draw("same")
    graphe.GetYaxis().SetTitle("Counts")
    if E_type == 'e_cal':
        graphe.GetXaxis().SetTitle("Energy (keV)")
        graphe.SetTitle("Calibrated {} keV Peak Fit".format(real_pk_energy))
    if E_type == 'e_ftp':
        graphe.GetXaxis().SetTitle("e_ftp")
        graphe.SetTitle("Uncalibrated {} keV Peak Fit".format(real_pk_energy))
    #legend = ROOT.TLegend(0.1,0.05,0.2,0.5)
    #legend.AddEntry("func","entry 1","l")
    #legend.Draw("same")
    c1.SaveAs("~/Desktop/{}".format(plot_name))

peak_fit()
