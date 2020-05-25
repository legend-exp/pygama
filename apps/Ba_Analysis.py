#!/usr/bin/env python3.7
import numpy as np
import pandas as pd
import tinydb as db
import matplotlib.pyplot as plt
from scipy.integrate import simps
from pygama import DataSet
import pygama.utils as pgu
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pga
from numpy import diff


"""""
This is a script to fit the different lines of an 133Ba scan.
This script is based on the pygama version from December 2019 and is a bit outdated. 
An update will be done soon

You need to have done a Calibration before and the output must be in the ds.calDB file

The function takes a DataSet (December version) and a t2-level file 
Then a fit on the 60kev line and on the 99/103 keV lines is performed, the
integrals are caluclated and the ratio is determind

A.Zschocke
"""


def Ba_lines(ds, t2, display=False, write_DB=True):

    inf = np.inf

    etype, ecal = "e_ftp", "e_cal"
    e_peak = 0

    # load calibration database file with tinyDB and convert to pandas
    calDB = ds.calDB
    query = db.Query()
    table = calDB.table("cal_pass3").all()
    df_cal = pd.DataFrame(table) 

    # apply calibration from db to tier 2 dataframe
    df_cal = df_cal.loc[df_cal.ds.isin(ds.ds_list)]
    isLin = df_cal.iloc[0]["lin"]
    slope = df_cal.iloc[0]["slope"]
    offset = df_cal.iloc[0]["offset"]
    eraw = t2[etype]
    
    #check for linearity and apply calibration 
    if isLin:
       t2[ecal] = eraw * (eraw * slope + offset) 
    else: 
       t2[ecal] = eraw * (slope + (offset/eraw**2)) 
   

    hE, xE, vE = pgh.get_hist(t2[ecal], range=(345,365), dx=0.08)
   
    a = 150000
    mu = 356
    sigma = 0.3
    tail = 50000
    tau = 0.5
    bkg = 4000
    step = 3500
    guess_60  = [a,mu,sigma,tail,tau,bkg,step]

    bounds_60 = ([10,353,0.001,0.0,0.001,10,10],[inf,358,0.8,inf,inf,10000000,1000000])
    xF, xF_cov = pga.fit_hist(pga.gauss_cdf, hE, xE, var=np.ones(len(hE)), guess=guess_60, bounds=bounds_60)
    line, tail, step, peak = pga.gauss_cdf(xE,*xF,components=True)

    area = simps(peak +tail,dx=0.08)
    chisq_60 = []
    print("Calculating the chi^2")

    for i, h in enumerate(hE):
        func = pga.gauss_cdf(xE[i], *xF)
        diff = (func - hE[i])
        dev = diff**2/func
        chisq_60.append(abs(dev))



    chi_60 = sum(np.array(chisq_60))# / (len(hE)-7)
    chisq_ndf_60 = chi_60/(len(hE))

    meta_dir = os.path.expandvars(ds.runDB["meta_dir"])
    runNum = ds.ds_list[0]

    print("chi", chisq_ndf_60)
    plt.plot(xE[1:],hE,ls='steps', lw=1, c='b', label="data")
    plt.plot(xE,pga.gauss_cdf(xE,*xF),c='r', label='Fit')
    plt.plot(xE,(peak+tail), c='m', label = 'Gauss+Tail')
    plt.plot(xE,step, c='g', label = 'Step')
    plt.xlabel("Energy [keV]",ha='right', x=1.0)
    plt.ylabel("Counts",ha='right', y=1.0)
    plt.legend()
    plt.savefig(meta_dir+"/plots/356_line_run" + str(runNum) +".png")
    plt.show()

   
    hE, xE, vE = pgh.get_hist(t2[ecal], range=(76,84), dx=0.08)   

    a = 150000
    mu = 81
    sigma = 0.3
    a2 = 15000
    mu2= 80
    sigma2 = 0.3
    bkg = 4000
    step = 3500
    guess_60  = [a,mu,sigma,a2,mu2,sigma2,bkg,step]

    bounds_60 = ([10,80,0.001,10.0,75,0.00010,0.10,0.1],[inf,82,0.8,inf,80,0.8,1e9,1e9])
    xF, xF_cov = pga.fit_hist(pga.double_gauss, hE, xE, var=np.ones(len(hE)), guess=guess_60, bounds=bounds_60)
    fitfunc, gaus1, gaus2, step = pga.double_gauss(xE,*xF,components=True)

    area2 = simps(gaus1 +gaus2,dx=0.08)
    chisq_60 = []
    print("Calculating the chi^2")

    for i, h in enumerate(hE):
        func = pga.double_gauss(xE[i], *xF)
        diff = (func - hE[i])
        dev = diff**2/func
        chisq_60.append(abs(dev))



    chi_60 = sum(np.array(chisq_60))# / (len(hE)-7)
    chisq_ndf_60 = chi_60/(len(hE))

    meta_dir = os.path.expandvars(ds.runDB["meta_dir"])
    runNum = ds.ds_list[0]

    print("chi", chisq_ndf_60)
    plt.plot(xE[1:],hE,ls='steps', lw=1, c='b', label="data")
    plt.plot(xE,pga.double_gauss(xE,*xF),c='r', label='Fit')
    plt.plot(xE,(gaus1+gaus2), c='m', label = 'Gauss+Gauss')
    plt.plot(xE,step, c='g', label = 'Step')
    plt.xlabel("Energy [keV]",ha='right', x=1.0)
    plt.ylabel("Counts",ha='right', y=1.0)
    plt.legend()
    plt.savefig(meta_dir+"/plots/356_line_run" + str(runNum) +".png")
    plt.show()

    """
    The 302 line
    """

    hE, xE, vE = pgh.get_hist(t2[ecal], range=(296,306), dx=0.08)

    a = 150000
    mu = 302
    sigma = 0.3
    tail = 50000
    tau = 0.5
    bkg = 4000
    step = 3500
    guess_60  = [a,mu,sigma,tail,tau,bkg,step]

    bounds_60 = ([10,300,0.001,0.0,0.001,10,10],[inf,305,0.8,inf,inf,10000000,1000000])
    xF, xF_cov = pga.fit_hist(pga.gauss_cdf, hE, xE, var=np.ones(len(hE)), guess=guess_60, bounds=bounds_60)
    line, tail, step, peak = pga.gauss_cdf(xE,*xF,components=True)

    area30 = simps(peak +tail,dx=0.08)
    chisq_60 = []
    print("Calculating the chi^2")

    for i, h in enumerate(hE):
        func = pga.gauss_cdf(xE[i], *xF)
        diff = (func - hE[i])
        dev = diff**2/func
        chisq_60.append(abs(dev))



    chi_60 = sum(np.array(chisq_60))# / (len(hE)-7)
    chisq_ndf_60 = chi_60/(len(hE))

    meta_dir = os.path.expandvars(ds.runDB["meta_dir"])
    runNum = ds.ds_list[0]

    print("chi", chisq_ndf_60)
    plt.plot(xE[1:],hE,ls='steps', lw=1, c='b', label="data")
    plt.plot(xE,pga.gauss_cdf(xE,*xF),c='r', label='Fit')
    plt.plot(xE,(peak+tail), c='m', label = 'Gauss+Tail')
    plt.plot(xE,step, c='g', label = 'Step')
    plt.xlabel("Energy [keV]",ha='right', x=1.0)
    plt.ylabel("Counts",ha='right', y=1.0)
    plt.legend()
    plt.savefig(meta_dir+"/plots/302_line_run" + str(runNum) +".png")
    plt.show()


    """
    The 384 line
    """

    hE, xE, vE = pgh.get_hist(t2[ecal], range=(379,389), dx=0.08)

    a = 150000
    mu = 384
    sigma = 0.3
    tail = 50000
    tau = 0.5
    bkg = 4000
    step = 3500
    guess_60  = [a,mu,sigma,tail,tau,bkg,step]

    bounds_60 = ([10,382,0.001,0.0,0.001,10,10],[inf,386,0.8,inf,inf,10000000,1000000])
    xF, xF_cov = pga.fit_hist(pga.gauss_cdf, hE, xE, var=np.ones(len(hE)), guess=guess_60, bounds=bounds_60)
    line, tail, step, peak = pga.gauss_cdf(xE,*xF,components=True)

    area38 = simps(peak +tail,dx=0.08)
    chisq_60 = []
    print("Calculating the chi^2")

    for i, h in enumerate(hE):
        func = pga.gauss_cdf(xE[i], *xF)
        diff = (func - hE[i])
        dev = diff**2/func
        chisq_60.append(abs(dev))



    chi_60 = sum(np.array(chisq_60))# / (len(hE)-7)
    chisq_ndf_60 = chi_60/(len(hE))

    meta_dir = os.path.expandvars(ds.runDB["meta_dir"])
    runNum = ds.ds_list[0]

    print("chi", chisq_ndf_60)
    print("a1", area)
    print("a2", area2)
    print("a38",area38)
    print("a30",area30)
    print("ratio",area/area2)
    
    plt.plot(xE[1:],hE,ls='steps', lw=1, c='b', label="data")
    plt.plot(xE,pga.gauss_cdf(xE,*xF),c='r', label='Fit')
    plt.plot(xE,(peak+tail), c='m', label = 'Gauss+Tail')
    plt.plot(xE,step, c='g', label = 'Step')
    plt.xlabel("Energy [keV]",ha='right', x=1.0)
    plt.ylabel("Counts",ha='right', y=1.0)
    plt.legend()
    plt.savefig(meta_dir+"/plots/384_line_run" + str(runNum) +".png")
    plt.show()



if __name__=="__main__":
    main()


