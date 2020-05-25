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
This is a script to fit the 60keV, 99keV and 103keV lines of an 241Am scan.
This script is based on the pygama version from December 2019 and is a bit outdated. 
An update will be done soon

You need to have done a Calibration before and the output must be in the ds.calDB file

The function takes a DataSet (December version) and a t2-level file 
Then a fit on the 60kev line and on the 99/103 keV lines is performed, the
integrals are caluclated and the ratio is determind

A.Zschocke
"""


def fit_Am_lines(ds, t2, display=False, write_DB=True):

    print("Fit Am lines")

    etype, ecal = "e_ftp", "e_cal"
    e_peak = 0

    #Load calibration Values 
    calDB = ds.calDB
    query = db.Query()
    table = calDB.table("cal_pass3").all()
    df_cal = pd.DataFrame(table) 

    slope = df_cal.iloc[0]["slope"]
    offset = df_cal.iloc[0]["offset"]
    
    # load in the energy and apply (linear) calibration
    ene = t2[etype]
    e_cal = ene* (ene * slope +offset)

    green_line = slope * 500 + offset


    
    fits = {}
    pk_names = ds.config["pks"]
    am_peaks = ds.config["peaks_of_interest"]
    
    # Here I did a quick study on the impact of the bin size on the integral
    # and the chi2 (this is the next for loop)
    ar = []
    chic = []
    scan = [0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
    aq = 1500000

    # For loop over different bin sizes 
    for bi in scan:

      # Do the 100keV lines first
      xlo, xhi, xpb =  90, 110,bi
      hE, xE, vE = pgh.get_hist(e_cal, range=(xlo, xhi), dx=xpb)
      inf = np.inf 
     
      # Set up initial values and limits
      guess_100 = [100000,99,0.5,11000,103,0.5,4050,101,0.5, 400000,39000,400,20000]   
      bounds_100 = ([-np.inf,97,-np.inf,-np.inf,102,-np.inf,-np.inf,100.1,0.001,-inf,-inf,-inf,-inf],[inf,100,inf,inf,104,inf,inf,101.7,0.8,inf,inf,inf,inf]) 

      #Do the fit (Am_double function from PeakFitting.py)
      xF, xF_cov = pga.fit_hist(pga.Am_double, hE, xE, var=np.ones(len(hE)), guess=guess_100, bounds=bounds_100)
      dg_fit, gaus1, gaus2, gaus3, step1, step2 = pga.Am_double(xE,*xF,components=True)

      results = {
                "99keV" : xF[1],
                "99keV_fwhm" : xF[2] * 2.355,
                "103keV" : xF[4],
                "103keV_fwhm" : xF[5] * 2.355

                # ...
              }

 
      #calculate the integral 
      area_g1 = simps(gaus1,dx = bi)
      area_g2 = simps(gaus2,dx = bi) 
 
      chisq = []
      for i, h in enumerate(hE):
        diff = (pga.Am_double(xE[i], *xF) - hE[i])**2 / hE[i]
        chisq.append(abs(diff))

      results["peak_integral1"] = area_g1
      results["peak_integral2"] = area_g2
      chisq_ndf_100 = sum(np.array(chisq) / (len(hE)-13))
      
      # Plot it if wanted
      if display:
        plt.plot(xE[1:],hE,ls='steps', lw=1, c='b', label="data")
        plt.plot(xE,pga.Am_double(xE,*xF),c='r', label='Fit')
        plt.plot(xE,gaus1+gaus2,c='m', label='Gauss 99 keV + 103 keV')
        plt.plot(xE,gaus3,c='y', label='Gauss bkg')
        plt.plot(xE,step1+step2,c='g', label='Step')
        plt.xlabel("Energy [keV]",ha='right', x=1.0)
        plt.ylabel("Counts",ha='right', y=1.0)
        plt.legend()
        meta_dir = os.path.expandvars(ds.config["meta_dir"])
        runNum = ds.ds_list[0]

        plt.savefig(meta_dir+"/plots/100keV_100ev_bin_lines_run" + str(runNum)+".png")
        plt.show()
    


      # Do the 60 keV line
      xlo, xhi, xpb =  50, 70, bi
      hE, xE, vE = pgh.get_hist(e_cal, range=(xlo, xhi), dx=xpb)
    
      a = aq
      mu = 59.5 
      sigma = 0.3
      tail = 50000
      tau = 0.5 
      bkg = 4000
      step = 3500
      guess_60  = [a,mu,sigma,tail,tau,bkg,step]
      bounds_60 = ([10,59,0.001,0.0,0.001,10,10],[inf,60.5,0.8,inf,inf,10000000,1000000])

      # The fit Function is a gauss_cdf
      xF, xF_cov = pga.fit_hist(pga.gauss_cdf, hE, xE, var=np.ones(len(hE)), guess=guess_60, bounds=bounds_60)  
      line, tail, step, peak = pga.gauss_cdf(xE,*xF,components=True)

      chisq_60 = []
      print("Calculating the chi^2")

      for i, h in enumerate(hE):
        func = pga.gauss_cdf(xE[i], *xF)
        diff = (func - hE[i])
        dev = diff**2/func
        chisq_60.append(abs(dev))
      
      chi_60 = sum(np.array(chisq_60))
      chisq_ndf_60 = chi_60/(len(hE))

      meta_dir = os.path.expandvars(ds.config["meta_dir"])
      runNum = ds.ds_list[0]

      if display:
        plt.plot(xE[1:],hE,ls='steps', lw=1, c='b', label="data")
        plt.plot(xE,pga.gauss_cdf(xE,*xF),c='r', label='Fit')
        plt.plot(xE,(peak+tail), c='m', label = 'Gauss+Tail')
        plt.plot(xE,step, c='g', label = 'Step')
        plt.xlabel("Energy [keV]",ha='right', x=1.0)
        plt.ylabel("Counts",ha='right', y=1.0)
        plt.legend()
        plt.savefig(meta_dir+"/plots/60keV_lines_100ev_bin__run" + str(runNum) +".png")
        plt.show()

      area = simps(peak+tail,dx=bi)
    
      print("xF\n",xF)
      print("chi_60", chisq_ndf_60)
      print("chi_100", chisq_ndf_100)
      print("Peak Integrals:")
      print("60 keV = ", area)
      print("99 keV = ", area_g1)
      print("10 3keV = ", area_g2)
      print("ratio 1 = ", area/area_g1)
      print("ratio 2 = ", area/area_g2)
      print("ratio 3 = ", area/(area_g1+area_g2))
    
      ar.append(area/(area_g1+area_g2))
      chic.append(chisq_ndf_60)

    plt.subplot(211)
    plt.plot(scan,chic,'bx',ms=15,label='chi^2/f')
    plt.grid()
    plt.axvline(green_line, c='g', lw=1, label="calibration value at 100 keV")
    plt.legend() 
    plt.subplot(212)
    plt.plot(scan,ar,'kx',ms=15,label='ratio "n60/(n99+n103)"')
    plt.axvline(green_line, c='g', lw=1, label="calibration value at 100 keV")
    plt.xlabel("bin size [keV]")
    plt.grid()
    plt.legend()
    plt.show() 
    
    if write_DB: 
        res_db = meta_dir+"/PeakRatios_100evbin.json"
        resDB = db.TinyDB(res_db)
        query = db.Query()
        ratiotable = resDB.table("Peak_Ratios")

        for dset in ds.ds_list:
         row = {
          "ds":dset,
          "chi2_ndf_60":chisq_ndf_60,
          "chi2_ndf_100":chisq_ndf_100,
          "60_keV": area,
          "99_keV": area_g1,
          "103_keV": area_g2,
          "r1": area/area_g1,
          "r2": area/area_g2,
          "r3":area/(area_g1+area_g2)

         }
        ratiotable.upsert(row, query.ds == dset)


     


if __name__=="__main__":
    main()
