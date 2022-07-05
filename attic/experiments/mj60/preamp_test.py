#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from pygama import DataSet
from pygama.utils import set_plot_style
from pygama.analysis.histograms import *
# set_plot_style("clint")

def main():
    """
    """
    get_spectra()
    # task1()
    # task2()


def get_spectra():

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = runDB["tier_dir"]

    ds = DataSet(runlist=[555], md='./runDB.json', tier_dir=tier_dir)
    t2df = ds.get_t2df()

    t2df = t2df.loc[t2df.e_ftp > 500] # Low energy cut


    #print(t2df.columns)
    # print(t2df)
    # exit()

    # 4 to 36 pF variable cap

    rise_time = t2df["tp90"] - t2df["tp10"]

    ds2 = DataSet(runlist=[556], md='./runDB.json', tier_dir=tier_dir)
    t2df_2 = ds2.get_t2df()

    t2df_2 = t2df_2.loc[t2df_2.e_ftp > 500]

    rise_time2 = t2df_2["tp90"] - t2df_2["tp10"]

    ds3 = DataSet(runlist=[554], md='./runDB.json', tier_dir=tier_dir)
    t2df_3 = ds3.get_t2df()

    t2df_3 = t2df_3.loc[t2df_3.e_ftp > 500]


    rise_time3 = t2df_3["tp90"] - t2df_3["tp10"]

    xlo, xhi, xpb = 0., 500., 1

    hP, xP, _ = get_hist(rise_time, range=(xlo, xhi), dx=xpb)
    hP2, xP2, _ = get_hist(rise_time2, range=(xlo, xhi), dx=xpb)
    hP3, xP3, _ = get_hist(rise_time3, range=(xlo, xhi), dx=xpb)

    #Note to self: for risetime histograms, use similar to above, but replace
    #first parameter with rise_time!

    plt.semilogy(xP[:-1]*0.423, hP, ls='steps', lw=1.5, c='k',
             label="Rise Time, Preamp 1".format(sum(hP)))
    # hist = plt.hist(rise_time, bins = 1000)
    plt.semilogy(xP2[:-1]*0.423, hP2, ls='steps', lw=1.5, c='c',
             label="Rise Time, Preamp 2".format(sum(hP)))
    plt.semilogy(xP3[:-1]*0.423, hP3, ls='steps', lw=1.5, c='0.5',
             label="Rise Time, Preamp 0".format(sum(hP)))
    plt.xlabel("Rise Time", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("Rise Time Comparison")


if __name__=="__main__":
    main()

def get_spectra2():
    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = runDB["tier_dir"]
