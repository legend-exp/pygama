#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import lat.utils as lat
from pprint import pprint
from ROOT import TFile, TChain, TTree, MGTWaveform

def main(argv):
    doc="""
    quick interactive waveform viewer for wave_skim.cc output files.
    Default is interactive mode.  
    - [Enter] advances to the next wf, 
    - 'p' goes to the previous one
    - 's' saves a wf to the folder ./plots/
    - 'q' to quit
    """
    par = argparse.ArgumentParser(description=doc)
    arg = par.add_argument
    s, st, sf = "store", "store_true", "store_false"
    arg("-f", nargs=1, action=s, help="set input file")
    arg("-c", nargs=1, action=s, help="set data cleaning t_cut")
    arg("-q", action=st, help="quick draw mode")
    args = par.parse_args()

    # user options
    if args.f:
        in_file = args.f
    else:
        # in_file = "/some/default/regex/*.root"
        in_file = "/Users/wisecg/Data/low_e/data/bkg/final/lat_final_ds6A.root"
    quick_draw = True if args.q else False
    if args.c:
        t_cut = args.c
    else:
        t_cut = ""
        t_cut = "trapENFCal > 1 && trapENFCal < 50" # some default cut

    # load chain and apply cut
    tt = TChain("skimTree")
    tt.Add(in_file)

    n = tt.Draw("Entry$:Iteration$",t_cut,"goff")
    evt, itr = tt.GetV1(), tt.GetV2()
    evt_list = [[int(evt[i]),int(itr[i])] for i in range(n)]

    nEnt = len(set([evt[i] for i in range(n)]))
    print("Found %d total entries, %d passing cut: %s" % (tt.GetEntries(), nEnt, t_cut))

    # loop over waveforms and draw them
    i, pEvt = -1, -1
    while(True):
        i += 1
        if not quick_draw and i!=0:
            val = input()
            if val == "q": break
            if val == "p": i -= 2
            if val.isdigit() : i = int(val)
            if val == "s":
                pltName = "./plots/wf-%d.pdf" % i
                print("Saving figure:",pltName)
                plt.savefig(pltName)
        if i >= len(evt_list): 
            break
        iE, iH = evt_list[i]

        if iE != pEvt:
            tt.GetEntry(iE)
        pEvt = iE

        run = tt.run
        chan = tt.channel.at(iH)
        hitE = tt.trapENFCal.at(iH)
        tOff = tt.tOffset.at(iH)
        wfMG = tt.MGTWaveforms.at(iH)
        period = wfMG.GetSamplingPeriod()
        wf = np.array(wfMG.GetVectorData()) # fastest conversion to ndarray
        ts = np.arange(tOff, tOff + len(wf) * period, period)
        
        # resize the waveform
        rem_lo, rem_hi = 4, 2 # should work for all DS's
        rm_samples = []
        if rem_lo > 0: rm_samples += [i for i in range(0, rem_lo+1)]
        if rem_hi > 0: rm_samples += [len(wf) - i for i in range(1, rem_hi+1)]
        ts = np.delete(ts, rm_samples)
        waveRaw = np.delete(wf, rm_samples)
        
        print("%d / %d  Run %d  chan %d  trapENF %.1f, iE %d, iH %d" % (i,len(evt_list),run,chan,hitE, iE, iH))

        # compute baseline and standard energy trapezoid
        waveBLSub = waveRaw - np.mean(waveRaw[:500])
        eTrap = lat.trapFilter(waveBLSub,400,250,-7200)
        nPad = len(waveBLSub)-len(eTrap)
        eTrap = np.pad(eTrap, (nPad,0), 'constant')
        eTrapTS = np.arange(0, len(eTrap)*10., 10)
        ePickoff = eTrapTS[nPad + 400 + 200]
        # plt.axvline(ePickoff, c='m')

        plt.cla()
        # plt.plot(ts, waveRaw, 'b', lw=2, label='Raw WF, %.2f keV' % (hitE))
        plt.plot(ts, waveBLSub, 'b', label='BlSub WF, %.2f keV' % (hitE))

        plt.xlabel("Time (ns)", ha='right', x=1)
        plt.ylabel("Voltage (ADC)", ha='right',y=1)
        plt.legend(loc=4)
        plt.tight_layout()
        plt.pause(0.0001) # scan speed, does plt.show(block=False) automatically


if __name__ == "__main__":
    main(sys.argv[1:])
