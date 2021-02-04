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
    quick interactive waveform viewer.
    Right now the cut (df.loc) is hardcoded, should decide if we want to 
    replace this with df.query, which can take a string.
    
    - [Enter] advances to the next wf, 
    - 'p' goes to the previous one
    - 's' saves a wf to the folder ./plots/
    - 'q' to quit
    """
    # par = argparse.ArgumentParser(description=doc)
    # arg = par.add_argument
    # s, st, sf = "store", "store_true", "store_false"
    # arg("-f", nargs=1, action=s, help="set input file")
    # args = par.parse_args()
    
    run = 42
    iwf_max = 100000 # tier 1 files can be a lot to load into memory
    ds = DataSet(run=run, md="runDB.json")
    ft1 = ds.paths[run]["t1_path"]
    t1df = pd.read_hdf(ft1, "ORSIS3302DecoderForEnergy", where=f"ievt < {iwf_max}")
    t1df.reset_index(inplace=True) # required step -- fix pygama "append" bug
    
    # get waveform dataframe
    wf_cols = []
    for col in t1df.columns:
        if isinstance(col, int):
            wf_cols.append(col)
    wfs = t1df[wf_cols]
    
    # apply a cut based on the t1 columns
    # idx = t1df.index[(t1df.energy > 1.5e6)&(t1df.energy < 2e6)]
    
    # apply a cut based on the t2 columns
    ft2 = ds.paths[run]['t2_path']
    t2df = pd.read_hdf(ft2, where=f"ievt < {iwf_max}")
    t2df.reset_index(inplace=True)

    # t2df['AoE'] = t2df.current_max / t2df.e_ftp # scipy method
    t2df['AoE'] = t2df.atrap_max / t2df.e_ftp # trapezoid method
    
    idx = t2df.index[(t2df.AoE < 0.7)
                     &(t2df.e_ftp > 1000) & (t2df.e_ftp < 10000)
                     &(t2df.index < iwf_max)]

    wfs = wfs.loc[idx]
    wf_idxs = wfs.index.values # kinda like a TEntryList
    
    # make sure the cut output makes sense
    cols = ['ievt', 'timestamp', 'energy', 'e_ftp', 'atrap_max', 'current_max', 't0', 
            't_ftp', 'AoE', 'tslope_pz', 'tail_tau']
    print(t2df.loc[idx][cols].head())
    print(t1df.loc[idx].head())
    print(wfs.head())
    
    # iterate over the waveform block 
    iwf = -1
    while True:
        if iwf != -1:
            val = input()
            if val == "q": break
            if val == "p": i -= 2
            if val.isdigit() : i = int(val)
            if val == "s":
                pltName = "./plots/wf-%d.pdf" % i
                print("Saving figure:",pltName)
                plt.savefig(pltName)
        iwf += 1
        iwf_cut = wf_idxs[iwf]
        
        # get waveform and dsp values
        wf = wfs.iloc[iwf]
        dsp = t2df.iloc[iwf_cut]
        ene = dsp.e_ftp
        aoe = dsp.AoE
        ts = np.arange(len(wf))
        
        # nice horizontal print of a pd.Series
        print(iwf, iwf_cut)
        print(wf.to_frame().T)
        print(t2df.iloc[iwf_cut][cols].to_frame().T)
        
        plt.cla()
        plt.plot(ts, wf, "-b", alpha=0.9, label=f'e: {ene:.1f}, a/e: {aoe:.1f}')
        
        # savitzky-golay smoothed
        # wfsg = signal.savgol_filter(wf, 47, 2)
        wfsg = signal.savgol_filter(wf, 47, 1)
        plt.plot(ts, wfsg, "-r", label='savitzky-golay filter')

        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)


if __name__ == "__main__":
    main(sys.argv[1:])
