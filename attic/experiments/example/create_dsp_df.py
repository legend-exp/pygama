#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

import pygama.lh5 as lh5
import pygama.analysis.histograms as pgh

def main():
    """
    an example of loading an LH5 DSP file and converting to pandas DataFrame.
    """
    # we will probably make this part simpler in the near future
    f = '/Users/wisecg/Data/lh5/hades_I02160A_r1_191021T162944_th_HS2_top_psa_dsp.lh5'
    sto = lh5.Store()
    groups = sto.ls(f) # the example file only has one group, 'raw'
    data = sto.read_object('raw', f)
    df_dsp = data.get_dataframe()

    # from here, we can use standard pandas to work with data
    print(df_dsp)
    
    # one example: create uncalibrated energy spectrum,
    # using a pygama helper function to get the histogram
    
    elo, ehi, epb = 0, 100000, 10
    ene_uncal = df_dsp['trapE']
    hist, bins, _ = pgh.get_hist(ene_uncal, range=(elo, ehi), dx=epb)
    bins = bins[1:] # trim zero bin, not needed with ds='steps'

    plt.semilogy(bins, hist, ds='steps', c='b', label='trapE')
    plt.xlabel('trapE', ha='right', x=1)
    plt.ylabel('Counts', ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
if __name__=="__main__":
    main()