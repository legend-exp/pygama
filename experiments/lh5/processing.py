#!/usr/bin/env python3
import json
from abc import ABC
from pygama import DataSet
import pyflashcam

def main():
    """
    testing Yoann's Tier 0 (daq_to_raw) FlashCam parser.
    Requires 'pyflashcam' python module.  Uses local copies of the Digitizer
    and DataLoader class s/t it's easier to see how to remove the pandas stuff
    and replace with the desired .lh5 format.
    """
    daq_to_raw()
    
    
def daq_to_raw():
    """
    formerly "Tier 0", aka the loop over the raw data file
    """
    # f_test = "/Users/wisecg/Data/L200/protothppmco_0.fcio"
    
    with open("config.json") as f:
        config = json.load(f)
    
    # -- declare the DataSet --
    ds = DataSet(run=0, md=config)

    print(ds.paths)




    

    
if __name__=="__main__":
    main()