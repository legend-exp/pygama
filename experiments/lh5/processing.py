#!/usr/bin/env python3
from pygama import DataSet

def main():
    """
    testing Yoann's Tier 0 (daq_to_raw) FlashCam parser.
    this is the high-level part of the code, something that a user might
    write for processing with a specific config file.
    """
    ds = DataSet(run=0, config="config.json")
    ds.daq_to_raw(overwrite=True, test=True)

    
if __name__=="__main__":
    main()