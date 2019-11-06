#!/usr/bin/env python3
from pygama import DataSet

def main():
    """
    testing Yoann's Tier 0 (daq_to_raw) FlashCam parser.
    this is the high-level part of the code, something that a user might
    write for processing with a specific config file.
    """
    # process_data()
    read_data()
    

def process_data():
    # build dataset and run processing
    ds = DataSet(run=0, config="config.json")
    ds.daq_to_raw(overwrite=True, test=False)


def read_data():    
    # read the output
    import h5py
    out_file = "/Users/wisecg/Data/L200/t1_run0.h5"
    hf = h5py.File(out_file)
    
    header = hf['/header']
    for name, val in header.attrs.items():
        print(name, val)
    
    # print(f.keys())
    hf.close()

    
if __name__=="__main__":
    main()