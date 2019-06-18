#!/usr/bin/env python3
import numpy as np
import tinydb as db
import pandas as pd
from pprint import pprint
from pygama import DataSet
from pygama.dsp.base import Intercom
from pygama.io.tier1 import ProcessTier1

def main():
    """
    """
    run_db, cal_db = "runDB.json", "calDB.json"
    ds = DataSet(run=279, md=run_db, cal=cal_db)
    optimize(ds)

    
def optimize(ds):
    """
    """
    run = ds.runs[0]
    t1_file = ds.paths[run]["t1_path"]
    t2_file = ds.paths[run]["t2_path"]
    conf = ds.paths[run]["build_opt"]
    
    # proc_list = ds.runDB["build_options"][conf]["tier1_options"]

    # short as as possible proc list
    proc_list = {
        'clk': 100000000.0,
        'avg_bl': {'ihi': 600},
        'blsub': {},
        'trap': [
            {"wfout":"wf_etrap", 'rise':4, 'flat':2.5, 'decay':72},
            {"wfout":"wf_atrap", 'rise':0.04, 'flat':0.1, 'fall':2}
        ],
        'get_max': [ {'wfin': 'wf_etrap'}, {'wfin': 'wf_atrap'} ],
        'ftp': {}
        }
    # pprint(proc_list)
    proc = Intercom(proc_list)
    ProcessTier1(
        t1_file,
        proc,
        output_dir=ds.tier_dir,
        overwrite=True,
        verbose=False,
        multiprocess=True,
        nevt=20000,
        chunk=ds.runDB["chunksize"])
    
    
if __name__=="__main__":
    main()
    
    