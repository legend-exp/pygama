#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pprint import pprint
import tinydb as db

from pygama import DataGroup
from pygama.io.orcadaq import parse_header
import pygama.io.lh5 as lh5

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    tqdm.pandas() # suppress annoying FutureWarning

def main():
    doc="""
    pygama energy calibration routine.  uses a 'DataGroup' to organize runs.
    metadata is handled in JSON format by convention of legend-metadata.
    
    T. Mathew, C. Wiseman
    """
    dg = DataGroup('oppi.json')
    dg.load_df()
    file_db = dg.file_keys
    # que = 'run==0'
    # que = 'cycle == 2180'
    # dg.file_keys.query(que, inplace=True)
    # dg.file_keys = dg.file_keys[:1]
    
    cal_db = db.TinyDB('oppi_ecalDB.json')
    query = db.Query()

    table = cal_db.table('cal_test_name')
    for i in range(5):
        row = {'a':i, 'b':i/2, 'c':i*3}
        table.upsert(row, query.a == i) 

    table = cal_db.table("cal_test_name").all()
    df_cal = pd.DataFrame(table)
    
    print(df_cal)
    
    
    
if __name__=='__main__':
    main()