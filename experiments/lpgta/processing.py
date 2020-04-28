#!/usr/bin/env python3
import argparse
from pprint import pprint

from pygama import DataGroup

def main():
    doc="""
    LPGTA data processing routine. 
    You must set these environment variables:
      * $LPGTA_DIR : base data directory
      * $LEGEND_META : the legend-metadata repository
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    args = par.parse_args()
    
    f = '$LEGEND_META/analysis/LPGTA/LPGTA.json'
    
    # run = 19
    # dg = DataGroup(run, 21, config=f, mode='daq', nfiles=3)
    dg = DataGroup(19, config=f, mode='daq')
    
    # print(dg.runs)
    # pprint(dg.config)
    # pprint(dg.runDB)
    
    
    
def daq_to_raw():
    """
    """
    print('hi')
    
    
    
if __name__=="__main__":
    main()
    