#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pprint import pprint
from pygama import DataGroup

def main():
    """
    """


    # analyze_lpgta()
    # analyze_cage()
    # analyze_hades()
    # analyze_ornl()
    analyze_surf()


def analyze_lpgta():

    dg = DataGroup('LPGTA.json')

    dg.lh5_dir_setup()
    dg.scan_daq_dir()

    # -- experiment-specific choices --

    # run 1 & 2 files don't match template
    dg.fileDB.query('run > 2', inplace=True)

    dg.fileDB.sort_values(['run','YYYYmmdd','hhmmss'], inplace=True)
    dg.fileDB.reset_index(drop=True, inplace=True)

    def get_cmap(row):
        row['cmap'] = dg.runDB[f"{row['run']:0>4d}"]["cmap"]
        return row

    dg.fileDB = dg.fileDB.apply(get_cmap, axis=1)

    dg.fileDB['runtype'] = dg.fileDB['rtp']

    dg.get_lh5_cols()

    dg.save_df('./LPGTA_fileDB.h5')

    print(dg.fileDB)


def analyze_cage():

    dg = DataGroup('CAGE.json')
    dg.lh5_dir_setup()

    dg.scan_daq_dir()

    # -- experiment-specific choices --
    dg.fileDB.sort_values(['cycle'], inplace=True)
    dg.fileDB.reset_index(drop=True, inplace=True)

    def get_cyc_info(row):
        """
        map cycle numbers to physics runs, and identify detector
        """
        cyc = row['cycle']
        for run, cycles in dg.runDB.items():
            tmp = cycles[0].split(',')
            for rng in tmp:
                if '-' in rng:
                    clo, chi = [int(x) for x in rng.split('-')]
                    if clo <= cyc <= chi:
                        row['run'] = run
                        break
                else:
                    clo = int(rng)
                    if cyc == clo:
                        row['run'] = run
                        break
        # label the detector ('runtype' matches 'run_types' in config file)
        if cyc < 126:
            row['runtype'] = 'oppi'
        else:
            row['runtype'] = 'icpc'
        return row

    dg.fileDB = dg.fileDB.apply(get_cyc_info, axis=1)

    dg.get_lh5_cols()

    for col in ['run']:
        dg.fileDB[col] = pd.to_numeric(dg.fileDB[col])

    print(dg.fileDB)

    dg.save_df('CAGE_fileDB.h5')


def analyze_hades():
    """
    """
    dg = DataGroup('HADES.json')

    dg.lh5_dir_setup()
    # dg.lh5_dir_setup(create=True)

    dg.scan_daq_dir()

    # -- experiment-specific stuff --
    dg.fileDB['runtype'] = dg.fileDB['detSN']

    # add a sortable timestamp column
    def get_ts(row):
        ts = f"{row['YYmmdd']} {row['hhmmss']}"
        row['date'] = pd.to_datetime(ts, format='%y%m%d %H%M%S')
        return row
    dg.fileDB = dg.fileDB.apply(get_ts, axis=1)
    dg.fileDB.sort_values('date', inplace=True)

    dg.get_lh5_cols()
    print(dg.fileDB['raw_file'].values)

    dg.save_df('HADES_fileDB.h5')


def analyze_ornl():

    dg = DataGroup('ORNL.json')
    # dg.lh5_dir_setup()
    dg.scan_daq_dir()

    # expt-specific organization
    dg.fileDB.sort_values(['cycle'], inplace=True)
    dg.fileDB.reset_index(drop=True, inplace=True)

    dg.save_keys()
    dg.load_keys()
    print(dg.fileDB)



def analyze_surf():
    """
    """
    dg = DataGroup('SURFCHAR.json')

    dg.lh5_dir_setup()
    # dg.lh5_dir_setup(create=True)

    dg.scan_daq_dir()

    # -- experiment-specific choices --
    dg.fileDB.sort_values(['cycle'], inplace=True)
    dg.fileDB.reset_index(drop=True, inplace=True)

    # TODO: adapt "get_cyc_info" function from analyze_cage to
    # fill in serial numbers for each detector
    dg.fileDB['runtype'] = "P9999A"

    dg.get_lh5_cols()

    # print(dg.fileDB.query('cycle < 10'))

    dg.save_df('SURFCHAR_fileDB.h5')


if __name__=='__main__':
    main()
