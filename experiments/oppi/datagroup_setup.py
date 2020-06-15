#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pprint import pprint
from pygama import DataGroup

def main():
    """
    Requires oppi.json config file.
    Save an hdf5 file with pygama daq, raw, and dsp names + paths.
    """
    dg = DataGroup('oppi.json')
    # dg.lh5_dir_setup(create=True) # <-- run this once with create=True
    dg.scan_daq_dir()

    # -- experiment-specific choices --
    dg.file_keys.sort_values(['cycle'], inplace=True)
    dg.file_keys.reset_index(drop=True, inplace=True)

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
        # label the detector
        row['runtype'] = 'oppi'
        return row

    dg.file_keys = dg.file_keys.apply(get_cyc_info, axis=1)

    dg.get_lh5_cols()

    for col in ['run']:
        dg.file_keys[col] = pd.to_numeric(dg.file_keys[col])

    # -- filter out MJ60 runs --
    dg.file_keys = dg.file_keys.loc[dg.file_keys.run>=0].copy()

    dg.file_keys['run'] = dg.file_keys['run'].astype(int)

    print(dg.file_keys)

    # dg.save_df('oppi_fileDB.h5')


if __name__=='__main__':
    main()
