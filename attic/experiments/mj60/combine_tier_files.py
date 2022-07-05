import pandas as pd
import json
import os
import sys
import time
import numpy as np

def main():

    combine_tier_files()
    #combine_tier2_files()

def combine_tier_files():
    '''
    This function can only handle about 10 full sized runs at a time. I would not suggest trying any more.
    '''

    if(len(sys.argv) != 3):
        print('Usage: combine_tier_files.py [lower run number] [upper run number]')
        sys.exit()

    start = time.time()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])

    print('Reading in raw_to_dsp and tier2 files:')

    #df_tier1 = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')
    df_tier2 = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]), columns=['e_ftp', 'energy', 'ttrap_max', 'dADC', 'fS', 'bl_p0'])
    df_tier2['run'] = np.full(len(df_tier2), sys.argv[1])
    print(sys.argv[1])

    for i in range(int(sys.argv[1])+1,int(sys.argv[2])+1):
        #df_tier1 = df_tier1.append(pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,i), '/ORSIS3302DecoderForEnergy'), ignore_index=True)
        df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,i), columns=['e_ftp', 'energy', 'ttrap_max', 'dADC', 'fS', 'bl_p0'])
        df['run'] = np.full(len(df), i)
        #df_tier2 = df_tier2.append(pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,i), columns=['e_ftp', 'energy']), ignore_index=True)
        df_tier2 = df_tier2.append(df, ignore_index=True)
        print(i)

    print('Resetting indices on raw_to_dsp and tier2 files ...')
    #df_raw_to_dsp = df_raw_to_dsp.reset_index(drop=True)
    df_tier2 = df_tier2.reset_index(drop=True)

    #print('Removing unnecessary columns from raw_to_dsp file ...')
    #del df_raw_to_dsp['energy']
    #del df_raw_to_dsp['channel']
    #del df_raw_to_dsp['energy_first']
    #del df_raw_to_dsp['ievt']
    #del df_raw_to_dsp['packet_id']
    #del df_raw_to_dsp['timestamp']
    #del df_raw_to_dsp['ts_hi']
    #del df_raw_to_dsp['ts_lo']    
   
    #print('Adding dADC into tier2 file ...')
    #df_tier2['dADC'] = df_raw_to_dsp.iloc[:,1499:3000].mean(axis=1) - df_raw_to_dsp.iloc[:,0:500].mean(axis=1)

    print('Saving combined tier2 file ...')
    df_tier2.to_hdf('./t2_run{}-{}.h5'.format(sys.argv[1],sys.argv[2]), key='df_tier2', mode='w')

    print('tier2: {} rows'.format(len(df_tier2)))
    print('The script has finished running! The run time was {:.0f} seconds'.format(time.time() - start))

def combine_tier2_files():
    
    if(len(sys.argv) < 2):
        print('Usage: combine_tier_files.py [runs of interest] [output file name]')
        sys.exit()

    start = time.time()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])

    print('Reading in tier2 files:')
    df_tier2 = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))
    print(sys.argv[1])

    for i in range(2,int(len(sys.argv))-1):
        df_tier2 = df_tier2.append(pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[i])), ignore_index=True)
        print(sys.argv[i])

    print('Resetting indices on tier2 file ...')
    df_tier2 = df_tier2.reset_index(drop=True)

    print('Saving tier2 file ...')
    df_tier2.to_hdf('{}/{}.h5'.format(tier_dir,sys.argv[-1]), key='df_tier2', mode='w')
    
    print('tier2: {} rows'.format(len(df_tier2)))
    print('The script has finished running! The run time was {:.0f} seconds'.format(time.time() - start))

if __name__ == '__main__':
        main()
