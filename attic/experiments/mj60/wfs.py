import pandas as pd
import sys
import time
import numpy as np
import scipy as sp
import scipy.optimize as opt
import scipy.signal as signal
import os, json
import pygama.dataset as ds
import pygama.analysis.histograms as pgh
import pygama.dsp.transforms as pgt
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from tqdm import tqdm
plt.style.use('style.mplstyle')

# for this code to work you need the raw_to_dsp, tier2, and spectrum files for the run of interest.

def main():

    plot_wfs()
    #flip_through_wfs()
    #ADC_difference()
    #ADC_difference_cut()
    #superpulse()
    #samples_scatter()
    #samples_hist()
    #baseline_hist()

def plot_wfs():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    start = time.time()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')

    runtime = ds.DataSet(run=int(sys.argv[1]), md='./runDB.json').get_runtime()
    counts = len(df)
    counts_per_second = (len(df))/runtime

    print('total runtime = {} seconds'.format(runtime))
    print('counts = {}'.format(counts))
    print('counting rate = {} counts/second'.format(counts_per_second))
    exit()

    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']

    def bl_sub(wf):
        return df.loc[wf,:]-df.iloc[wf,0:850].mean()

    nsamp = 3000

    xvals = np.arange(0,nsamp)

    for i in range(0,11):
        plt.plot(xvals, bl_sub(i), lw=1)
    plt.xlabel('Sample Number', ha='right', x=1.0)
    plt.ylabel('ADC Value', ha='right', y=1.0)
    plt.tight_layout()
    print('python script time = {} seconds'.format(time.time() - start))
    plt.show()


def flip_through_wfs():

    if(len(sys.argv) != 4):
        print('Usage: wfs.py [run number] [lower energy limit (keV)] [upper energy limit (keV)]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')
    df['e_cal'] = pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[1]), key="df")['e_cal']
    df['AoverE'] = (pd.read_hdf("{}/t2_run{}.h5".format(tier_dir,sys.argv[1]))['current_max'])/df['e_cal']

    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']

    df = df.loc[(df.AoverE>=0.05)&(df.AoverE<=0.06)]
    df = df.loc[(df.e_cal>int(sys.argv[2]))&(df.e_cal<int(sys.argv[3]))]
    df = df.reset_index(drop=True)
    df_2 = pd.DataFrame(df['e_cal'])
    del df['e_cal']
    del df['AoverE']

    def bl_sub(wf):
        return df.loc[wf,:]-df.iloc[wf,0:500].mean()

    def savgol(wf):
        return signal.savgol_filter(df.loc[wf,:]-df.iloc[wf,0:500].mean(), 47, 2)

    nsamp = 3000

    xvals = np.arange(0,nsamp)

    i = -1
    while True:
        if i != -1:
            inp = input()
            if inp == "q": exit()
            if inp == "p": i -= 2
        i += 1
        print(i)

        plt.cla()
        plt.plot(xvals, bl_sub(i), color="black", lw=2, label="raw wf, run {}, E = {:.03f} keV".format(str(sys.argv[1]), df_2['e_cal'][i]))
        plt.plot(xvals, savgol(i), color="red", lw=1, label="Savitzky-Golay Filter")
        plt.xlabel('Sample Number', ha='right', x=1.0)
        plt.ylabel('ADC Value', ha='right', y=1.0)
        plt.legend(frameon=True, loc='upper left', fontsize='x-small')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)


def ADC_difference():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    start = time.time()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')
    df_2 = pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[1]), key="df")

    df_2 = df_2.reset_index(drop=True)
    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']
    #df_2['diff'] = [0]*len(df_2)

    #def dADC(wf):
        #return df.iloc[wf,1499:3000].mean()-df.iloc[wf,0:500].mean()

    #for i in range(len(df)):
        #df_2['diff'][i] = dADC(i)

    df_2['diff'] = df.iloc[:,1499:3000].mean(axis=1) - df.iloc[:,0:500].mean(axis=1)
    df_2['ratio'] = df_2['diff']/df_2['e_cal']
    print('python script time = {:.0f} seconds'.format(time.time() - start))

    plt.hist2d(df_2['e_cal'], df_2['diff'], np.arange(-5,100,0.1), norm=LogNorm())
    plt.xlim(0,50)
    plt.ylim(-5,100)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('dADC', ha='right', y=1.0)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()


def ADC_difference_cut():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    start = time.time()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')
    df_2 = pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[1]), key="df")

    df_2 = df_2.reset_index(drop=True)
    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']
    #df_2['diff'] = [0]*len(df_2)

    #def dADC(wf):
        #return df.iloc[wf,1499:3000].mean()-df.iloc[wf,0:500].mean()

    #for i in range(len(df)):
        #df_2['diff'][i] = dADC(i)

    df_2['diff'] = df.iloc[:,1499:3000].mean(axis=1) - df.iloc[:,0:500].mean(axis=1)
    df_2['ratio'] = df_2['diff']/df_2['e_cal']

    df_2 = df_2.loc[(df_2['diff']>3)&(df_2.e_cal>5)]
    print('python script time = {:.0f} seconds'.format(time.time() - start))

    plt.hist2d(df_2['e_cal'], df_2['diff'], np.arange(-5,100,0.1), norm=LogNorm())
    plt.xlim(0,50)
    plt.ylim(-5,100)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('dADC', ha='right', y=1.0)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()


def superpulse():

    if(len(sys.argv) != 5):
        print('Usage: wfs.py [run number 1] [run number 2] [lower energy cut (keV)] [upper energy cut (keV)]')
        sys.exit()

    start = time.time()

    def pulse(run_number, lower_energy_cut, upper_energy_cut):

        with open("runDB.json") as f:
            runDB = json.load(f)
        tier_dir = os.path.expandvars(runDB["tier_dir"])
        meta_dir = os.path.expandvars(runDB["meta_dir"])

        df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,int(run_number)), '/ORSIS3302DecoderForEnergy')
        df_2 = pd.read_hdf('{}/Spectrum_{}.hdf5'.format(meta_dir,int(run_number)))

        df['e_cal'] = df_2['e_cal']

        df = df.loc[(df.e_cal>=float(lower_energy_cut))&(df.e_cal<=float(upper_energy_cut))]

        df = df.reset_index(drop=True)
        del df['energy']
        del df['channel']
        del df['energy_first']
        del df['ievt']
        del df['packet_id']
        del df['timestamp']
        del df['ts_hi']
        del df['ts_lo']
        del df['e_cal']

        superpulse = pd.DataFrame(df[0:-1][0:len(df)].mean(), columns=['superpulse'])
        superpulse['superpulse'] = superpulse['superpulse'] - superpulse['superpulse'][0:500].mean()

        return superpulse['superpulse'].values

    a = pulse(sys.argv[2],sys.argv[3],sys.argv[4])
    #b = pulse(sys.argv[1],sys.argv[3],sys.argv[4])

    #b = b*sum(a[2500:2999])/sum(b[2500:2999])


    nsamp = 3000
    xvals = np.arange(0,nsamp)
    plt.plot(xvals, a, lw=1, color='black', label='superpulse -- run {}, {}<E<{} keV'.format(sys.argv[2],sys.argv[3],sys.argv[4]))
    #plt.plot(xvals, b, lw=1, color='purple', label='superpulse -- run {}, {}<E<{} keV'.format(sys.argv[1],sys.argv[3],sys.argv[4]))
    #plt.plot(xvals, signal.savgol_filter(b, 47, 2), lw=1, color='blue', label='savgol filtered superpulse -- run {}, {}<E<{} keV'.format(sys.argv[1],sys.argv[3],sys.argv[4]))
    plt.plot(xvals, signal.savgol_filter(a, 47, 2), lw=1, color='red', label='savgol filtered superpulse -- run {}, {}<E<{} keV'.format(sys.argv[2],sys.argv[3],sys.argv[4]))
    plt.xlabel('Sample Number', ha='right', x=1.0)
    plt.ylabel('ADC Value', ha='right', y=1.0)
    plt.legend(frameon=True, loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()


def samples_scatter():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')

    runtime = ds.DataSet(run=int(sys.argv[1]), md='./runDB.json').get_runtime()
    counts_per_second = (len(df))/runtime

    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']

    xvals = np.arange(0,len(df))

    plt.scatter(xvals, df[df.columns[454]], s=18, color='red', label='sample 454, run {}'.format(sys.argv[1]))
    plt.scatter(xvals, df[df.columns[456]], s=12, color='yellow', label='sample 456, run {}'.format(sys.argv[1]))
    plt.scatter(xvals, df[df.columns[455]], s=0.5, color='black', label='sample 455, run {}'.format(sys.argv[1]))
    plt.xlabel('waveform number', ha='right', x=1.0)
    plt.ylabel('ADV Value', ha='right', y=1.0)
    plt.legend(frameon=True, loc='best', fontsize='x-small')
    plt.tight_layout()
    plt.show()


def samples_hist():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')

    runtime = ds.DataSet(run=int(sys.argv[1]), md='./runDB.json').get_runtime()
    counts_per_second = (len(df))/runtime

    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']

    plt.hist(df[df.columns[455]], np.arange(-8000,-1000,1), histtype='step', color='black', label='sample 455, run {}'.format(sys.argv[1]))
    plt.hist(df[df.columns[456]], np.arange(-8000,-1000,1), histtype='step', color='red', label='sample 456, run {}'.format(sys.argv[1]))
    plt.hist(df[df.columns[454]], np.arange(-8000,-1000,1), histtype='step', color='blue', label='sample 454, run {}'.format(sys.argv[1]))
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('ADC Value', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.legend(frameon=True, loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()


def baseline_hist():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    with open("runDB.json") as f:
       runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')
    df_2 = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))

    df_2 = df_2.reset_index(drop=True)
    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']

    df[df.columns[0]] = df[df.columns[0]] - df_2['bl_p0']
    baseline_samples = df[df.columns[0]].values

    for i in tqdm(range(1,501)):
        df[df.columns[i]] = df[df.columns[i]] - df_2['bl_p0']
        a = df[df.columns[int(i)]].values
        baseline_samples = np.append(baseline_samples, a)

    plt.hist(signal.savgol_filter(baseline_samples,47,2), np.arange(-50,50,0.25),  histtype='step', color='black', label='filtered baseline samples of waveforms, run {}'.format(sys.argv[1]))
    #plt.ylim(0,plt.ylim()[1])
    plt.xlabel('ADC Value', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.semilogy()
    plt.legend(frameon=True, loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
        main()
