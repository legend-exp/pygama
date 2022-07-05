#!/usr/bin/env python3
import os
import json
import h5py
import argparse
import pandas as pd
import numpy as np
import tinydb as db
from tinydb.storages import MemoryStorage
from pprint import pprint
import matplotlib.pyplot as plt
plt.style.use('../clint.mpl')
from matplotlib.colors import LogNorm

from pygama import DataGroup
import pygama.io.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf

def main():
    doc="""
    analysis of Aug 2020 OPPI+CAGE commissioning runs (138-141)
    tasks:
    - load calibration from energy_cal
    - show 1460 peak stability
    - show removal of low-e retrigger noise
    - look at waveforms near 5 MeV, confirm they're muon-related
    - look at low-e waveforms, examine noise
    - determine pz correction value
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('-q', '--query', nargs=1, type=str,
        help="select file group to calibrate: -q 'run==1' ")
    args = par.parse_args()

    # load main DataGroup, select files from cmd line
    dg = DataGroup('cage.json', load=True)
    if args.query:
        que = args.query[0]
        dg.fileDB.query(que, inplace=True)
    else:
        dg.fileDB = dg.fileDB[-1:]
    view_cols = ['runtype', 'run', 'cycle', 'startTime', 'runtime', 'threshold']
    print(dg.fileDB[view_cols])

    # -- run routines --
    # show_raw_spectrum(dg)
    # show_cal_spectrum(dg)
    # show_wfs(dg)
    # data_cleaning(dg) 
    # peak_drift(dg)
    # pole_zero(dg)
    label_alpha_runs(dg)


def show_raw_spectrum(dg):
    """
    show spectrum w/ onbd energy and trapE
    - get calibration constants for onbd energy and 'trapE' energy
    - TODO: fit each expected peak and get resolution vs energy
    """
    # get file list and load energy data (numpy array)
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    dsp_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
    edata = lh5.load_nda(dsp_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/dsp')
    rt_min = dg.fileDB['runtime'].sum()
    u_start = dg.fileDB.iloc[0]['startTime']
    t_start = pd.to_datetime(u_start, unit='s') # str

    print('Found energy data:', [(et, len(ev)) for et, ev in edata.items()])
    print(f'Runtime (min): {rt_min:.2f}')

    elo, ehi, epb, etype = 0, 25000, 10, 'trapEmax'

    ene_uncal = edata[etype]
    hist, bins, _ = pgh.get_hist(ene_uncal, range=(elo, ehi), dx=epb)

    # normalize by runtime
    hist_rt = np.divide(hist, rt_min * 60)

    plt.plot(np.nan, np.nan, '-w', lw=1, label=t_start)

    plt.semilogy(bins[1:], hist_rt, ds='steps', c='b', lw=1,
                 label=f'{etype}, {rt_min:.2f} mins')

    plt.xlabel(etype, ha='right', x=1)
    plt.ylabel('cts / sec', ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_cal_spectrum(dg):
    """
    apply calibration to dsp file
    """
    # get file list and load energy data (numpy array)
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    dsp_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
    edata = lh5.load_nda(dsp_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/dsp')
    rt_min = dg.fileDB['runtime'].sum()
    u_start = dg.fileDB.iloc[0]['startTime']
    t_start = pd.to_datetime(u_start, unit='s') # str
    print('Found energy data:', [(et, len(ev)) for et, ev in edata.items()])
    print(f'Runtime (min): {rt_min:.2f}')

    # load calibration from peakfit
    cal_db = db.TinyDB(storage=MemoryStorage)
    with open('ecalDB.json') as f:
        raw_db = json.load(f)
        cal_db.storage.write(raw_db)
    runs = dg.fileDB.run.unique()
    if len(runs) > 1:
        print("sorry, I can't do combined runs yet")
        exit()
    run = runs[0]
    tb = cal_db.table("peakfit_trapEmax").all()
    df_cal = pd.DataFrame(tb)
    df_cal['run'] = df_cal['run'].astype(int)
    df_run = df_cal.loc[df_cal.run==run]
    cal_pars = df_run.iloc[0][['cal0','cal1','cal2']]

    # compute calibrated energy
    pol = np.poly1d(cal_pars) # handy numpy polynomial object
    cal_data = pol(edata['trapEmax'])

    elo, ehi, epb, etype = 0, 3000, 1, 'trapEmax_cal' # gamma region
    elo, ehi, epb, etype = 2500, 8000, 10, 'trapEmax_cal' # overflow region
    # elo, ehi, epb, etype = 0, 250, 1, 'trapEmax_cal' # low-e region

    hist, bins, _ = pgh.get_hist(cal_data, range=(elo, ehi), dx=epb)

    # normalize by runtime
    hist_rt = np.divide(hist, rt_min * 60)

    plt.plot(np.nan, np.nan, '-w', lw=1, label=f'start: {t_start}')

    plt.plot(bins[1:], hist_rt, ds='steps', c='b', lw=1,
                 label=f'{etype}, {rt_min:.2f} mins')

    plt.xlabel(etype, ha='right', x=1)
    plt.ylabel('cts / sec', ha='right', y=1)
    plt.legend(loc=1, fontsize=12)
    plt.tight_layout()
    plt.savefig('./plots/CalSpectrum.png')
#     plt.show()


def show_wfs(dg):
    """
    show waveforms in different enery regions.
    use the hit file to select events
    """
    # get file list and load hit data
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    hit_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
    df_hit = lh5.load_dfs(hit_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/hit')
    print(df_hit)
    print(df_hit.columns)

    # settings
    etype = 'trapEmax_cal'
    nwfs = 20
    # elo, ehi, epb = 0, 100, 0.2 # low-e region
    elo, ehi, epb = 0, 20, 0.2 # noise region
    # elo, ehi, epb = 1458, 1468, 1 # good physics events
    # elo, ehi, epb = 6175, 6250, 1 # overflow peak
    # elo, ehi, epb = 5000, 5200, 0.2 # lower overflow peak

    # # diagnostic plot
    # hE, xE, vE = pgh.get_hist(df_hit[etype], range=(elo, ehi), dx=epb)
    # plt.plot(xE[1:], hE, c='b', ds='steps')
    # plt.show()
    # exit()

    # select waveforms
    idx = df_hit[etype].loc[(df_hit[etype] >= elo) &
                            (df_hit[etype] <= ehi)].index[:nwfs]
    raw_store = lh5.Store()
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    f_raw = raw_list.values[0] # fixme, only works for one file rn
    data_raw = raw_store.read_object(tb_name, f_raw, start_row=0, n_rows=idx[-1]+1)

    wfs_all = data_raw['waveform']['values'].nda
    wfs = wfs_all[idx.values, :]
    ts = np.arange(0, wfs.shape[1], 1)

    # plot wfs
    for iwf in range(wfs.shape[0]):
        plt.plot(ts, wfs[iwf,:], lw=1)

    plt.xlabel('time (clock ticks)', ha='right', x=1)
    plt.ylabel('ADC', ha='right', y=1)
    plt.show()
    # plt.savefig('./plots/noise_wfs.png', dpi=300)
    # plt.cla()


def data_cleaning(dg):
    """
    using parameters in the hit file, plot 1d and 2d spectra to find cut values.

    columns in file:
        ['trapE', 'bl', 'bl_sig', 'A_10', 'AoE', 'packet_id', 'ievt', 'energy',
        'energy_first', 'timestamp', 'crate', 'card', 'channel', 'energy_cal',
        'trapE_cal']

    note, 'energy_first' from first value of energy gate.
    """
    i_plot = 0 # run all plots after this number

    # get file list and load hit data
    lh5_dir = dg.lh5_user_dir if user else dg.lh5_dir
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    hit_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
    df_hit = lh5.load_dfs(hit_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/hit')
    # print(df_hit)
    print(df_hit.columns)

    # get info about df -- 'describe' is very convenient
    dsc = df_hit[['bl','bl_sig','A_10','ts_sec']].describe()
    # print(dsc)
    # exit()

    if i_plot <= 0:
        # bl vs energy

        elo, ehi, epb = 0, 50, 1
        blo, bhi, bpb = 0, 10000, 100
        nbx = int((ehi-elo)/epb)
        nby = int((bhi-blo)/bpb)

        h = plt.hist2d(df_hit['trapEmax_cal'], df_hit['bl'], bins=[nbx,nby],
                       range=[[elo, ehi], [blo, bhi]], cmap='jet')

        cb = plt.colorbar(h[3], ax=plt.gca())
        plt.xlabel('trapEmax_cal', ha='right', x=1)
        plt.ylabel('bl', ha='right', y=1)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./plots/oppi_bl_vs_e.png', dpi=300)
        cb.remove()
        plt.cla()

        # make a formal baseline cut from 1d histogram
        hE, bins, vE = pgh.get_hist(df_hit['bl'], range=(blo, bhi), dx=bpb)
        xE = bins[1:]
        plt.semilogy(xE, hE, c='b', ds='steps')

        bl_cut_lo, bl_cut_hi = 8000, 9500
        plt.axvline(bl_cut_lo, c='r', lw=1)
        plt.axvline(bl_cut_hi, c='r', lw=1)

        plt.xlabel('bl', ha='right', x=1)
        plt.ylabel('counts', ha='right', y=1)
        # plt.show()
        plt.savefig('./plots/oppi_bl_cut.png')
        plt.cla()


    if i_plot <= 1:
        # A_10/trapEmax_cal vs trapEmax_cal (A/E vs E)

        # use baseline cut
        df_cut = df_hit.query('bl > 8000 and bl < 9500').copy()

        # add new A/E column
        df_cut['aoe'] = df_cut['A_10'] / df_cut['trapEmax_cal']

        # alo, ahi, apb = -1300, 350, 1
        # elo, ehi, epb = 0, 250, 1
        alo, ahi, apb = 0, 0.4, 0.005
        # elo, ehi, epb = 0, 3000, 10
        elo, ehi, epb = 0, 6000, 10

        nbx = int((ehi-elo)/epb)
        nby = int((ahi-alo)/apb)

        h = plt.hist2d(df_cut['trapEmax_cal'], df_cut['aoe'], bins=[nbx,nby],
                       range=[[elo, ehi], [alo, ahi]], cmap='jet', norm=LogNorm())

        plt.xlabel('trapEmax_cal', ha='right', x=1)
        plt.ylabel('A/E', ha='right', y=1)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./plots/oppi_aoe_vs_e.png', dpi=300)
        plt.cla()


    if i_plot <= 2:
        # show effect of baseline cut on low-energy spectrum

        df_cut = df_hit.query('bl > 8000 and bl < 9500')

        etype = 'trapEmax_cal'
        elo, ehi, epb = 0, 250, 0.5

        # no cuts
        h1, x1, v1 = pgh.get_hist(df_hit[etype], range=(elo, ehi), dx=epb)
        x1 = x1[1:]
        plt.plot(x1, h1, c='k', lw=1, ds='steps', label='raw')

        # baseline cut
        h2, x2, v2 = pgh.get_hist(df_cut[etype], range=(elo, ehi), dx=epb)
        plt.plot(x1, h2, c='b', lw=1, ds='steps', label='bl cut')

        plt.xlabel(etype, ha='right', x=1)
        plt.ylabel('counts', ha='right', y=1)
        plt.legend()
        # plt.show()
        plt.savefig('./plots/oppi_lowe_cut.png')
        plt.cla()

    if i_plot <= 3:
        # show DCR vs E
        etype = 'trapEmax_cal'
        elo, ehi, epb = 0, 6000, 10
        dlo, dhi, dpb = -1000, 1000, 10

        nbx = int((ehi-elo)/epb)
        nby = int((dhi-dlo)/dpb)

        h = plt.hist2d(df_cut['trapEmax_cal'], df_cut['dcr'], bins=[nbx,nby],
                       range=[[elo, ehi], [dlo, dhi]], cmap='jet', norm=LogNorm())

        plt.xlabel('trapEmax_cal', ha='right', x=1)
        plt.ylabel('DCR', ha='right', y=1)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./plots/oppi_dcr_vs_e.png', dpi=300)
        plt.cla()


def peak_drift(dg):
    """
    show any drift of the 1460 peak (5 minute bins)
    """
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    hit_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
    df_hit = lh5.load_dfs(hit_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/hit')
    df_hit.reset_index(inplace=True)
    rt_min = dg.fileDB['runtime'].sum()
    print(f'runtime: {rt_min:.2f} min')

    # settings
    elo, ehi, epb, etype = 1450, 1470, 1, 'trapEmax_cal'
    df_hit = df_hit.query(f'trapEmax_cal > {elo} and trapEmax_cal < {ehi}').copy()

    # # diagnostic plot
    # hE, xE, vE = pgh.get_hist(df_hit[etype], range=(elo, ehi), dx=epb)
    # plt.plot(xE[1:], hE, c='b', ds='steps')
    # plt.show()

    t0 = df_hit['ts_glo'].values[0]
    df_hit['ts_adj'] = (df_hit['ts_glo'] - t0) / 60 # minutes after 0

    tlo, thi, tpb = 0, df_hit['ts_adj'].max(), 1

    nbx = int((thi-tlo)/tpb)
    nby = int((ehi-elo)/epb)

    h = plt.hist2d(df_hit['ts_adj'], df_hit['trapEmax_cal'], bins=[nbx,nby],
                   range=[[tlo, thi], [elo, ehi]], cmap='jet')

    plt.xlabel(f'Time ({tpb:.1f} min/bin)', ha='right', x=1)
    plt.ylabel('trapEmax_cal', ha='right', y=1)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./plots/oppi_1460_drift.png', dpi=300)


def pole_zero(dg):
    """
    NOTE: I think this result might be wrong, for the CAGE amp it should be
    around 250 usec.  Need to check.
    """
    # load hit data
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    hit_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
    df_hit = lh5.load_dfs(hit_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/hit')
    df_hit.reset_index(inplace=True)
    rt_min = dg.fileDB['runtime'].sum()
    # print(f'runtime: {rt_min:.2f} min')

    # load waveforms
    etype = 'trapEmax_cal'
    nwfs = 20
    elo, ehi = 1455, 1465

    # select waveforms
    idx = df_hit[etype].loc[(df_hit[etype] >= elo) &
                            (df_hit[etype] <= ehi)].index[:nwfs]
    raw_store = lh5.Store()
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    f_raw = raw_list.values[0] # fixme, only works for one file rn
    data_raw = raw_store.read_object(tb_name, f_raw, start_row=0, n_rows=idx[-1]+1)

    wfs_all = data_raw['waveform']['values'].nda
    wfs = wfs_all[idx.values, :]
    df_wfs = pd.DataFrame(wfs)
    # print(df_wfs)

    # simple test function to compute pole-zero constant for a few wfs.
    # the final one should become a dsp processor
    clock = 1e8 # 100 MHz
    istart = 5000
    iwinlo, iwinhi, iwid = 500, 2500, 20 # two-point slope
    # ts = np.arange(istart, df_wfs.shape[1]-1, 1) / 1e3 # usec
    ts = np.arange(0, df_wfs.shape[1]-1-istart, 1) / 1e3 # usec

    def get_rc(row):
        # two-point method
        wf = row[istart:-1].values
        wflog = np.log(wf)
        win1 = np.mean(np.log(row[istart+iwinlo : istart+iwinlo+iwid]))
        win2 = np.mean(np.log(row[istart+iwinhi : istart+iwinhi+iwid]))
        slope = (win2 - win1) / (ts[iwinhi] - ts[iwinlo])
        tau = 1/slope

        # # diagnostic plot: check against expo method
        # guess_tau = 60
        # a = wf.max()
        # expdec = lambda x : a * np.exp(-x / guess_tau)
        # logdec = lambda x : np.log(a * np.exp(-x / guess_tau))
        # slopeway = lambda x: wflog[0] + x / tau
        # plt.plot(ts, wflog, '-r', lw=1)
        # plt.plot(ts, logdec(ts), '-b', lw=1)
        # plt.plot(ts, slopeway(ts), '-k', lw=1)
        # plt.show()
        # exit()

        return tau

        # return tau

    res = df_wfs.apply(get_rc, axis=1)

    tau_avg, tau_std = res.mean(), res.std()
    print(f'average RC decay constant: {tau_avg:.2f} pm {tau_std:.2f}')


def label_alpha_runs(dg):
    """
    example of filtering the fileDB for alpha runs, adding new information
    from a text file, and saving it to a new file, alphaDB.
    """
    # load fileDB
    df_fileDB = pd.read_hdf(dg.f_fileDB)
    
    # print(df_fileDB.columns)
    # ['unique_key', 'YYYY', 'mm', 'dd', 'cycle', 'daq_dir', 'daq_file', 'run',
    #  'runtype', 'detector', 'skip', 'raw_file', 'raw_path', 'dsp_file',
    #  'dsp_path', 'hit_file', 'hit_path', 'startTime', 'threshold', 'daq_gb',
    #  'stopTime', 'runtime']
    
    view_cols = ['unique_key', 'run', 'cycle', 'runtype', 'detector', 'skip']
    # print(df_fileDB[view_cols].to_string())
    
    # select alpha files only
    df_alphaDB = df_fileDB.query("runtype == 'alp'")
    # print(df_alphaDB[view_cols])
    
    # load our beam position info -- manually curated list
    df_beam = pd.read_csv('scan_key.txt')
    # print(df_beam)
    
    # add beam position columns to df_alphaDB (our subset)
    g = df_alphaDB.groupby(['run'])
    
    def add_info(df_run, df_beam):
        run = df_run.iloc[0]['run']
        pos_vals = df_beam.loc[df_beam.run == run]
        if len(pos_vals) == 0:
            df_run['radius'] = np.nan
            df_run['angle'] = np.nan
        else:
            df_run['radius'] = pos_vals.iloc[0]['radius']
            df_run['angle'] = pos_vals.iloc[0]['angle']
        return df_run
        
    df_alphaDB = g.apply(add_info, df_beam)
    
    view_cols += ['radius','angle']
    print(df_alphaDB[view_cols].to_string())
    
    # two options to proceed here:
    # 1. move this function to setup.py and have it overwrite the fileDB
    # 2. just write df_alphaDB to a separate analysis file here
    
    df_alphaDB.to_hdf('alphaDB.h5', key='alphaDB')


if __name__=="__main__":
    main()
