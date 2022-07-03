#!/usr/bin/env python3
import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../../pygama/clint.mpl')
from matplotlib.colors import LogNorm

import pygama.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf

def main():
    """
    tasks:
    - create combined dsp file (w/ onbd E and timestamp)
    - calibrate one run (onboard E, pygama trapE)
    - show resolution of 1460, 238, etc peaks
    - low-energy noise analysis (use dsp params)
    - determine pz correction value for OPPI
    """
    # show_groups()
    # show_raw_spectrum()
    # dsp_to_hit()
    # show_cal_spectrum()
    # get_resolution()
    # show_wfs()
    data_cleaning()
    # show_lowe_wfs()


def show_groups():
    """
    show example of accessing the names of the HDF5 groups in our LH5 files
    """
    f_raw = '/Users/wisecg/Data/OPPI/raw/oppi_run0_cyc2027_raw.lh5'
    f_dsp = '/Users/wisecg/Data/OPPI/dsp/oppi_run0_cyc2027_dsp_test.lh5'
    f_hit = '/Users/wisecg/Data/OPPI/hit/oppi_run0_cyc2027_hit.lh5'

    # h5py method
    # hf = h5py.File(f_raw)
    # hf = h5py.File(f_dsp)

    # some examples of navigating the groups
    # print(hf.keys())
    # print(hf['ORSIS3302DecoderForEnergy/raw'].keys())
    # print(hf['ORSIS3302DecoderForEnergy/raw/waveform'].keys())
    # exit()

    # lh5 method
    sto = lh5.Store()
    groups = sto.ls(f_dsp)
    data = sto.read_object('ORSIS3302DecoderForEnergy/raw', f_dsp)

    # testing -- make sure data columns all have same shape
    for col in data.keys():
        print(col, data[col].nda.shape)

    # directly access timestamps in a raw file w/o loading all the wfs
    # groups = sto.ls(f_raw, 'ORSIS3302DecoderForEnergy/raw/')
    # data = sto.read_object('ORSIS3302DecoderForEnergy/raw/timestamp', f_raw)
    # ts = data.nda


    # check pandas conversion
    df_dsp = data.get_dataframe()
    print(df_dsp.columns)
    print(df_dsp)


def linear_cal(etype):
    """
    get calibration constants for an energy estimator
    """
    peak_table = {
        '212Pb':238.6, '214Pb':351.9, 'beta+':511.0, '208Tl':583.2,
        '214Bi':609.3, '228Ac':911.2, '228Ac':969.0, '214Bi':1120.3,
        '40K':1460.8, '214Bi':1764.5, '208Tl':2614.5
    }

    # these will be different for each experiment -- user has to find these
    expected_peaks = ['212Pb', '40K']
    data_vals = {"energy":[506615, 3.08943e6], "trapE":[533.798, 3255.82]}

    # pass-1 fit is simple
    raw_peaks = np.array(data_vals[etype])
    true_peaks = np.array([peak_table[pk] for pk in expected_peaks])
    pfit = np.polyfit(raw_peaks, true_peaks, 1)

    return pfit


def show_raw_spectrum():
    """
    show spectrum w/ onbd energy and trapE
    - get calibration constants for onbd energy and 'trapE' energy
    - TODO: fit each expected peak and get resolution vs energy
    """
    f_dsp = '/Users/wisecg/Data/OPPI/dsp/oppi_run0_cyc2027_dsp_test.lh5'

    # we will probably make this part simpler in the near future
    sto = lh5.Store()
    groups = sto.ls(f_dsp)
    data = sto.read_object('ORSIS3302DecoderForEnergy/raw', f_dsp)
    df_dsp = data.get_dataframe()

    # from here, we can use standard pandas to work with data
    print(df_dsp)

    # elo, ehi, epb, etype = 0, 8e6, 1000, 'energy'
    # elo, ehi, epb, etype = 0, 8e6, 1000, 'energy' # whole spectrum
    # elo, ehi, epb, etype = 0, 800000, 1000, 'energy' # < 250 keV
    elo, ehi, epb, etype = 0, 10000, 10, 'trapE'

    ene_uncal = df_dsp[etype]
    hist, bins, _ = pgh.get_hist(ene_uncal, range=(elo, ehi), dx=epb)
    bins = bins[1:] # trim zero bin, not needed with ds='steps'

    plt.plot(bins, hist, ds='steps', c='b', lw=2, label=etype)
    plt.xlabel(etype, ha='right', x=1)
    plt.ylabel('Counts', ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def dsp_to_hit():
    """
    save calibrated energies into the dsp file.
    this is a good example of adding a column, reading & writing to an LH5 file.
    """
    f_dsp = '/Users/wisecg/Data/OPPI/dsp/oppi_run0_cyc2027_dsp_test.lh5'
    f_hit = '/Users/wisecg/Data/OPPI/hit/oppi_run0_cyc2027_hit.lh5'
    sto = lh5.Store()
    groups = sto.ls(f_dsp)
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    data = sto.read_object(tb_name, f_dsp)
    df_dsp = data.get_dataframe()

    # add a new column for each energy estimator of interest
    for etype in ['energy', 'trapE']:
        ecal_name = etype + '_cal'
        pfit = linear_cal(etype)
        df_dsp[ecal_name] = df_dsp[etype] * pfit[0] + pfit[1]

        e_cal_lh5 = lh5.Array(df_dsp[ecal_name].values, attrs={'units':'keV'})
        data.add_field(f'{etype}_cal', e_cal_lh5)

    # write to hit file.  delete if exists, LH5 overwrite is broken rn
    if os.path.exists(f_hit):
        os.remove(f_hit)
    sto.write_object(data, tb_name, f_hit)


def show_cal_spectrum():
    """
    """
    f_hit = '/Users/wisecg/Data/OPPI/hit/oppi_run0_cyc2027_hit.lh5'
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    sto = lh5.Store()
    groups = sto.ls(f_hit)
    data = sto.read_object(tb_name, f_hit)
    df_hit = data.get_dataframe()

    print(df_hit)

    # energy in keV
    elo, ehi, epb = 0, 3000, 0.5

    # choose energy estimator
    etype = 'energy_cal'
    # etype = 'trapE_cal'

    hist, bins, _ = pgh.get_hist(df_hit[etype], range=(elo, ehi), dx=epb)
    bins = bins[1:] # trim zero bin, not needed with ds='steps'

    plt.plot(bins, hist, ds='steps', c='b', lw=2, label=etype)
    plt.xlabel(etype, ha='right', x=1)
    plt.ylabel('Counts', ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_resolution():
    """
    """
    # load hit file
    f_hit = '/Users/wisecg/Data/OPPI/hit/oppi_run0_cyc2027_hit.lh5'
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    sto = lh5.Store()
    groups = sto.ls(f_hit)
    data = sto.read_object(tb_name, f_hit)
    df_hit = data.get_dataframe()

    # load parameters
    e_peak = 1460.8
    etype = 'trapE_cal'
    # etype = 'energy_cal'
    elo, ehi, epb = 1445, 1475, 0.2

    # get histogram
    hE, bins, vE = pgh.get_hist(df_hit[etype], range=(elo, ehi), dx=epb)
    xE = bins[1:]

    # simple numerical width
    i_max = np.argmax(hE)
    h_max = hE[i_max]
    upr_half = xE[(xE > xE[i_max]) & (hE <= h_max/2)][0]
    bot_half = xE[(xE < xE[i_max]) & (hE >= h_max/2)][0]
    fwhm = upr_half - bot_half
    sig = fwhm / 2.355

    # # fit to gaussian: amp, mu, sig, bkg
    # amp = h_max * fwhm
    # bg0 = np.mean(hE[:20])
    # x0 = [amp, xE[i_max], sig, bg0]
    # xF, xF_cov = pgf.fit_hist(pgf.gauss_bkg, hE, bins, var=vE, guess=x0)
    # fit_func = pgf.gauss_bkg

    # fit to radford peak: mu, sigma, hstep, htail, tau, bg0, amp
    amp = h_max * fwhm
    hstep = 0.001 # fraction that the step contributes
    htail = 0.1
    tau = 10
    bg0 = np.mean(hE[:20])
    x0 = [xE[i_max], sig, hstep, htail, tau, bg0, amp]
    xF, xF_cov = pgf.fit_hist(pgf.radford_peak, hE, bins, var=vE, guess=x0)
    fit_func = pgf.radford_peak

    xF_err = np.sqrt(np.diag(xF_cov))
    chisq = []
    for i, h in enumerate(hE):
        model = fit_func(xE[i], *xF)
        diff = (model - h)**2 / model
        chisq.append(abs(diff))

    # collect results (for output, should use a dict or DataFrame)
    e_fit = xF[0]
    fwhm_fit = xF[1] * 2.355 #  * e_peak / e_fit
    print(fwhm, fwhm_fit)
    fwhmerr = xF_err[1] * 2.355 * e_peak / e_fit
    rchisq = sum(np.array(chisq) / len(hE))

    # plotting
    plt.plot(xE, hE, ds='steps', c='b', lw=2, label=etype)

    # peak shape
    plt.plot(xE, fit_func(xE, *x0), '-', c='orange', alpha=0.5,
             label='init. guess')
    plt.plot(xE, fit_func(xE, *xF), '-r', alpha=0.8, label='peakshape fit')
    plt.plot(np.nan, np.nan, '-w', label=f'mu={e_fit:.1f}, fwhm={fwhm_fit:.2f}')

    plt.xlabel(etype, ha='right', x=1)
    plt.ylabel('Counts', ha='right', y=1)
    plt.legend(loc=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./plots/resolution_1460_{etype}.pdf')
    plt.cla()


def show_wfs():
    """
    show low-e waveforms in different enery regions
    """
    f_raw = '/Users/wisecg/Data/OPPI/raw/oppi_run0_cyc2027_raw.lh5'
    f_hit = '/Users/wisecg/Data/OPPI/hit/oppi_run0_cyc2027_hit.lh5'

    # use the hit file to select events
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    hit_store = lh5.Store()
    data = hit_store.read_object(tb_name, f_hit)
    df_hit = data.get_dataframe()

    # settings
    nwfs = 20
    elo, ehi, epb = 0, 100, 0.2

    # etype = 'energy_cal'  # noise stops @ 18 keV
    # noise_lo, noise_hi, phys_lo, phys_hi = 10, 15, 25, 30

    etype = 'trapE_cal' # noise stops @ 35 keV
    noise_lo, noise_hi, phys_lo, phys_hi = 25, 30, 40, 45

    # # diagnostic plot
    # hE, bins, vE = pgh.get_hist(df_hit[etype], range=(elo, ehi), dx=epb)
    # xE = bins[1:]
    # plt.plot(xE, hE, c='b', ds='steps')
    # plt.show()
    # exit()

    # select noise and phys events
    idx_noise = df_hit[etype].loc[(df_hit[etype] > noise_lo) &
                                  (df_hit[etype] < noise_hi)].index[:nwfs]

    idx_phys = df_hit[etype].loc[(df_hit[etype] > phys_lo) &
                                 (df_hit[etype] < phys_hi)].index[:nwfs]

    # print(df_hit.loc[idx_noise])
    # print(df_hit.loc[idx_phys])

    # get phys waveforms, normalized by max value
    i_max = max(idx_noise[-1], idx_phys[-1])

    raw_store = lh5.Store()
    data_raw = raw_store.read_object(tb_name, f_raw, start_row=0, n_rows=i_max+1)

    wfs = data_raw['waveform']['values'].nda
    wfs_noise = wfs[idx_noise.values, :]
    wfs_phys = wfs[idx_phys.values, :]
    ts = np.arange(0, wfs_noise.shape[1], 1)

    # noise wfs
    for iwf in range(wfs_noise.shape[0]):
        plt.plot(ts, wfs_noise[iwf,:], lw=1)

    # # phys wfs
    # for iwf in range(wfs_phys.shape[0]):
    #     plt.plot(ts, wfs_phys[iwf,:], lw=1)

    plt.xlabel('time (clock ticks)', ha='right', x=1)
    plt.ylabel('ADC', ha='right', y=1)
    # plt.show()
    plt.savefig('./plots/noise_wfs.png', dpi=300)
    plt.cla()


def data_cleaning():
    """
    using parameters in the hit file, plot 1d and 2d spectra to find cut values.

    columns in file:
        ['trapE', 'bl', 'bl_sig', 'A_10', 'AoE', 'packet_id', 'ievt', 'energy',
        'energy_first', 'timestamp', 'crate', 'card', 'channel', 'energy_cal',
        'trapE_cal']

    note, 'energy_first' from first value of energy gate.
    """
    i_plot = 3 # run all plots after this number

    f_hit = '/Users/wisecg/Data/OPPI/hit/oppi_run0_cyc2027_hit.lh5'

    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    hit_store = lh5.Store()
    data = hit_store.read_object(tb_name, f_hit)
    df_hit = data.get_dataframe()

    # get info about df -- 'describe' is very convenient
    dsc = df_hit[['bl','bl_sig','A_10','energy_first','timestamp']].describe()
    # print(dsc)
    # print(dsc.loc['min','bl'])

    # correct energy_first (inplace) to allow negative values
    df_hit['energy_first'] = df_hit['energy_first'].astype(np.int64)
    efirst = df_hit['energy_first'].values
    idx = np.where(efirst > 4e9)
    eshift = efirst[idx] - 4294967295
    efirst[idx] = eshift
    # print(df_hit[['energy','energy_first','bl']])

    if i_plot <= 0:
        # bl vs energy

        elo, ehi, epb = 0, 250, 1
        blo, bhi, bpb = 54700, 61400, 100
        nbx = int((ehi-elo)/epb)
        nby = int((bhi-blo)/bpb)

        h = plt.hist2d(df_hit['trapE_cal'], df_hit['bl'], bins=[nbx,nby],
                       range=[[elo, ehi], [blo, bhi]], cmap='jet')

        cb = plt.colorbar(h[3], ax=plt.gca())
        plt.xlabel('trapE_cal', ha='right', x=1)
        plt.ylabel('bl', ha='right', y=1)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./plots/bl_vs_e.png', dpi=300)
        cb.remove()
        plt.cla()

        # make a formal baseline cut from 1d histogram
        hE, bins, vE = pgh.get_hist(df_hit['bl'], range=(blo, bhi), dx=bpb)
        xE = bins[1:]
        plt.semilogy(xE, hE, c='b', ds='steps')

        bl_cut_lo, bl_cut_hi = 57700, 58500
        plt.axvline(bl_cut_lo, c='r', lw=1)
        plt.axvline(bl_cut_hi, c='r', lw=1)

        plt.xlabel('bl', ha='right', x=1)
        plt.ylabel('counts', ha='right', y=1)
        # plt.show()
        plt.savefig('./plots/bl_cut.pdf')
        plt.cla()

    if i_plot <= 1:
        # energy_first vs. E

        flo, fhi, fpb = -565534, 70000, 1000
        elo, ehi, epb = 0, 250, 1

        nbx = int((ehi-elo)/epb)
        nby = int((fhi-flo)/fpb)

        h = plt.hist2d(df_hit['trapE_cal'], df_hit['energy_first'], bins=[nbx,nby],
                       range=[[elo, ehi], [flo, fhi]], cmap='jet', norm=LogNorm())

        cb = plt.colorbar(h[3], ax=plt.gca())
        plt.xlabel('trapE_cal', ha='right', x=1)
        plt.ylabel('energy_first', ha='right', y=1)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./plots/efirst_vs_e.png', dpi=300)
        cb.remove()
        plt.cla()

        # make a formal baseline cut from 1d histogram
        flo, fhi, fpb = -20000, 20000, 100
        hE, xE, vE = pgh.get_hist(df_hit['energy_first'], range=(flo, fhi),
                                    dx=fpb)
        xE = xE[1:]
        plt.semilogy(xE, hE, c='b', ds='steps')

        ef_cut_lo, ef_cut_hi = -5000, 4000
        plt.axvline(ef_cut_lo, c='r', lw=1)
        plt.axvline(ef_cut_hi, c='r', lw=1)

        plt.xlabel('energy_first', ha='right', x=1)
        plt.ylabel('counts', ha='right', y=1)
        # plt.show()
        plt.savefig('./plots/efirst_cut.pdf')
        plt.cla()

    if i_plot <= 3:
        # trapE_cal - energy_cal vs trapE_cal

        # use baseline cut
        df_cut = df_hit.query('bl > 57700 and bl < 58500').copy()

        # add new diffE column
        df_cut['diffE'] = df_cut['trapE_cal'] - df_cut['energy_cal']

        elo, ehi, epb = 0, 3000, 1
        dlo, dhi, dpb = -10, 10, 0.1

        nbx = int((ehi-elo)/epb)
        nby = int((dhi-dlo)/dpb)

        h = plt.hist2d(df_cut['trapE_cal'], df_cut['diffE'], bins=[nbx,nby],
                       range=[[elo, ehi], [dlo, dhi]], cmap='jet', norm=LogNorm())

        plt.xlabel('trapE_cal', ha='right', x=1)
        plt.ylabel('diffE (trap-onbd)', ha='right', y=1)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./plots/diffE.png', dpi=300)
        plt.cla()


    if i_plot <= 4:
        # A_10/trapE_cal vs trapE_cal (A/E vs E)

        # i doubt we want to introduce a pulse shape cut at this point,
        # since i'm tuning on bkg data and we don't know a priori what (if any)
        # features the Kr waveforms will have.  also, the efficiency as a
        # function of energy would have to be determined, which is hard.
        # so this is just for fun.

        # use baseline cut
        df_cut = df_hit.query('bl > 57700 and bl < 58500').copy()

        # add new A/E column
        df_cut['aoe'] = df_cut['A_10'] / df_cut['trapE_cal']

        # alo, ahi, apb = -1300, 350, 1
        # elo, ehi, epb = 0, 250, 1
        alo, ahi, apb = -0.5, 5, 0.05
        elo, ehi, epb = 0, 50, 0.2

        nbx = int((ehi-elo)/epb)
        nby = int((ahi-alo)/apb)

        h = plt.hist2d(df_cut['trapE_cal'], df_cut['aoe'], bins=[nbx,nby],
                       range=[[elo, ehi], [alo, ahi]], cmap='jet', norm=LogNorm())

        plt.xlabel('trapE_cal', ha='right', x=1)
        plt.ylabel('A/E', ha='right', y=1)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./plots/aoe_vs_e_lowe.png', dpi=300)
        plt.cla()



    if i_plot <= 5:
        # show effect of cuts on energy spectrum

        # baseline cut and efirst cut are very similar
        df_cut = df_hit.query('bl > 57700 and bl < 58500')
        # df_cut = df_hit.query('energy_first > -5000 and energy_first < 4000')

        etype = 'trapE_cal'
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
        plt.savefig('./plots/cut_spectrum.pdf')
        plt.cla()


def show_lowe_wfs():
    """
    separate function to show really low-e waveforms after the data cleaning cut
    """
    f_raw = '/Users/wisecg/Data/OPPI/raw/oppi_run0_cyc2027_raw.lh5'
    f_hit = '/Users/wisecg/Data/OPPI/hit/oppi_run0_cyc2027_hit.lh5'

    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    hit_store = lh5.Store()
    data = hit_store.read_object(tb_name, f_hit)
    df_hit = data.get_dataframe()

    # correct energy_first (inplace) to allow negative values
    df_hit['energy_first'] = df_hit['energy_first'].astype(np.int64)
    efirst = df_hit['energy_first'].values
    idx = np.where(efirst > 4e9)
    eshift = efirst[idx] - 4294967295
    efirst[idx] = eshift

    nwfs = 40
    elo, ehi, epb = 1, 10, 0.1
    blo, bhi = 57700, 58500 # cut values
    etype = 'trapE_cal' # noise stops @ 35 keV

    idx_lowe = df_hit[etype].loc[(df_hit[etype] > elo) &
                                 (df_hit[etype] < ehi) &
                                 (df_hit.bl > blo) & (df_hit.bl < bhi)]
    idx_lowe = idx_lowe.index[:nwfs]
    # print(df_hit.loc[idx_lowe])

    # get phys waveforms, normalized by max value
    i_max = idx_lowe[-1]

    raw_store = lh5.Store()
    data_raw = raw_store.read_object(tb_name, f_raw, start_row=0, n_rows=i_max+1)

    wfs = data_raw['waveform']['values'].nda
    wfs_lowe = wfs[idx_lowe.values, :]
    ts = np.arange(0, wfs_lowe.shape[1], 1)

    # plot wfs
    for iwf in range(wfs_lowe.shape[0]):
        plt.plot(ts, wfs_lowe[iwf,:], lw=1, alpha=0.5)

    plt.xlabel('time (clock ticks)', ha='right', x=1)
    plt.ylabel('ADC', ha='right', y=1)
    # plt.show()
    plt.savefig('./plots/lowe_wfs.png', dpi=300)
    plt.cla()



if __name__=="__main__":
    main()
