#!/usr/bin/env python3
import os
import re
import time
import json
import argparse
import pandas as pd
import numpy as np
from pprint import pprint
from datetime import datetime
import itertools
from scipy.optimize import curve_fit

import tinydb as db
from tinydb.storages import MemoryStorage

import matplotlib
if os.environ.get('HOSTNAME'): # cenpa-rocks
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../clint.mpl')

from pygama import DataGroup
from pygama.io.orcadaq import parse_header
import pygama.io.lh5 as lh5
import pygama.analysis.metadata as pmd
import pygama.analysis.histograms as pgh
import pygama.analysis.calibration as pgc
import pygama.analysis.peak_fitting as pgf

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    tqdm.pandas() # suppress annoying FutureWarning


def main():
    doc="""
    === pygama: energy_cal.py ====================================================

    energy calibration app

    - Initial guesses are determined by running 'check_raw_spectrum'
    - Uses a DataGroup to organize files and processing.
    - Metadata is handled in JSON format with 'legend-metadata' conventions.

    === T. Mathew, C. Wiseman (UW) =============================================
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    # initial setup
    arg('--init_db', action=st, help='initialize primary ecal output file')
    arg('--raw', action=st, help='display/save uncalibrated energy histogram')

    # primary operations
    arg('-q', '--query', nargs=1, type=str,
        help="select file group to calibrate: -q 'run==1' ")
    arg('-p1', '--peakdet', action=st, help='first pass: peak detection')
    arg('-p2', '--peakfit', action=st, help='second pass: individual peak fit')
    arg('--run_all', action=st, help='run all passes, write to db')

    # options
    arg('-w', '--write_db', action=st, help='write results to ecalDB file')
    arg('-s', '--show_db', action=st, help='show ecalDB results file')
    arg('-p', '--show_plot', action=st, help='show debug plot')
    arg('-o', '--order', nargs=1, type=int, help='set cal poly order, default: 2')
    arg('-b', '--batch', action=st, help="batch mode: save & don't display plots")
    arg('--show_config', action=st, help='show current configuration')
    arg('--indiv', action=st, help='calibrate individual cycle files')
    arg('--match', nargs=1, type=str, help='set peak match mode (default: ratio)')
    arg('--epar', nargs=1, type=str,
        help="specify raw energy parameters: --epar 'asd sdf dfg' ")
    arg('--group', nargs=1, type=str,
        help="select alternate groupby: --group 'YYYY run' ")

    args = par.parse_args()

    # -- setup --

    # load main DataGroup, select files to calibrate
    dg = DataGroup('cage.json', load=True)
    if args.query:
        que = args.query[0]
        dg.fileDB.query(que, inplace=True)
    else:
        dg.fileDB = dg.fileDB[-1:]

    view_cols = ['run','cycle','daq_file','runtype','startTime','threshold',
                 'stopTime','runtime']
    print(dg.fileDB[view_cols].to_string())
    print(len(dg.fileDB))
    # exit()

    # merge main and ecal config JSON as dicts
    config = dg.config
    with open(config['ecal_config']) as f:
        config = {**dg.config, **json.load(f)}

    # initialize JSON output file.  only run this once
    if args.init_db:
        init_ecaldb(config)
    try:
        # load ecal db in memory s/t the pretty on-disk formatting isn't changed
        db_ecal = db.TinyDB(storage=MemoryStorage)
        with open(config['ecaldb']) as f:
            raw_db = json.load(f)
            db_ecal.storage.write(raw_db)
    except:
        print('JSON database file not found or corrupted.  Rerun --init_db')
        exit()

    # set additional options, augmenting the config dict
    config['gb_cols'] = args.group.split(' ') if args.group else ['run']
    config['rawe'] = args.epar[0].split(' ') if args.epar else config['rawe_default']
    config['match_mode'] = args.match if args.match else 'first'
    config['batch_mode'] = True if args.batch else False
    config['indiv'] = True if args.indiv else False
    config['show_plot'] = True if args.show_plot else False
    config['write_db'] = True if args.write_db else False
    config['pol_order'] = args.order if args.order else 2
    config['mp_tol'] = 10 # raw peaks must be within keV
    config = {**config, **db_ecal.table('_file_info').all()[0]}

    if args.show_config:
        print('Current configuration:')
        pprint(config)
        print('\n')

    # -- raw spectrum check --
    if args.raw:
        check_raw_spectrum(dg, config, db_ecal)
        exit()

    # show status
    print(f'Ready to calibrate.\n'
          f"Output file: {config['ecaldb']} \n"
          'Calibrating raw energy parameters:', config['rawe'], '\n'
          'Current DataGroup:')
    print(dg.fileDB[['run', 'cycle', 'startTime', 'runtime']])
    print('Columns:', dg.fileDB.columns.values)

    # -- main calibration routines --
    if args.show_db: show_ecaldb(config)
    if args.peakdet: run_peakdet(dg, config, db_ecal)
    if args.peakfit: run_peakfit(dg, config, db_ecal)

    if args.run_all:
        config['write_db'] = True
        run_peakdet(dg, config, db_ecal)
        run_peakfit(dg, config, db_ecal)


def init_ecaldb(config):
    """
    one-time set up of primary database file
    """
    ans = input('(Re)create main ecal JSON file?  Are you really sure? (y/n) ')
    if ans.lower() != 'y':
        exit()

    f_db = config['ecaldb'] # for pgt, should have one for each detector

    if os.path.exists(f_db):
        os.remove(f_db)

    # create the database in-memory
    db_ecal = db.TinyDB(storage=MemoryStorage)
    query = db.Query()

    # create a table with metadata (provenance) about this calibration file
    file_info = {
        "system" : config['system'],
        "cal_type" : "energy",
        "created_gmt" : datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
        "input_table" : config['input_table']
        }
    db_ecal.table('_file_info').insert(file_info)

    # pretty-print the JSON database to file
    raw_db = db_ecal.storage.read()
    pmd.write_pretty(raw_db, f_db)

    # show the file as-is on disk
    with open(f_db) as f:
        print(f.read())


def show_ecaldb(config):
    """
    $ ./energy_cal.py --show_db
    """
    # show the file as-is on disk
    with open(config['ecaldb']) as f:
        print(f.read())

    # make sure the file is usable by TinyDB
    db_ecal = db.TinyDB(storage=MemoryStorage)
    with open(config['ecaldb']) as f:
        raw_db = json.load(f)
        db_ecal.storage.write(raw_db)


def check_raw_spectrum(dg, config, db_ecal):
    """
    $ ./energy_cal.py -q 'query' --raw
    """
    import h5py

    # load energy data
    lh5_dir = os.path.expandvars(config['lh5_dir'])
    dsp_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
    raw_data = lh5.load_nda(dsp_list, config['rawe'], config['input_table'])
    runtime_min = dg.fileDB['runtime'].sum()

    print('\nShowing raw spectra ...')
    for etype in config['rawe']:
        xlo, xhi, xpb = config['init_vals'][etype]["raw_range"]

        # load energy data for this estimator
        data = raw_data[etype]

        # print columns of table
        file_info = db_ecal.table('_file_info').all()[0]
        tb_in = file_info['input_table']
        with h5py.File(dsp_list.iloc[0], 'r') as hf:
            print("LH5 columns:", list(hf[f'{tb_in}'].keys()))

        # generate histogram
        hist, bins, var = pgh.get_hist(data, range=(xlo, xhi), dx=xpb)
        bins = bins[1:] # trim zero bin, not needed with ds='steps'

        # normalize by runtime
        hist_rt = np.divide(hist, runtime_min * 60)

        print('\nPlease determine the following parameters for ecal config file:\n'
              "  - 'raw_range': Optimal binning, and hi/lo raw energy limits\n"
              "  - 'peakdet_thresh': ~1/2 the height of a target peak\n"
              "  - 'lowe_cut' energy threshold for peak detection")

        print(f'\nRaw E: {etype}, {len(data)} cts, runtime: {runtime_min:.2f} min')

        plt.plot(bins, hist_rt, ds='steps', c='b', lw=1, label=etype)
        plt.xlabel(etype, ha='right', x=1)
        plt.ylabel(f'cts/sec, {xpb}/bin', ha='right', y=1)

        if config['batch_mode']:
            plt.savefig('./plots/cal_spec_test.png')
        else:
            plt.show()
        plt.close()


def run_peakdet(dg, config, db_ecal):
    """
    $ ./energy_cal.py -q 'query' -p1 [-p : show plot]
    Run "first guess" calibration of an arbitrary energy estimator.

    NOTE: if you use too high a peakdet threshold, you may not capture
    all the lines you're looking for.  If it's too low, then you have
    to deal with more lines than you probably want for this 1-pt calibration.
    We include an option to show a diagnostic plot for this reason.
    """
    print('\nRunning peakdet ...')

    # do the analysis
    gb = dg.fileDB.groupby(config['gb_cols'])
    gb_args = [config]
    result = gb.apply(peakdet_group, *gb_args)

    # write the results
    if config['write_db']:

        # write separate tables for each energy estimator to the TinyDB
        for epar in config['rawe']:

            # format output
            epar_cols = [r for r in result.columns if epar in r]
            df_epar = result[epar_cols].copy()
            df_epar.rename(columns={c:c.split('_')[-1] for c in epar_cols},
                           inplace=True)
            df_epar['tsgen'] = int(time.time())
            df_epar.reset_index(inplace=True)
            df_epar['run'] = df_epar['run'].astype(str)

            # write the DataFrame to JSON TinyDB
            table = db_ecal.table(f'peakdet_{epar}')
            query = db.Query()
            for i, row in df_epar.iterrows():
                table.upsert(row.to_dict(), query['run'] == row['run'])

        # show in-memory state and then write to file
        pprint(db_ecal.storage.read())
        pmd.write_pretty(db_ecal.storage.read(), config['ecaldb'])


def peakdet_group(df_group, config):
    """
    Access all files in this group, load energy histograms, and find the
    "first guess" linear calibration constant.
    Return the value, and a bool indicating success.
    """
    # get file list and load energy data
    lh5_dir = os.path.expandvars(config['lh5_dir'])
    dsp_list = lh5_dir + df_group['dsp_path'] + '/' + df_group['dsp_file']

    edata = lh5.load_nda(dsp_list, config['rawe'], config['input_table'])
    print('Found energy data:', [(et, len(ev)) for et, ev in edata.items()])

    runtime_min = df_group['runtime'].sum()
    print(f'Runtime (min): {runtime_min:.2f}')

    # loop over energy estimators of interest
    pd_results = {}
    for et in config['rawe']:

        # get histogram, error, normalize by runtime, and derivative
        xlo, xhi, xpb = config['init_vals'][et]['raw_range']
        hist, bins, var = pgh.get_hist(edata[et], range=(xlo, xhi), dx=xpb)
        hist_norm = np.divide(hist, runtime_min * 60)
        hist_err = np.array([np.sqrt(hbin / (runtime_min * 60)) for hbin in hist])

        # plt.plot(bins[1:], hist_norm, ds='steps')
        # plt.show()
        # hist_deriv = np.diff(hist_norm)
        # hist_deriv = np.insert(hist_deriv, 0, 0)

        # run peakdet
        pd_thresh = config['init_vals'][et]['peakdet_thresh']
        lowe_cut = config['init_vals'][et]['lowe_cut']
        ctr_bins = (bins[:-1] + bins[1:]) / 2.
        idx = np.where(ctr_bins > lowe_cut)

        maxes, mins = pgc.peakdet(hist_norm[idx], pd_thresh, ctr_bins[idx])
        # maxes, mins = pgc.peakdet(hist_deriv[idx], pd_thresh, ctr_bins[idx])
        if len(maxes)==0:
            print('warning, no maxima!  adjust peakdet threshold')
        # print(maxes) # x (energy) [:,0], y (counts) [:,1]

        # run peak matching
        exp_pks = config['expected_peaks']
        tst_pks = config['test_peaks']
        mode = config['match_mode']
        etol = config['raw_ene_tol']
        lin_cal, mp_success = match_peaks(maxes, exp_pks, tst_pks, mode, etol)

        if config['show_plot']:

            # plot uncalibrated and calibrated energy spectrum, w/ maxima
            fig, (p0, p1) = plt.subplots(2, 1, figsize=(8, 8))

            idx = np.where(bins[1:] > lowe_cut)
            imaxes = [np.where(np.isclose(ctr_bins, x[0]))[0][0] for x in maxes]
            imaxes = np.asarray(imaxes)

            # energy, uncalibrated
            p0.plot(bins[imaxes], hist_norm[imaxes], '.m')
            p0.plot(bins[idx], hist_norm[idx], ds='steps', c='b', lw=1, label=et)
            p0.set_ylabel(f'cts/s, {xpb}/bin', ha='right', y=1)
            p0.set_xlabel(et, ha='right', x=1)

            # energy, with rough calibration
            bins_cal = bins[1:] * lin_cal
            p1.plot(bins_cal, hist_norm, ds='steps', c='b', lw=1,
                    label=f'E = {lin_cal:.3f}*{et}')

            # compute best-guess location of all peaks, assuming rough calibration
            cal_maxes = lin_cal * maxes[:, 0]
            all_pks = np.concatenate((exp_pks, tst_pks))
            raw_guesses = []
            for pk in all_pks:
                imatch = np.isclose(cal_maxes, pk, atol=config['mp_tol'])
                if imatch.any():
                    # print(pk, cal_maxes[imatch], maxes[:,0][imatch])
                    raw_guesses.append([pk, maxes[:,0][imatch][0]])
            rg = np.asarray(raw_guesses)
            rg = rg[rg[:,0].argsort()] # sort by energy

            cmap = plt.cm.get_cmap('jet', len(rg))
            for i, epk in enumerate(rg):
                idx_nearest = (np.abs(bins_cal - epk[0])).argmin()
                cts_nearest = hist_norm[idx_nearest]
                p1.plot(epk[0], cts_nearest, '.r', c=cmap(i),
                        label=f'{epk[0]:.1f} keV')

            p1.set_xlabel(f'{et}, pass-1 cal', ha='right', x=1)
            p1.set_ylabel(f'cts/s, {xpb} kev/bin', ha='right', y=1)
            p1.legend(fontsize=10)

            if config['batch_mode']:
                plt.savefig(f'./plots/peakdet_cal_{et}.pdf')
            else:
                plt.show()

        pd_results[f'{et}_lincal'] = lin_cal
        pd_results[f'{et}_lcpass'] = str(mp_success)

    return pd.Series(pd_results)


def match_peaks(maxes, exp_pks, tst_pks, mode='first', ene_tol=10):
    """
    modes:
    - 'first' : pin the first expected peak, search for the first test peak
    - 'ratio' : compute ratio match
    """
    if mode == 'first':

        # set expected and test peak
        exp_pk, tst_pk = exp_pks[0], tst_pks[0]
        # print(f'Pinning {exp_pk} looking for {tst_pk}, tolerance: {ene_tol} keV')

        # loop over raw peaks, apply a linear cal, and see if there
        # is a raw peak near the test location, within an energy tolerance
        lin_cals = []
        for xpk in maxes[:,0]:
            lin_cal = exp_pk / xpk
            cal_maxes = lin_cal * maxes[:,0]
            imatch = np.isclose(cal_maxes, tst_pk, atol=ene_tol)
            if imatch.any():
                lin_cals.append(lin_cal)

        if len(lin_cals) == 0:
            print('Found no matches!')
            return 1, False
        elif len(lin_cals) > 1:
            print('Warning, found multiple matches. Using first one...')
            print(lin_cals)
            # exit()

        # first pass calibration constant
        return lin_cals[0], True

    elif mode == 'ratio':
        """
        # NOTE: maybe we can improve on "first" mode by computing all
        # permutations and returning a calibration constant that 'averages'
        # between the most correct ones.

        Uses a peak matching algorithm based on finding ratios of uncalibrated (u)
        and "true, keV-scale" (e) energies.
        We run peakdet to find the maxima in the spectrum, then compute all ratios:
            - e1/e2, u1/u2, ..., u29/u30 etc.
        We find the subset of uncalibrated ratios (u7/u8, ... etc) that match the
        "true" ratios, and compute a calibration constant for each.

        Then for each uncalibrated ratio, we assume it to be true, then loop over
        the expected peak positions.

        We shift the uncalibrated peaks so that the true peak would be very close
        to 0, and calculate its distance from 0.  The "true" calibration constant
        will minimize this value for all ratios, and this is the one we select.
        """

        # run peakdet to identify the uncalibrated maxima
        maxes, mins = pu.peakdet(h, pk_thresh, b)
        umaxes = np.array(sorted([x[0] for x in maxes], reverse=True))

        # compute all ratios
        ecom = [c for c in it.combinations(epeaks, 2)]
        ucom = [c for c in it.combinations(umaxes, 2)]
        eratios = np.array([x[0] / x[1] for x in ecom]) # assumes x[0] > x[1]
        uratios = np.array([x[0] / x[1] for x in ucom])

        # match peaks to true energies
        cals = {}
        for i, er in enumerate(eratios):

            umatch = np.where( np.isclose(uratios, er, rtol=match_thresh) )
            e1, e2 = ecom[i][0], ecom[i][1]
            if test:
                print(f"\nratio {i} -- e1 {e1:.0f}  e2 {e2:.0f} -- {er:.3f}")

            if len(umatch[0]) == 0:
                continue

            caldists = []
            for ij, j in enumerate(umatch[0]):
                u1, u2 = ucom[j][0], ucom[j][1]
                cal = (e2 - e1) / (u2 - u1)
                cal_maxes = cal * umaxes

                # shift peaks by the amount we would expect if this const were true.
                # compute the distance (in "keV") of the peak that minimizes this.
                dist = 0
                for e_true in epeaks:
                    idx = np.abs(cal_maxes - e_true).argmin()
                    dist += np.abs(cal_maxes[idx] - e_true)
                caldists.append([cal, dist])

                if test:
                    dev = er - uratios[j] # set by match_thresh parameter
                    print(f"{ij}  {u1:-5.0f}  {u2:-5.0f}  {dev:-7.3f}  {cal:-5.2f}")

            # get the cal ratio with the smallest total dist
            caldists = np.array(caldists)
            imin = caldists[:,1].argmin()
            cals[i] = caldists[imin, :]

            if test:
                print(f"best: {imin}  {caldists[imin, 0]:.4f}  {caldists[imin, 1]:.4f}")

        if test:
            print("\nSummary:")
            for ipk in cals:
                e1, e2 = ecom[ipk][0], ecom[ipk][1]
                print(f"{ipk}  {e1:-6.1f}  {e2:-6.1f}  cal {cals[ipk][0]:.5f}")

        # get first-pass const for this DataSet
        cal_vals = np.array([c[1][0] for c in cals.items()])
        ds_cal = np.median(cal_vals)
        ds_std = np.std(cal_vals)
        print(f"Pass-1 cal for {etype}: {ds_cal:.5e} pm {ds_std:.5e}")


def run_peakfit(dg, config, db_ecal):
    """
    """
    print('\nRunning peakfit ...')

    # do the analysis
    gb = dg.fileDB.groupby(config['gb_cols'])
    gb_args = [config, db_ecal]
    result = gb.apply(peakfit_group, *gb_args)

    # write the results
    if config['write_db']:

        # write separate tables for each energy estimator to the TinyDB
        for epar in config['rawe']:

            # format output
            epar_cols = [r for r in result.columns if epar in r]
            df_epar = result[epar_cols].copy()
            df_epar.rename(columns={c:c.split('_')[-1] for c in epar_cols},
                           inplace=True)
            df_epar['tsgen'] = int(time.time())
            df_epar.reset_index(inplace=True)
            df_epar['run'] = df_epar['run'].astype(str)

            # write the DataFrame to JSON TinyDB
            table = db_ecal.table(f'peakfit_{epar}')
            query = db.Query()
            for i, row in df_epar.iterrows():
                table.upsert(row.to_dict(), query['run'] == row['run'])

        # show in-memory state and then write to file
        pprint(db_ecal.storage.read())
        pmd.write_pretty(db_ecal.storage.read(), config['ecaldb'])


def peakfit_group(df_group, config, db_ecal):
    """
    """
    # get list of peaks to look for
    epeaks = config['expected_peaks'] + config['test_peaks']
    epeaks = np.array(sorted(epeaks))

    # right now a lookup by 'run' is hardcoded.
    # in principle the lookup should stay general using the gb_cols,
    # but it's kind of hard to see right now how to write the right db queries

    gb_run = df_group['run'].unique()
    if len(gb_run) > 1:
        print("Multi-run (or other) groupbys aren't supported yet, sorry")
        exit()

    # load data
    lh5_dir = os.path.expandvars(config['lh5_dir'])
    dsp_list = lh5_dir + df_group['dsp_path'] + '/' + df_group['dsp_file']
    raw_data = lh5.load_nda(dsp_list, config['rawe'], config['input_table'])
    runtime_min = df_group['runtime'].sum()

    # loop over energy estimators of interest
    pf_results = {}
    for et in config['rawe']:

        # load first-guess calibration constant from its table in the DB
        db_table = db_ecal.table(f'peakdet_{et}').all()
        df_cal = pd.DataFrame(db_table)
        lin_cal = df_cal.loc[df_cal.run == str(gb_run[0])]['lincal'].values[0]
        cal_data = raw_data[et] * lin_cal


        # compute expected peak locations and widths (fit to Gaussians)
        fit_results = {}
        for ie, epk in enumerate(epeaks):

            # adjust the window.  resolution goes as roughly sqrt(energy)
            window = np.sqrt(epk) * 0.8
            xlo, xhi = epk - window/2, epk + window/2
            nbins = int(window) * 5
            xpb = (xhi-xlo)/nbins
            ibin_bkg = int(nbins * 0.2)

            # get histogram, error, normalize by runtime
            pk_data = cal_data[(cal_data >= xlo) & (cal_data <= xhi)]
            hist, bins, _ = pgh.get_hist(pk_data, range=(xlo, xhi), dx=xpb)
            hist_norm = np.divide(hist, runtime_min * 60)
            hist_var = np.array([np.sqrt(h / (runtime_min * 60)) for h in hist])

            # compute expected peak location and width (simple Gaussian)
            bkg0 = np.mean(hist_norm[:ibin_bkg])
            b, h = bins[1:], hist_norm - bkg0
            imax = np.argmax(h)
            upr_half = b[np.where((b > b[imax]) & (h <= np.amax(h)/2))][0]
            bot_half = b[np.where((b < b[imax]) & (h <= np.amax(h)/2))][-1]
            fwhm = upr_half - bot_half
            sig0 = fwhm / 2.355
            
            
#             # fit to simple gaussian
#             amp0 = np.amax(h) * fwhm
#             p_init = [amp0, bins[imax], sig0, bkg0] # a, mu, sigma, bkg
#             p_fit, p_cov = pgf.fit_hist(pgf.gauss_bkg, hist_norm, bins,
#                                         var=hist_var, guess=p_init)
#             fit_func = pgf.gauss_bkg
            
#             p_err = np.sqrt(np.diag(p_cov))
            
#             # goodness of fit
#             chisq = []
#             for i, h in enumerate(hist_norm):
#                 model = fit_func(b[i], *p_fit)
#                 diff = (model - h)**2 / model
#                 chisq.append(abs(diff))
#             rchisq = sum(np.array(chisq) / len(hist_norm))
#             # fwhm_err = p_err[1] * 2.355 * e_peak / e_fit

#             # collect interesting results for this row
#             fit_results[ie] = {
#                 'epk':epk,
#                 'mu':p_fit[1], 'fwhm':p_fit[2]*2.355, 'sig':p_fit[2],
#                 'amp':p_fit[0], 'bkg':p_fit[3], 'rchisq':rchisq,
#                 'mu_raw':p_fit[1] / lin_cal, # <-- this is in terms of raw E
#                 'mu_unc':p_err[1] / lin_cal
#                 }
#             print(fit_results[ie])

        
            # fit to radford peak: mu, sigma, hstep, htail, tau, bg0, amp
            amp0 = np.amax(h) * fwhm
            hstep = 0.001 # fraction that the step contributes
            htail = 0.1
            tau = 10
            p_init = [bins[imax], sig0, hstep, htail, tau, bkg0, amp0]
            p_fit, p_cov = pgf.fit_hist(pgf.radford_peak, hist_norm, bins, var=hist_var, guess=p_init)
            fit_func = pgf.radford_peak
            
            #just for debugging
            print('Len Fit params:', len(p_fit))
            
            p_err = np.sqrt(np.diag(p_cov))
            
            # goodness of fit
            chisq = []
            for i, h in enumerate(hist_norm):
                model = fit_func(b[i], *p_fit)
                diff = (model - h)**2 / model
                chisq.append(abs(diff))
            rchisq = sum(np.array(chisq) / len(hist_norm))
            # fwhm_err = p_err[1] * 2.355 * e_peak / e_fit

            # collect interesting results for this row
            fit_results[ie] = {
                'epk':epk,
                'mu':p_fit[1], 'fwhm':p_fit[2]*2.355, 'sig':p_fit[2],
                'amp':p_fit[0], 'bkg':p_fit[3], 'rchisq':rchisq,
                'mu_raw':p_fit[1] / lin_cal, # <-- this is in terms of raw E
                'mu_unc':p_err[1] / lin_cal
                }
            
#             print('Len Fit params:', len(p_fit))
            print('Fit results: ', fit_results[ie])
            
            

            # diagnostic plot, don't delete
            if config['show_plot']:
                plt.axvline(bins[ibin_bkg], c='m', label='bkg region')
                xfit = np.arange(xlo, xhi, xpb * 0.1)
                plt.plot(xfit, fit_func(xfit, *p_init), '-', c='orange',
                         label='init')
                plt.plot(xfit, fit_func(xfit, *p_fit), '-', c='red',
                         label='fit')
                plt.plot(bins[1:], hist_norm, c='b', lw=1.5, ds='steps')
                plt.xlabel('pass-1 energy (kev)', ha='right', x=1)
                plt.legend(fontsize=12)
                if config['batch_mode']:
                    plt.savefig('./plots/fit%d_peakfit.png' %ie)
                else:
                    plt.show()
                plt.close()
                
        exit()



        # ----------------------------------------------------------------------
        # compute energy calibration by matrix inversion (thanks Tim and Jason!)

        view_cols = ['epk', 'mu', 'fwhm', 'bkg', 'rchisq', 'mu_raw']
        df_fits = pd.DataFrame(fit_results).T
        print(df_fits[view_cols])

        true_peaks = df_fits['epk']
        raw_peaks, raw_error = df_fits['mu_raw'], df_fits['mu_unc']

        error = raw_error / raw_peaks * true_peaks
        cov = np.diag(error**2)
        weights = np.diag(1 / error**2)

        degree = config['pol_order']
        raw_peaks_matrix = np.zeros((len(raw_peaks), degree+1))
        for i, pk in enumerate(raw_peaks):
            temp_degree = degree
            row = np.array([])
            while temp_degree >= 0:
                row = np.append(row, pk**temp_degree)
                temp_degree -= 1
            raw_peaks_matrix[i] += row
        print(raw_peaks_matrix)

        # perform matrix inversion
        xTWX = np.dot(np.dot(raw_peaks_matrix.T, weights), raw_peaks_matrix)
        xTWY = np.dot(np.dot(raw_peaks_matrix.T, weights), true_peaks)
        if np.linalg.det(xTWX) == 0:
            print("singular matrix, determinant is 0, can't get cal constants")
            exit()
        xTWX_inv = np.linalg.inv(xTWX)

        # get polynomial coefficients and error
        cal_pars = np.dot(xTWX_inv, xTWY)
        cal_errs = np.sqrt(np.diag(xTWX_inv))
        n = len(cal_pars)
        print(f'Fit:', ' '.join([f'p{i}:{cal_pars[i]:.4e}' for i in range(n)]))
        print(f'Unc:', ' '.join([f'p{i}:{cal_errs[i]:.4e}' for i in range(n)]))

        # ----------------------------------------------------------------------
        # repeat the peak fit with the calibrated energy (affects widths)

        # compute calibrated energy
        pol = np.poly1d(cal_pars) # handy numpy polynomial object
        cal_data = pol(raw_data[et])

        fit_results = {}
#         print('fit_results', fit_results)
        print('cal_data', cal_data)
        for ie, epk in enumerate(epeaks):
            print('epk:', epk, '\n epeaks:', epeaks)

            # adjust the window.  resolution goes as roughly sqrt(energy)
            window = np.sqrt(epk) * 0.5
            xlo, xhi = epk - window/2, epk + window/2
            nbins = int(window) * 5
            xpb = (xhi-xlo)/nbins
            ibin_bkg = int(nbins * 0.2)
            print('xhi: ', xhi, 'xlo:', xlo)

            # get histogram, error, normalize by runtime
            pk_data = cal_data[(cal_data >= xlo) & (cal_data <= xhi)]
            hist, bins, _ = pgh.get_hist(pk_data, range=(xlo, xhi), dx=xpb)
            hist_norm = np.divide(hist, runtime_min * 60)
            hist_var = np.array([np.sqrt(h / (runtime_min * 60)) for h in hist])
            
            print('cal_data:', cal_data)
            
            print('bins:', bins)

            # compute expected peak location and width (simple Gaussian)
            bkg0 = np.mean(hist_norm[:ibin_bkg])
#             print(bkg0)
            b, h = bins[1:], hist_norm - bkg0
            imax = np.argmax(h)
            upr_half = b[np.where((b > b[imax]) & (h <= np.amax(h)/2))][0]
            bot_half = b[np.where((b < b[imax]) & (h <= np.amax(h)/2))][-1]
            fwhm = upr_half - bot_half
            sig0 = fwhm / 2.355
            amp0 = np.amax(h) * fwhm
            p_init = [amp0, bins[imax], sig0, bkg0] # a, mu, sigma, bkg
            p_fit, p_cov = pgf.fit_hist(pgf.gauss_bkg, hist_norm, bins,
                                        var=hist_var, guess=p_init)
            p_err = np.sqrt(np.diag(p_cov))
            
            print('p_err: ', p_err)

            # save results
            fit_results[ie] = {
                'epk':epk,
                'mu':p_fit[1], 'fwhm':p_fit[2] * 2.355, 'sig':p_fit[2], 'amp':p_fit[0], 'bkg':p_fit[3],
                }
            print('fit results:', fit_results[ie])

        # consolidate results again
        view_cols = ['epk', 'mu', 'fwhm', 'residual']
        df_fits = pd.DataFrame(fit_results).T

        # compute the difference between lit and measured values
        cal_peaks = pol(raw_peaks)
        df_fits['residual'] = true_peaks - cal_peaks
        print(df_fits[view_cols])

        # fit fwhm vs. energy
        # FWHM(E) = sqrt(A_noise^2 + A_fano^2 * E + A_qcol^2 E^2)
        # Ref: Eq. 3 of https://arxiv.org/abs/1902.02299
        # TODO: fix error handling
        def sqrt_fwhm(x, a_n, a_f, a_c):
            return np.sqrt(a_n**2 + a_f**2 * x + a_c**2 * x**2)
        p_guess = [0.3, 0.05, 0.001]
        p_fit, p_cov = curve_fit(sqrt_fwhm, df_fits['mu'], df_fits['fwhm'],
                                 p0=p_guess)#, sigma = np.sqrt(h), absolute_sigma=True)
        p_err = np.sqrt(np.diag(p_cov))

        if config['show_plot']:

            # show a split figure with calibrated spectrum + used peaks on top,
            # and calib.function and resolution vs. energy on bottom
            fig, (p0, p1) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
                                         # gridspec_kw={'height_ratios':[2, 1]}))

            # get histogram (cts / keV / d)
            xlo, xhi, xpb = config['cal_range']
            hist, bins, _ = pgh.get_hist(cal_data, range=(xlo, xhi), dx=xpb)
            hist_norm = np.divide(hist, runtime_min * 60 * xpb)

            # show peaks
            cmap = plt.cm.get_cmap('brg', len(df_fits)+1)
            for i, row in df_fits.iterrows():

                # get a pretty label for the isotope
                lbl = config['pks'][str(row['epk'])]
                iso = ''.join(r for r in re.findall('[0-9]+', lbl))
                ele = ''.join(r for r in re.findall('[a-z]', lbl, re.I))
                pk_lbl = r'$^{%s}$%s' % (iso, ele)

                pk_diff = row['epk'] - row['mu']
                p0.axvline(row['epk'], ls='--', c=cmap(i), lw=1,
                            label=f"{pk_lbl} : {row['epk']} + {pk_diff:.3f}")

            p0.semilogy(bins[1:], hist_norm, ds='steps', c='b', lw=1)

            p0.set_ylabel('cts / s / keV', ha='right', y=1)
            p0.legend(loc=3, fontsize=11)

            # TODO: add fwhm errorbar
            x_fit = np.arange(xlo, xhi, xpb)
            y_init = sqrt_fwhm(x_fit, *p_guess)
            p1.plot(x_fit, y_init, '-', lw=1, c='orange', label='guess')

            y_fit = sqrt_fwhm(x_fit, *p_fit)
            a_n, a_f, a_c = p_fit
            fit_label = r'$\sqrt{(%.2f)^2 + (%.3f)^2 E + (%.4f)^2  E^2}$' % (a_n, a_f, a_c)
            p1.plot(x_fit, y_fit, '-r', lw=1, label=f'fit: {fit_label}')

            p1.plot(df_fits['mu'], df_fits['fwhm'], '.b')

            p1.set_xlabel('Energy (keV)', ha='right', x=1)
            p1.set_ylabel('FWHM (keV)', ha='right', y=1)
            p1.legend(fontsize=11)

            if config['batch_mode']:
                plt.savefig('./plots/peakfit.png')
            else:
                plt.show()

        # the order of the polynomial should be in the table name
        pf_results[f'{et}_Anoise'] = p_fit[0]
        pf_results[f'{et}_Afano'] = p_fit[1]
        pf_results[f'{et}_Aqcol'] = p_fit[2]
        for i in range(len(cal_pars)):
            pf_results[f'{et}_cal{i}'] = cal_pars[i]
        for i in range(len(cal_pars)):
            pf_results[f'{et}_unc{i}'] = cal_errs[i]

    return pd.Series(pf_results)


if __name__=='__main__':
    main()
