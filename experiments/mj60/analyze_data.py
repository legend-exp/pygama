#!/usr/bin/env python3
import os, time, json
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

from pygama import DataSet
from pygama.analysis.calibration import *
from pygama.analysis.histograms import *
from pygama.utils import set_plot_style
set_plot_style("clint")

def main():
    """
    mj60 analysis suite
    """
    global runDB
    with open("runDB.json") as f:
        runDB = json.load(f)

    global tier_dir
    tier_dir = runDB["tier_dir"]

    # -- analysis routines --
    # quick_gat()
    # compare_spectra()
    # get_spectra()
    # get_multiple_spectra()
    # check_hist()
    calibrate()
    # resolution()


def quick_gat():
    """ process data with majorcaroot & GAT """
    import subprocess as sp

    run = 72
    raw_file = "/Users/wisecg/project/pygama/2018-9-30-BackgroundRun72"
    t_start = time.time()
    pwd = os.getcwd()
    os.chdir("/Users/wisecg/project/pygama/")
    sp.run(["majorcaroot_basic_struck",raw_file])
    sp.run(["process_mjd_data_p1","OR_run{}.root".format(run)])
    os.chdir(pwd)
    print("Elapsed: {:.2f} sec".format(time.time() - t_start))


def compare_spectra():
    from ROOT import TFile, TTree

    xlo, xhi, xpb = 0, 10000, 10

    # gat
    tf = TFile("/Users/wisecg/project/pygama/mjd_run72.root")
    tt = tf.Get("mjdTree")
    n = tt.Draw("trapE","","goff")
    trapE = tt.GetV1()
    trapE = np.asarray([trapE[i] for i in range(n)])
    xG, hG = get_hist(trapE, xlo, xhi, xpb)

    # pygama
    t2_df = pd.read_hdf("/Users/wisecg/project/mj60/t2_run72.h5")
    xP, hP = get_hist(t2_df["trap_max"], xlo, xhi, xpb)

    # plot
    plt.plot(xG, hG, ls='steps', lw=1.5, c='b',
             label="gat trapE, {} cts".format(len(trapE)))
    plt.plot(xP, hP, ls='steps', lw=1.5, c='m',
             label="pygama trap_max, {} cts".format(sum(hP)))
    plt.legend()
    plt.show()


def bias_tests():
    """
    only need to do this for the test data, not generally
    parse special test data into dataframes
    (access with runDB["df_test_[num]"])
    """

    test_list = [k for k in runDB["tests"] if "test" in k]
    for t in test_list:
        tmp = io.StringIO('\n'.join(runDB["tests"][t]))
        key = "df_{}".format(t)
        self.runDB[key] = pd.read_csv(tmp, delim_whitespace=True)


def get_spectra():

    ds = DataSet(runlist=[143, 144, 145], md='./runDB.json', tier_dir=tier_dir)
    t2df = ds.get_t2df()

    xlo, xhi, xpb = 0, 10000, 10
    xP, hP = get_hist(t2df["trap_max"], xlo, xhi, xpb)

    plt.plot(xP, hP, ls='steps', lw=1.5, c='m',
             label="pygama trap_max, {} cts".format(sum(hP)))
    plt.xlabel("Energy (uncal)", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_multiple_spectra():

    # energy (onboard)
    # xlo, xhi, xpb = 0, 2000000, 1000
    # xlo, xhi, xpb = 0, 500000, 1000
    # xlo, xhi, xpb = 0, 50000, 100

    # energy (onboard, calibrated)
    xlo, xhi, xpb = 0, 40, 0.1

    # trap_max
    # xlo, xhi, xpb = 0, 10000, 10
    # xlo, xhi, xpb = 0, 300, 0.3
    # xlo, xhi, xpb = 0, 80, 0.2
    # xlo, xhi, xpb = 0, 40, 0.1

    # ds = DataSet(run=147, md='./runDB.json', tier_dir=tier_dir)

    # get calibration
    cal = runDB["cal_onboard"]["11"]
    m, b = cal[0], cal[1]

    ds = DataSet(10, md='./runDB.json', tier_dir=tier_dir)
    rt1 = ds.get_runtime() / 3600
    t2df = ds.get_t2df()
    ene1 = m * t2df["energy"] + b

    x, h1 = get_hist(ene1, xlo, xhi, xpb)
    # x, h1 = get_hist(t2df["trap_max"], xlo, xhi, xpb)
    h1 = np.divide(h1, rt1)

    ds2 = DataSet(11, md='./runDB.json', tier_dir=tier_dir)
    t2df2 = ds2.get_t2df()
    rt2 = ds2.get_runtime() / 3600
    ene2 = m * t2df2["energy"] + b
    x, h2 = get_hist(ene2, xlo, xhi, xpb)
    # x, h2 = get_hist(t2df2["trap_max"], xlo, xhi, xpb)
    h2 = np.divide(h2, rt2)

    plt.figure(figsize=(7, 5))

    plt.plot(x, h1, ls='steps', lw=1, c='b',
             label="bkg, {:.2f} hrs".format(rt1))

    plt.plot(x, h2, ls='steps', lw=1, c='r',
             label="Kr83, {:.2f} hrs".format(rt2))

    plt.axvline(9.4057, color='m', lw=2, alpha=0.4, label="9.4057 keV") # kr83 lines
    plt.axvline(12.651, color='g', lw=2, alpha=0.4, label="12.651 keV") # kr83 lines

    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("Counts / hr / {:.2f} keV".format(xpb), ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    # plt.savefig("./plots/krSpec_{:.0f}_{:.0f}_onboard.pdf".format(xlo,xhi))
    # plt.savefig("./plots/krSpec_{:.0f}_{:.0f}_uncal.pdf".format(xlo,xhi))
    plt.savefig("./plots/krSpec_{:.0f}_{:.0f}_cal.pdf".format(xlo,xhi))


def check_hist():
    """
    double-check the pygama/LAT get_hist method is correct
    """
    xE, hE = get_hist(ene, xlo, xhi, xpb)

    nb = int((xhi-xlo)/xpb)
    hist, bins = np.histogram(ene, range=(xlo, xhi), bins=nb)
    bin_centers = bins[:-1] + 0.5 * (bins[1] - bins[0])

    plt.plot(bin_centers, hist, ls='steps', lw=1, c='b', label="ben")
    plt.plot(xE, hE, ls='steps', lw=1, c='r', label='pyg')
    plt.legend()
    plt.show()


def calibrate():
    """
    do a rough energy calibration
    "automatic": based on finding ratios
    """
    from scipy.signal import medfilt, find_peaks_cwt
    from scipy.stats import linregress

    pks_lit = [239, 911, 1460.820, 1764, 2614.511]

    # ds = DataSet(11, md='./runDB.json', tier_dir=tier_dir)
    ds = DataSet(run=204, md='./runDB.json', tier_dir=tier_dir)

    t2df = ds.get_t2df()
    rt = ds.get_runtime() / 3600 # hrs

    ene = t2df["e_ftp"]

    xlo, xhi, xpb = 0, 10000, 10 # damn, need to remove the overflow peak
    nbins = int((xhi-xlo)/xpb)

    hE, xE, _ = get_hist(ene, nbins, (xlo, xhi))

    # xE, hE = get_hist(ene, xlo, xhi, xpb)

    # -- pygama's cal routine needs some work ... --
    # need to manually remove the overflow peak?
    # data_peaks = get_most_prominent_peaks(ene, xlo, xhi, xpb, test=True)
    # ene_peaks = get_calibration_energies("uwmjlab")
    # ene_peaks = get_calibration_energies("th228")
    # best_m, best_b = match_peaks(data_peaks, ene_peaks)
    # ecal = best_m * t2df["trap_max"] + best_b

    # -- test out a rough automatic calibration here --

    npks = 15

    hE_med = medfilt(hE, 21)
    hE_filt = hE - hE_med
    pk_width = np.arange(1, 10, 0.1)
    pk_idxs = find_peaks_cwt(hE_filt, pk_width, min_snr=5)
    pks_data = xE[pk_idxs]

    pk_counts = hE[pk_idxs]
    idx_sorted = np.argsort(pk_counts)
    pk_idx_max = pk_idxs[idx_sorted[-npks:]]
    pks_data = np.sort(xE[pk_idx_max])

    r0 = pks_lit[4]/pks_lit[2]

    # this is pretty ad hoc, should use more of the match_peaks function
    found_match = False
    for pk1 in pks_data:
        for pk2 in pks_data:
            r = pk1/pk2
            if np.fabs(r - r0) < 0.005:
                print("found match to peak list:\n    "
                      "r0 {:.3f}  r {:.3f}  pk1 {:.0f}  pk2 {:.0f}"
                      .format(r0, r, pk1, pk2))
                found_match = True # be careful, there might be more than one
                break

        if found_match:
            break

    # # check uncalibrated spectrum
    # plt.plot(xE, hE, ls='steps', lw=1, c='b')
    # # plt.plot(xE, hE_filt, ls='steps', lw=1, c='b')
    # # for pk in pks_data:
    # #     plt.axvline(pk, color='r', lw=1, alpha=0.6)
    # plt.axvline(pk1, color='r', lw=1)
    # plt.axvline(pk2, color='r', lw=1)
    # plt.show()
    # exit()

    # two-point calibration
    data = np.array(sorted([pk1, pk2]))
    lit = np.array([pks_lit[2], pks_lit[4]])
    m, b, _, _, _ = linregress(data, y=lit)
    print("Paste this into runDB.json:\n    ", m, b)

    # err = np.sum((lit - (m * data + b))**2)
    # plt.plot(data, lit, '.b', label="E = {:.2e} x + {:.2e}".format(m, b))
    # xf = np.arange(data[0], data[1], 1)
    # plt.plot(xf, m * xf + b, "-r")
    # plt.legend()
    # plt.show()

    # apply calibration
    ecal = m * ene + b

    # # check calibrated spectrum
    xlo, xhi, xpb = 0, 3000, 1
    hC, xC, _ = get_hist(ecal, int((xhi-xlo)/xpb), (xlo, xhi))
    hC = np.concatenate((hC, [0])) # FIXME: annoying - have to add an extra zero

    plt.semilogy(xC, hC / rt, c='b', ls='steps', lw=1,label="MJ60 data, {:.2f} hrs".format(rt))
    plt.axvline(pks_lit[2], c='r', lw=3, alpha=0.7, label="40K, 1460.820 keV")
    plt.axvline(pks_lit[4], c='m', lw=3, alpha=0.7, label="208Tl, 2614.511 keV")
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("Counts / hr / {:.2f} keV".format(xpb), ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("./plots/surface_spec.pdf")
    # exit()

    # check low-e spectrum
    plt.figure()
    xlo, xhi, xpb = 0, 50, 0.1
    hC, xC, _ = get_hist(ecal, int((xhi-xlo)/xpb), (xlo, xhi))
    hC = np.concatenate((hC, [0])) # FIXME: annoying - have to add an extra zero
    plt.plot(xC, hC, c='b', ls='steps', lw=1, label="Kr83 data")
    plt.axvline(9.4057, color='r', lw=1.5, alpha=0.6, label="9.4057 keV") # kr83 lines
    plt.axvline(12.651, color='g', lw=1.5, alpha=0.6, label="12.651 keV") # kr83 lines
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("Counts / {:.2f} keV".format(xpb), ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("./plots/test_kr83_cal.pdf")


def resolution():
    """
    fit the 208Tl 2615 keV peak and give me the resolution
    test out pygama's peak fitting routines
    """
    ds_num = 11
    ds = DataSet(ds_num, md='./runDB.json', tier_dir=tier_dir)
    t2df = ds.get_t2df()
    ene = t2df["energy"].values
    rt = ds.get_runtime() / 3600 # hrs

    # apply calibration
    cal = runDB["cal_onboard"][str(ds_num)]
    m, b = cal[0], cal[1]
    ene = m * ene + b

    # zoom in to the area around the 2615 peak
    xlo, xhi, xpb = 2565, 2665, 0.5
    ene2 = ene[np.where((ene > xlo) & (ene < xhi))]
    xE, hE = get_hist(ene, xlo, xhi, xpb)

    # set peak bounds
    guess_ene = 2615
    guess_sig = 5
    idxpk = np.where((xE > guess_ene-guess_sig) & (xE > guess_ene+guess_sig))
    guess_area = np.sum(hE[idxpk])

    # radford_peak function pars: mu, sigma, hstep, htail, tau, bg0, a
    p0 = [guess_ene, guess_sig, 1E-3, 0.7, 5, 0, guess_area]

    bnd = [[0.9 * guess_ene,
            0.5 * guess_sig,
            0, 0, 0, 0, 0],
           [1.1 * guess_ene,
            2 * guess_sig,
            0.1, 0.75, 10, 10,
            5 * guess_area]]

    pars = fit_binned(radford_peak, hE, xE, p0)#, bounds=bnd)

    print("mu:",pars[0],"\n",
          "sig",pars[1],"\n",
          "hstep:",pars[2],"\n",
          "htail:",pars[3],"\n",
          "tau:",pars[4],"\n",
          "bg0:",pars[5],"\n",
          "a:",pars[6])

    plt.plot(xE, hE, c='b', ls='steps', lw=1,
             label="MJ60 data, {:.2f} hrs".format(rt))

    plt.plot(xE, radford_peak(xE, *pars), color="r", alpha=0.7,
             label=r"Radford peak, $\sigma$={:.2f} keV".format(pars[1]))

    plt.axvline(2614.511, color='r', alpha=0.6, lw=1,
                label=r"$E_{lit}$=2614.511")

    plt.axvline(pars[0], color='g', alpha=0.6, lw=1,
                label=r"$E_{fit}$=%.3f"%(pars[0]))

    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("Counts / {:.2f} keV".format(xpb), ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("./plots/kr83_resolution.pdf")


if __name__=="__main__":
    main()
