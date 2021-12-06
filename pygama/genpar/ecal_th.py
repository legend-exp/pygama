import pygama.lh5 as lh5
import matplotlib.pyplot as plt
import numpy as np
import os,json
import pathlib
from scipy.optimize import curve_fit
import pygama.genpar_tmp.cuts as cut
import pygama.analysis.histograms as pgh
import pygama.analysis.calibration as cal
import pygama.analysis.peak_fitting as pgp

def fwhm_slope(x, m0, m1):
    """
    Fit the energy resolution curve
    """
    return np.sqrt(m0 + m1*x)


def energy_cal_th(files, energy_params, save_path, cut_parameters= {'bl_mean':4,'bl_std':4, 'pz_std':4}):
    
    if isinstance(energy_params, str): energy_params = [energy_params]

    ####################
    # Start the analysis
    ####################
    print('Load and apply quality cuts...',end=' ')
    uncal_pass, uncal_cut = cut.load_nda_with_cuts(files,'raw',energy_params,  cut_parameters= cut_parameters, verbose=False)
    print("Done")

    Npass = len(uncal_pass[energy_params[0]])
    Ncut  = len(uncal_cut[energy_params[0]])
    Ratio = 100.*float(Ncut)/float(Npass+Ncut)
    print(f'{Npass} events pass')
    print(f'{Ncut} events cut')
    
    glines    = [583.191, 727.330, 860.564,1592.53,1620.50,2103.53,2614.50] # gamma lines used for calibration
    range_keV = [(25,40),(25,40), (25,40),(25,20),(25,40),(25,40),(70,70)] # side bands width
    funcs = [pgp.radford_peak,pgp.radford_peak,pgp.radford_peak,
             pgp.radford_peak,pgp.radford_peak,pgp.radford_peak,pgp.radford_peak]
    output_dict = {}
    for energy_param in energy_params:
        guess     = (2620/np.nanpercentile(uncal_pass[energy_param],99))
        print(f'Find peaks and compute calibration curve for {energy_param}', end = ' ')
        pars, cov, results = cal.hpge_E_calibration(uncal_pass[energy_param],
                                                    glines,
                                                    guess,
                                                    deg=1,
                                                    range_keV = range_keV,
                                                    funcs = funcs,
                                                    verbose=False
                                                    )
        mus = np.array([])
        for i in range(len(results['pk_pars'])):
            mus= np.append(mus, results['pk_pars'][i][0])
        found_peak = results['got_peaks_locs'][-1]
        if (np.abs(mus-found_peak)>20).all() or results['pk_fwhms'][:,1][-1]>0.05: #check error on 2.6 keV peak
            print('2.6 failed rerunning with larger width')
            range_keV = [(25,40),(25,40), (25,40),(25,20),(25,40),(25,40),(85,85)]
            pars, cov, results = cal.hpge_E_calibration(uncal_pass[energy_param],
                                                    glines,
                                                    guess,
                                                    deg=1,
                                                    range_keV = range_keV,
                                                    funcs = funcs,
                                                    verbose=False
                                                    )
        print("done")
        print(" ")
        ecal_pass = pgp.poly(uncal_pass[energy_param], pars)
        xpb = 1
        xlo = 0
        xhi = 4000
        nb = int((xhi-xlo)/xpb)
        hist_pass, bin_edges = np.histogram(ecal_pass, range=(xlo, xhi), bins=nb)
        bins_pass = pgh.get_bin_centers(bin_edges)
        fitted_peaks = results['fitted_keV']
        pk_pars      = results['pk_pars']
        pk_covs      = results['pk_covs']
        datatype, detector, measurement, run, timestamp = os.path.basename(files[0]).split('-')
        plot_title = f'{detector}-{measurement}-{run}'
        peaks_kev = results['got_peaks_keV']
        fitted_peaks = results['fitted_keV']
        mus = np.zeros(len(fitted_peaks))
        for i,ppars in enumerate(zip(funcs,pk_pars)):
            if ppars[0]==pgp.radford_peak:
                mus[i] = np.array([ppars[1][0]]).astype(float)
            elif ppars[0] ==pgp.gauss_step:
                mus[i] = np.array([ppars[1][1]]).astype(float)

        fwhms        = results['pk_fwhms'][:,0]
        dfwhms       = results['pk_fwhms'][:,1]


        plt.rcParams['figure.figsize'] = (12, 20)
        plt.rcParams['font.size'] = 12

        plt.figure()
        range_adu = 10/pars[0] #10keV window around peak in adu
        for i, peak in enumerate(mus):
            plt.subplot(4,2,i+1)
            binning = np.arange(peak-range_adu, peak+range_adu, 1)
            bin_cs = (binning[1:]+binning[:-1])/2
            fit_vals = pgp.radford_peak(bin_cs, *pk_pars[i])
            counts, bs, bars = plt.hist(uncal_pass[energy_param], bins=binning, histtype='step')
            plt.plot(bin_cs, fit_vals)
            plt.plot(bin_cs, (fit_vals-counts)) 
            locs,labels = plt.xticks()
            new_labels = get_peak_labels(locs, pars)
            plt.xticks(ticks = locs[1:-1], labels = new_labels)
    
            plt.plot([bin_cs[10]],[0],label=get_peak_label(fitted_peaks[i]), linestyle='None' )
            plt.plot([bin_cs[10]],[0],label = f'{fitted_peaks[i]:.1f} keV', linestyle='None')
            plt.plot([bin_cs[10]],[0],label = f'{fwhms[i]:.2f} +- {dfwhms[i]:.2f} keV', linestyle='None')
    
            plt.xlabel('Energy (keV)')
            plt.ylabel('Counts')
            plt.legend(loc = 'upper left', frameon=False)
    

        plt.tight_layout()
        fits_save_path = os.path.join(save_path, 'plots', detector, f'{energy_param}_fits.pdf')
        pathlib.Path(os.path.dirname(fits_save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(fits_save_path)
        plt.close()

        #####
        # Remove the Tl SEP and DEP from calibration if found
        fwhm_peaks   = np.array([], dtype=np.float32)
        indexes=[]
        for i,peak in enumerate(fitted_peaks):
            if(peak==2103.53): 
                print(f"Tl SEP found at index {i}")
                indexes.append(i)
                continue
            if(peak==1592.53): 
                print(f"Tl DEP found at index {i}")
                indexes.append(i)
                continue
            fwhm_peaks = np.append(fwhm_peaks,peak)
        fwhms  = np.delete(fwhms,[indexes])
        dfwhms = np.delete(dfwhms,[indexes])
        mus = np.delete(mus,[indexes])
        #####
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 8
        for i,peak in enumerate(fwhm_peaks):
            print(f'FWHM of {peak} keV peak is: {fwhms[i]:1.2f} +- {dfwhms[i]:1.2f} keV')
        param_guess  = [0.2,0.001]
        param_bounds = (0, [10., 1.])
        fit_pars, fit_covs = curve_fit(fwhm_slope, fwhm_peaks, fwhms, sigma=dfwhms, 
                               p0=param_guess, bounds=param_bounds, absolute_sigma=True)
        sderrs = np.sqrt(np.diag(fit_covs))
        qbb_err = fwhm_slope(2039.0,*(fit_pars+sderrs))-fwhm_slope(2039.0,*fit_pars)
        print(f'FWHM curve fit: {fit_pars}')
        fit_vals = fwhm_slope(fwhm_peaks,*fit_pars)
        print(f'FWHM fit values: {fit_vals}')
        fit_qbb = fwhm_slope(2039.0,*fit_pars)
        print(f'FWHM energy resolution at Qbb: {fit_qbb:1.2f} +- {qbb_err:1.2f} keV' )
        qbb_line_hx = [2039.0,2039.0]
        qbb_line_hy = [np.amin(fwhms),fit_qbb]
        qbb_line_vx = [np.amin(fwhm_peaks),2039.0]
        qbb_line_vy = [fit_qbb,fit_qbb]
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        ax1.errorbar(fwhm_peaks,fwhms,yerr=dfwhms, marker='o',lw=0, c='b')
        ax1.plot(fwhm_peaks,fit_vals,ls=' ')
        fwhm_slope_bins = np.arange(np.amin(fwhm_peaks),np.amax(fwhm_peaks),10)
        ax1.plot(fwhm_slope_bins ,fwhm_slope(fwhm_slope_bins,*fit_pars),lw=1, c='g')
        ax1.plot(qbb_line_hx,qbb_line_hy,lw=1, c='r')
        ax1.plot(qbb_line_vx,qbb_line_vy,lw=1, c='r')
        ax1.set_ylim([1,3])
        ax1.set_ylabel("FWHM energy resolution (keV)", ha='right', y=1)
        ax2.plot(fwhm_peaks,pgp.poly(mus, pars)-fwhm_peaks, lw=1, c='b')
        ax2.set_xlabel("Energy (keV)",    ha='right', x=1)
        ax2.set_ylabel("Residuals (keV)", ha='right', y=1)
        fig.suptitle(plot_title)
        plot_save_path = os.path.join(save_path, 'plots', detector, f'{energy_param}.png')
        pathlib.Path(os.path.dirname(plot_save_path)).mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_save_path, bbox_inches='tight', transparent=True)
        plt.close()

        

        output_dict[energy_param] = {'Qbb_fwhm': round(fit_qbb,2), 'Qbb_fwhm_err': round(qbb_err,2),
                                    '2.6_fwhm': round(fwhms[-1],2), '2.6_fwhm_err': round(dfwhms[-1],2), 
                                    "m0":fit_pars[0], "m1":fit_pars[1], 
                                    "Calibration_pars":pars.tolist(),
                                    "Number_passed": Npass,'Number_cut': Ncut,"Cut Percentage": Ratio 

                                    }
    dict_save_path = os.path.join(save_path, f'{detector}.json')
    with open(dict_save_path,'w') as fp:
        json.dump(output_dict,fp, indent=4)


def get_peak_labels(labels, pars):
    out = np.array([])
    for i,label in enumerate(labels):
        if i == 0 or i == len(labels)-1:
            pass
        else:
            out = np.append(out, f'{pgp.poly(label, pars):.1f}')
    return out

def get_peak_label(peak):
    if peak == 583.191: 
        return 'Tl 583'
    elif peak == 727.33:
        return 'Bi 727'
    elif peak == 860.564:
        return 'Tl 860'
    elif peak == 1592.53:
        return 'Tl DEP'
    elif peak == 1620.5:
        return 'Bi FEP'
    elif peak == 2103.53:
        return 'Tl SEP'
    elif peak == 2614.5:
        return 'Tl FEP'
