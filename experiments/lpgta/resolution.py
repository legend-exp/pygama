# import python modules
from setup import *

# the qbb energy
qbb = 2039.06

# the lines
line_vals, line_keys = zip(*sorted(zip(
    list(db['lines'].values()), 
    list(db['lines'].keys  ()))))

# the energy windows
windows = [[l - 10, l + 10] for l in line_vals]

# process the events for the specified file and channel
def process_events(args):
    # interpret the arguments
    i_fi, i_ch = args

    # the mask
    all_dqc = find_mask('all_dqc', i_fi, channels[i_ch])
    
    # the events to process
    idx = np.arange(len(all_dqc))[all_dqc]
    
    # the correction variable
    var = db['channels'][channels[i_ch]]['optimization']['opt_var']
    
    # read in the tables
    tftp_tp, _ = store.read_object(f'{channels[i_ch]}/tftp_tp', files_dsp[i_fi], idx=idx)
    ct_corr, _ = store.read_object(f'{channels[i_ch]}/{var}'  , files_dsp[i_fi], idx=idx)
    
    # the calibration factor
    adu_to_kev = db['lines']['tlfep'] / db['channels'][channels[i_ch]]['calibration']['dsp_opt']

    # loop over the energy windows
    for i, window in enumerate(windows):
        # select events at the calibration peak
        idx = np.where(
            (window[0] < tftp_tp.nda * adu_to_kev) & 
            (window[1] > tftp_tp.nda * adu_to_kev))[0]
        
        # the output arrays
        ar_eftp = lh5.Array(np.array([], dtype='float32'))
        ar_corr = lh5.Array(np.array([], dtype='float32'))
        
        # check if there are events to process
        if len(idx) > 0:
            # extract the parameters
            eftp = tftp_tp.nda[idx] * adu_to_kev
            corr = ct_corr.nda[idx] / 1000
            corr = corr - np.average(corr)
            
            # update the output arrays
            ar_eftp = lh5.Array(eftp)
            ar_corr = lh5.Array(corr)
        
        # build the output table
        tb_ana = lh5.Table(col_dict={
            'eftp': ar_eftp,
            'corr': ar_corr})
        
        # write the output to disk
        lock .acquire()
        store.write_object(tb_ana, f'{channels[i_ch]}/{line_keys[i]}', files_ana[i_fi])
        lock .release()
        
    # status printout
    logging.info(f'Finished processing {channels[i_ch]} in {os.path.basename(files_raw[i_fi])}')

# the stage that produces the data
def produce():
    # loop over the files
    for i_fi in range(len(files_raw)):
        # skip if not the selected file
        if proc_index and i_fi not in proc_index:
            continue
        
        # delete the output file if it exists
        if os.path.isfile(files_ana[i_fi]):
            os.remove(files_ana[i_fi])

        # the argument list
        args = [[i_fi, i_ch] for i_ch in range(len(channels))]
        
        # launch the parallel processes
        with mp.Pool(proc_count) as p:
            p.map(process_events, args)

# analyze the events for the specified channel
def analyze_events(args):
    # interpret the arguments
    channel, = args
    
    # the results
    e_peaks = []
    factors = []
    heights = []
    fwhm_vs = []
    fwhm_es = []
    fwhm_ps = []

    # loop over the energy windows
    for i, window in enumerate(windows):
        # read in the tables
        eftp, _ = store.read_object(f'{channel}/{line_keys[i]}/eftp', files_ana)
        corr, _ = store.read_object(f'{channel}/{line_keys[i]}/corr', files_ana)
                
        # select the events
        idx = np.where(
            np.isfinite(eftp.nda) &
            np.isfinite(corr.nda))[0]
        
        # extract the events
        eftp = eftp.nda[idx]
        corr = corr.nda[idx]
        
        # skip if there are no events to process
        if len(eftp) == 0:
            continue
                    
        # run the optimization
        scales, values = figure_of_merit(eftp, corr, find_bins(eftp, min_bin_width=0.01))
        
        # skip if the optimization failed
        if len(values) == 0:
            continue
        
        # find the optimal result
        factor = scales[np.argmax(values)]
        
        # the corrected energy
        ecor = eftp - corr * factor

        # find the peak height
        _, height = find_peak(ecor, np.arange(*window, 0.4))
        
        # find the resolution
        opt_bin, opt_res, opt_err, opt_plt = find_fwhm(ecor, window, n_slope=0)
        
        # histogram the data for plotting
        n = numba_histogram(ecor, opt_bin)
        
        # append the results
        e_peaks.append(line_vals[i]         )
        factors.append(np.float64(factor )  )
        heights.append(np.float64(height )  )
        fwhm_vs.append(np.float64(opt_res)  )
        fwhm_es.append(np.float64(opt_err)  )
        fwhm_ps.append([n, opt_bin, opt_plt])
                
    # status printout
    logging.info(f'Finished processing {channel}')

    # return the results
    return e_peaks, factors, heights, fwhm_vs, fwhm_es, fwhm_ps

# the stage that analyzes the data
def analyze():
    # the output files
    pdf_eny = pdf.PdfPages(fig_dir + 'p_ana_res_eny.pdf')
    pdf_cor = pdf.PdfPages(fig_dir + 'p_ana_res_cor.pdf')
    pdf_res = pdf.PdfPages(fig_dir + 'p_ana_res_res.pdf')
    pdf_qbb = pdf.PdfPages(fig_dir + 'p_ana_res_qbb.pdf')

    # the argument list
    args = [[channel] for channel in channels]
        
    # launch the parallel processes
    with mp.Pool(proc_count) as p:
        results = p.map(analyze_events, args)
        
    # loop over the channels
    for i, channel in enumerate(channels):
        # extract the results
        e_peaks = results[i][0]
        factors = results[i][1]
        heights = results[i][2]
        fwhm_vs = results[i][3]
        fwhm_es = results[i][4]
        fwhm_ps = results[i][5]

        # loop over the energy windows
        for n, b, p in fwhm_ps:
            # plot the distribution
            fig, ax = plt.subplots()
            ax.hist(b[:-1], b, weights=n, histtype='stepfilled', alpha=0.5, log=True, label=channel)
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel('Number of events')
            ax.set_xlim(b[0], b[-1])
            ax.legend()
            if not np.all(np.isnan(p)):
                ax.axhline(p[3]    , ls='--', c='k', lw=1)
                ax.axhline(p[3] / 2, ls='--', c='k', lw=1)
                ax.axvline(p[2]    , ls='--', c='k', lw=1)
                ax.axvspan(p[0], p[1], color='k', alpha=0.1)
            ax.get_yaxis().set_label_coords(-0.1, 0.5)
            pdf_eny.savefig()
            plt.close()

        # check the number processed
        if len(e_peaks) > 1:
            # plot the result
            fig, ax = plt.subplots()
            ax.errorbar(e_peaks, factors, marker='.', ms=10, ls='-', lw=1, label=channel)
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel('Correction factor')
            ax.legend()
            ax.get_yaxis().set_label_coords(-0.1, 0.5)
            pdf_cor.savefig()
            plt.close()

        # the resolution at qbb
        popt = np.zeros(2)
        qbb_res = 0
        qbb_err = 0

        # the data to be fit
        x = np.array(e_peaks)[np.array(fwhm_vs) > 0]
        y = np.array(fwhm_vs)[np.array(fwhm_vs) > 0]
        z = np.array(fwhm_es)[np.array(fwhm_vs) > 0]

        # check that there are enough data to fit
        if len(x) >= 2:
            try:
                # fit the data
                popt, pcov = optimize.curve_fit(sqrt_func, x, y, sigma=z, p0=[0.03, 200])

                # interpolate the resolution at qbb
                qbb_res = sqrt_func(qbb, *popt)
                qbb_err = np.sqrt(
                    pcov[0][0] * (qbb + popt[1]) +\
                    pcov[1][1] * np.power(popt[0], 2) / (4 * (qbb + popt[1])))
            except:
                # continue processing
                pass

            # plot the result
            fig, ax = plt.subplots()
            ax.errorbar(x, y, yerr=z, marker='.', ms=10, ls='None', lw=1, label=channel)
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel('FHWM [keV]')
            ax.legend()
            if np.all(popt != 0):
                xf = np.linspace(e_peaks[0], e_peaks[-1], 1000)
                ax.plot(xf, sqrt_func(xf, *popt), c='k', ls='-', lw=1)
                ax.axhline(qbb_res, ls='--', color='k', lw=1)
                ax.axvline(qbb    , ls='--', color='k', lw=1)
            ax.get_yaxis().set_label_coords(-0.1, 0.5)
            pdf_res.savefig()
            plt.close()

        # include the information
        db['channels'][channel]['resolution']['line_idx'] = e_peaks
        db['channels'][channel]['resolution']['line_ctc'] = factors
        db['channels'][channel]['resolution']['line_fom'] = heights
        db['channels'][channel]['resolution']['line_res'] = fwhm_vs
        db['channels'][channel]['resolution']['line_err'] = fwhm_es
        db['channels'][channel]['resolution']['fit_par' ] = popt.tolist()
        db['channels'][channel]['resolution']['qbb_res' ] = qbb_res
        db['channels'][channel]['resolution']['qbb_err' ] = qbb_err

        # status printout
        if qbb_err > 0:
            logging.info(f'Found resolution in the region of interest for {channel} of {qbb_res:.1f} keV')
        else:
            logging.info(f'Found no resolution in the region of interest for {channel}')

    # plot the result
    fig, ax = plt.subplots()
    xs, ys, zs = [], [], []
    for channel in channels:
        x = int(channel[1:])
        y = db['channels'][channel]['resolution']['qbb_res']
        z = db['channels'][channel]['resolution']['qbb_err']
        if z > 0:
            xs.append(x)
            ys.append(y)
            zs.append(z)
    ax.errorbar(xs, ys, yerr=zs, marker='.', ms=10, ls='--', lw=1, label=channel)
    ax.set_xlabel('Channel index')
    ax.set_ylabel(r'FHWM at $Q_{\beta \beta}$ [keV]')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_qbb.savefig()
    plt.close()

    # close the output files
    pdf_eny.close()
    pdf_cor.close()
    pdf_res.close()
    pdf_qbb.close()

    # update the database
    update_database(db)

# process based on the configuration
if proc_stage in ['', 'produce']: produce()
if proc_stage in ['', 'analyze']: analyze()
