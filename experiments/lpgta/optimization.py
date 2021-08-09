# import python modules
from setup import *

# the output files
pdf_hst = pdf.PdfPages(fig_dir + 'p_opt_opt_hst.pdf')
pdf_ftr = pdf.PdfPages(fig_dir + 'p_opt_opt_ftr.pdf')
pdf_hgt = pdf.PdfPages(fig_dir + 'p_opt_opt_hgt.pdf')
pdf_plt = pdf.PdfPages(fig_dir + 'p_opt_opt_plt.pdf')

# process the events for the specified channel
def process_events(args):
    # interpret the arguments
    channel, = args
    
    # the results
    metrics = []
    factors = []
    heights = []

    # loop over the grid parameters
    for i in range(len(trap)):
        # read in the tables
        tftp_tp, _ = store.read_object(f'{channel}/{i}/tftp_tp', files_opt)
        
        # find the calibration peak
        peak, _ = find_peak(tftp_tp.nda, np.arange(0, np.power(2, 16), 1))

        # calibrate the reconstructed energy
        eftp = tftp_tp.nda * (db['lines']['tlfep'] / peak)
        
        # the results
        metric = ''
        factor = 0
        height = 0

        # loop over the correction variables
        for variable in variables:
            # read in the tables
            ct_corr, _ = store.read_object(f'{channel}/{i}/{variable}', files_opt)
        
            # extract the parameters for processing
            corr = ct_corr.nda / 1000
            corr = corr - np.average(corr)

            # run the optimization
            scales, values = figure_of_merit(eftp, corr, find_bins(eftp))

            # find the optimal result
            f = scales[np.argmax(values)]

            # the corrected energy
            ecor = eftp - corr * f

            # find the peak height
            _, h = find_peak(ecor, np.arange(
                db['lines']['tlfep'] - 100,
                db['lines']['tlfep'] + 100, 0.4))
            
            # check if the global optimum
            if h > height:
                # update the results
                metric = variable
                factor = f
                height = h

        # include the result
        metrics.append(metric            )
        factors.append(np.float64(factor))
        heights.append(np.float64(height))
    
    # status printout
    logging.info(f'Finished processing {channel}')

    # return the results
    return metrics, factors, heights

# the correction variables
variables = [
    'drift_t']

# the argument list
args = [[channel] for channel in channels]

# launch the parallel processes
with mp.Pool(proc_count) as p:
    results = p.map(process_events, args)
    
# loop over the channels
for i, channel in enumerate(channels):
    # extract the results
    metrics = results[i][0]
    factors = results[i][1]
    heights = results[i][2]
    
    # find the optimal result
    i_opt = np.argmax(heights)
    
    # read in the tables
    tftp_tp, _ = store.read_object(f'{channel}/{i_opt}/tftp_tp'         , files_opt)
    ct_corr, _ = store.read_object(f'{channel}/{i_opt}/{metrics[i_opt]}', files_opt)
    
    # find the calibration peak
    peak, _ = find_peak(tftp_tp.nda, np.arange(0, np.power(2, 16), 2))

    # extract the parameters for processing
    eftp = tftp_tp.nda * (db['lines']['tlfep'] / peak)
    corr = ct_corr.nda / 1000
    corr = corr - np.average(corr)
    ecor = eftp - corr * factors[i_opt]
    
    # find the resolution
    opt_bin, opt_res, opt_err, opt_plt = find_fwhm(ecor, [
        db['lines']['tlfep'] - 100, 
        db['lines']['tlfep'] + 100], n_slope=0)
    
    # include the information
    db['channels'][channel]['calibration' ]['dsp_opt'] = peak
    db['channels'][channel]['optimization']['opt_par'] = trap   [i_opt]
    db['channels'][channel]['optimization']['opt_var'] = metrics[i_opt]
    db['channels'][channel]['optimization']['opt_ctc'] = factors[i_opt]
    db['channels'][channel]['optimization']['opt_res'] = opt_res
    db['channels'][channel]['optimization']['opt_err'] = opt_err
    
    # plot the distribution
    fig, ax = plt.subplots()
    ax.hist(eftp, opt_bin, histtype='stepfilled', alpha=0.5, log=True, label=channel)
    ax.hist(ecor, opt_bin, histtype='stepfilled', alpha=0.5, log=True, label=channel+', corrected')
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Number of events')
    ax.set_xlim(
        db['lines']['tlfep'] - 20, 
        db['lines']['tlfep'] + 20)
    ax.legend()
    if not np.all(np.isnan(opt_plt)):
        ax.axhline(opt_plt[3]    , ls='--', c='k', lw=1)
        ax.axhline(opt_plt[3] / 2, ls='--', c='k', lw=1)
        ax.axvline(opt_plt[2]    , ls='--', c='k', lw=1)
        ax.axvspan(opt_plt[0], opt_plt[1], color='k', alpha=0.1)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_hst.savefig()
    plt.close()
    
    # the data for plotting
    w_ctc = []
    w_fom = []
    count = 0
    for r in rise:
        for f in flat:
            r_ct = int(r / db['tick_width'])
            f_ct = int(f / db['tick_width'])
            p = r_ct + f_ct - int(0.5 / db['tick_width'])
            if index_hi + p < db['tick_count']:
                w_ctc.append(factors[count])
                w_fom.append(heights[count])
                count += 1
            else:
                w_ctc.append(np.nan)
                w_fom.append(np.nan)
                
    # the axis labels
    xlabel = 'Rise time [$\mu$s]'
    ylabel = 'Flat time [$\mu$s]'

    # plot the optimization grid
    plot_matrix(rise, flat, w_ctc, pdf_ftr, xlabel, ylabel, 'Correction factor', text=channel)
    plot_matrix(rise, flat, w_fom, pdf_hgt, xlabel, ylabel, 'Figure of merit'  , text=channel)

    # status printout
    if not np.all(np.isnan(opt_plt)):
        logging.info(f'Found resolution for {channel} of {opt_res:.1f} keV')
    else:
        logging.info(f'Found no resolution for {channel}')
        
# plot the result
fig, ax = plt.subplots()
opt_res = [db['channels'][c]['optimization']['opt_res'] for c in channels]
opt_err = [db['channels'][c]['optimization']['opt_err'] for c in channels]
x    = [int(c[1:]) for c, r, e in zip(channels, opt_res, opt_err) if r > 0]
y    = [r          for c, r, e in zip(channels, opt_res, opt_err) if r > 0]
yerr = [e          for c, r, e in zip(channels, opt_res, opt_err) if r > 0]
ax.errorbar(x, y, yerr=yerr, marker='o', ms=5, ls='--', lw=1)
ax.set_xlabel('Channel')
ax.set_ylabel('FWHM at 2.6 MeV [keV]')
ax.set_ylim(0, ax.get_ylim()[1])
ax.get_yaxis().set_label_coords(-0.1, 0.5)
pdf_plt.savefig()
plt.close()
    
# close the output files
pdf_hst.close()
pdf_ftr.close()
pdf_hgt.close()
pdf_plt.close()

# update the database
update_database(db)
