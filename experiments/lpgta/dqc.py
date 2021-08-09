# import python modules
from setup import *

# the output files
pdf_unc = pdf.PdfPages(fig_dir + f'p_{proc_label}_dqc_uen.pdf')
pdf_cal = pdf.PdfPages(fig_dir + f'p_{proc_label}_dqc_cen.pdf')
pdf_tps = pdf.PdfPages(fig_dir + f'p_{proc_label}_dqc_tps.pdf')

# the window of interest
window_size = 100
window = [
    db['lines']['tlfep'] - window_size,
    db['lines']['tlfep'] + window_size]

# the statistics
stats = np.zeros(5)

# loop over the channels
for channel in channels:
    # read in the tables
    sat_lo   , _ = store.read_object(f'{channel}/sat_lo'   , files_arg)
    sat_hi   , _ = store.read_object(f'{channel}/sat_hi'   , files_arg)
    pftp_lo  , _ = store.read_object(f'{channel}/pftp_lo'  , files_arg)
    pftp_hi  , _ = store.read_object(f'{channel}/pftp_hi'  , files_arg)
    tftp_tp  , _ = store.read_object(f'{channel}/tftp_tp'  , files_arg)
    tp_00    , _ = store.read_object(f'{channel}/tp_00'    , files_arg)
    tp_00_dqc, _ = store.read_object(f'{channel}/tp_00_dqc', files_arg)
    
    # find the calibration peak
    if proc_label == 'cal': peak, _ = find_peak(tftp_tp.nda, np.arange(0, np.power(2, 16), 2))
    if proc_label == 'dsp': peak    = db['channels'][channel]['calibration']['dsp_opt']
    
    # plot the distribution
    fig, ax = plt.subplots()
    if proc_label == 'cal': bins = np.arange(*(np.array(window) * (peak / db['lines']['tlfep'])),  2)
    if proc_label == 'dsp': bins = np.arange(0, np.power(2, 16), 100)
    ax.hist(tftp_tp.nda, bins, histtype='stepfilled', alpha=0.5, log=True, label=channel)
    ax.set_xlabel('Energy [ADU]')
    ax.set_ylabel('Number of events')
    ax.set_xlim(bins[0], bins[-1])
    ax.legend(loc='upper right')
    ax.axvline(peak, ls='--', c='k', lw=1)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_unc.savefig()
    plt.close()
    
    # the calibration factor
    adu_to_kev = db['lines']['tlfep'] / peak
    
    # extract the data
    sat_count = sat_lo   .nda + sat_hi .nda
    pftp_diff = pftp_hi  .nda - pftp_lo.nda
    eftp      = tftp_tp  .nda * adu_to_kev
    tp_00     = tp_00    .nda / 1000 / db['tick_width']
    tp_00_dqc = tp_00_dqc.nda / 1000 / db['tick_width']

    # check the processing label
    if proc_label == 'cal':
        # find events at the calibration peak
        idx_cal = np.where(
            (window[0] < eftp) &\
            (window[1] > eftp))[0]
        
        # select the events of interest
        sat_count = sat_count[idx_cal]
        pftp_diff = pftp_diff[idx_cal]
        eftp      = eftp     [idx_cal]
        tp_00     = tp_00    [idx_cal]
        tp_00_dqc = tp_00_dqc[idx_cal]
    
    # plot the distribution
    fig, ax = plt.subplots()
    if proc_label == 'cal': bins = np.arange(*window,   2 * adu_to_kev)
    if proc_label == 'dsp': bins = np.arange(0, 3500, 100 * adu_to_kev)
    ax.hist(eftp, bins, histtype='stepfilled', alpha=0.5, log=True, label=channel)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Number of events')
    ax.set_xlim(bins[0], bins[-1])
    ax.legend(loc='upper right')
    ax.axvline(db['lines']['tlfep'], ls='--', c='k', lw=1)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_cal.savefig()
    plt.close()
    
    # the selection window
    index_lo = int(db['tick_count'] / 2) - int(3 / db['tick_width'])
    index_hi = int(db['tick_count'] / 2)
    
    # plot the distribution
    fig, ax = plt.subplots()
    bin_width = 10
    bins = np.arange(0, db['tick_count'] + bin_width, bin_width)
    ax.hist(tp_00    , bins, alpha=0.5, log=True, label=channel)
    ax.hist(tp_00_dqc, bins, alpha=0.5, log=True, label=channel+', DQC')
    ax.set_xlabel('Time point [c.t.]')
    ax.set_ylabel('Number of events')
    ax.set_xlim(0, db['tick_count'])
    ax.legend(loc='upper right')
    ax.axvspan(index_lo, index_hi, color='k', lw=1, alpha=0.1)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_tps.savefig()
    plt.close()
    
    # check the processing label
    if proc_label == 'cal':
        # select the events
        idx_dqc = idx_cal[
            (np.isfinite(eftp)    ) &
            (tp_00     >  index_lo) &
            (tp_00     <  index_hi) &
            (tp_00_dqc >  index_lo) &
            (tp_00_dqc <  index_hi) &
            (sat_count == 0       ) &
            (pftp_diff >  0       )]

        # translate the index to the raw file
        mask    = rd_mask(f'{channel}_cal_def')
        idx_raw = np.arange(len(mask))[mask][idx_dqc]
        
        # write the index mask to disk
        wr_mask(len(mask), idx_raw, f'{channel}_cal_dqc')
        
        # include the information
        db['channels'][channel]['calibration']['dsp_def'] = peak
        
    # check the processing label
    if proc_label == 'dsp':
        # select the events
        idx_dqc = np.where(
            (np.isfinite(eftp)    ) &
            (tp_00     >  index_lo) &
            (tp_00     <  index_hi) &
            (tp_00_dqc >  index_lo) &
            (tp_00_dqc <  index_hi) &
            (sat_count == 0       ) &
            (pftp_diff >  0       ))[0]

        # select the events
        idx_xtk = np.where(
            (np.isfinite(eftp)    ) &
            (tp_00     >  index_lo) &
            (tp_00_dqc >  index_lo) &
            (tp_00_dqc <  index_hi) &
            (sat_count == 0       ))[0]
        
        # write the index mask to disk
        wr_mask(len(eftp), idx_dqc, f'{channel}_all_dqc')
        wr_mask(len(eftp), idx_xtk, f'{channel}_all_xtk')
        
    # include the statistics
    stats[0] += len(tp_00)
    stats[1] += len(np.where(~np.isfinite(eftp))[0])
    stats[2] += len(np.where(sat_count != 0    )[0])
    stats[3] += len(np.where(pftp_diff <= 0    )[0])
    stats[4] += len(np.where(
        (tp_00     <  index_lo) |
        (tp_00     >  index_hi) |
        (tp_00_dqc <  index_lo) |
        (tp_00_dqc >  index_hi))[0])

    # status printout
    logging.info(f'Finished processing {channel}')
    
# status printout
logging.info(f'Found {(stats[1] / stats[0]) * 100:.2f} % cut by non-finite energy')
logging.info(f'Found {(stats[2] / stats[0]) * 100:.2f} % cut by saturation'       )
logging.info(f'Found {(stats[3] / stats[0]) * 100:.2f} % cut by sample difference')
logging.info(f'Found {(stats[4] / stats[0]) * 100:.2f} % cut by time point'       )

# close the output files
pdf_unc.close()
pdf_cal.close()
pdf_tps.close()

# update the database
update_database(db)
