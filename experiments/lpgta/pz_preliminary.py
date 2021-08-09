# import python modules
from setup import *

# the window of interest
window = [200, 700]

# set which outputs to save to disk
processors['outputs'] = [
    'tau_base',
    'tau_tail']

# process the events for the specified file and channel
def process_events(args):
    # interpret the arguments
    i_fi, i_ch = args

    # the mask
    cal_def = find_mask('cal_def', i_fi, channels[i_ch])
    
    # the events to process
    idx = np.arange(len(cal_def))[cal_def][1:]
    
    # read in the tables
    ts_0, _ = store.read_object(f'{channels[i_ch]}/raw/timestamp', files_raw[i_fi], idx=idx-1)
    ts_1, _ = store.read_object(f'{channels[i_ch]}/raw/timestamp', files_raw[i_fi], idx=idx  )
    wf_1, _ = store.read_object(f'{channels[i_ch]}/raw/waveform' , files_raw[i_fi], idx=idx  )
    bl_1, _ = store.read_object(f'{channels[i_ch]}/raw/baseline' , files_raw[i_fi], idx=idx  )
    
    # build the table for processing
    tb_data = lh5.Table(col_dict={
        'waveform': wf_1, 
        'baseline': bl_1})
    
    # build the processing chain
    pc, tb_out = bpc.build_processing_chain(tb_data, processors, verbosity=0)
    
    # process the events
    pc.execute()
    
    # build the output table
    tb_ana = lh5.Table(col_dict={
        'tau_base': lh5.Array(tb_out['tau_base'].nda),
        'tau_tail': lh5.Array(tb_out['tau_tail'].nda),
        'delta_ts': lh5.Array(ts_1.nda - ts_0.nda   )})

    # write the output to disk
    lock .acquire()
    store.write_object(tb_ana, channels[i_ch], files_ana[i_fi])
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

# the stage that analyzes the data
def analyze():
    # the output files
    pdf_tau = pdf.PdfPages(fig_dir + 'p_cal_pzp_tau.pdf')

    # loop over the channels
    for channel in channels:
        # read in the tables
        tau_base, _ = store.read_object(f'{channel}/tau_base', files_ana)
        tau_tail, _ = store.read_object(f'{channel}/tau_tail', files_ana)
        delta_ts, _ = store.read_object(f'{channel}/delta_ts', files_ana)

        # extract the data
        tau_base = tau_base.nda[np.where(delta_ts.nda < 0.002)[0]]
        tau_tail = tau_tail.nda[np.where(delta_ts.nda > 0.002)[0]]

        # find the peak
        bins_base    = find_bins(tau_base[(tau_base > window[0]) & (tau_base < window[1])])
        peak_base, _ = find_peak(tau_base, bins_base)

        # find the peak
        bins_tail    = find_bins(tau_tail[(tau_tail > window[0]) & (tau_tail < window[1])])
        peak_tail, _ = find_peak(tau_tail, bins_tail)

        # plot the distributions
        fig, ax = plt.subplots()
        ax.hist(tau_base, bins_base, histtype='stepfilled', alpha=0.5, log=True, label=channel)
        ax.hist(tau_tail, bins_tail, histtype='stepfilled', alpha=0.5, log=True, label=channel)
        ax.set_xlabel(r'Time constant [$\mu$s]')
        ax.set_ylabel('Number of events')
        ax.set_xlim(*window)
        ax.legend(loc='upper right')
        ax.axvline(peak_base, ls='--', c='tab:blue'  , lw=1)
        ax.axvline(peak_tail, ls='--', c='tab:orange', lw=1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        pdf_tau.savefig()
        plt.close()

        # include the information
        db['channels'][channel]['pole_zero']['cal_base'] = peak_base
        db['channels'][channel]['pole_zero']['cal_tail'] = peak_tail

        # status printout
        logging.info(f'Found time constants for {channel} of {peak_base:.1f} and {peak_tail:.1f} microseconds')

    # close the output files
    pdf_tau.close()

    # update the database
    update_database(db)

# process based on the configuration
if proc_stage in ['', 'produce']: produce()
if proc_stage in ['', 'analyze']: analyze()
