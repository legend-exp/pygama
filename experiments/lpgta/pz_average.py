# import python modules
from setup import *

# set which outputs to save to disk
processors['outputs'] = [
    'wf_bl']

# process the events for the specified file and channel
def process_events(args):
    # interpret the arguments
    i_fi, i_ch = args

    # the mask
    cal_dqc = find_mask('cal_dqc', i_fi, channels[i_ch])
    
    # the events to process
    idx = np.arange(len(cal_dqc))[cal_dqc]
    
    # read in the tables
    waveform, _ = store.read_object(f'{channels[i_ch]}/raw/waveform', files_raw[i_fi], idx=idx)

    # build the table for processing
    tb_data = lh5.Table(col_dict={'waveform': waveform})
    
    # copy the dictionary
    pcs = copy.deepcopy(processors)
    
    # this channel's parameters
    if pole_count == 1: tau = db['channels'][channels[i_ch]]['pole_zero']['fit_tau0']
    if pole_count == 2: tau = db['channels'][channels[i_ch]]['pole_zero']['fit_tau2']

    # update the processors
    pcs['processors']['wf_bl']['args'][2] = f'{tau}*us'

    # build the processing chain
    pc, tb_out = bpc.build_processing_chain(tb_data, pcs, verbosity=0)

    # process the events
    pc.execute()
    
    # the result
    wf_bl = np.zeros(db['tick_count'])
    
    # look over the waveforms
    for w in tb_out['wf_bl'].nda:
        # include the waveform
        wf_bl += w
    
    # build the output table
    tb_ana = lh5.Table(col_dict={
        'wf_bl': lh5.Array(np.array([wf_bl   ])),
        'count': lh5.Array(np.array([len(idx)]))})

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
    pdf_bls = pdf.PdfPages(fig_dir + 'p_cal_pza_bls.pdf')
    pdf_spu = pdf.PdfPages(fig_dir + 'p_cal_pza_spu.pdf')
    pdf_dpu = pdf.PdfPages(fig_dir + 'p_cal_pza_dpu.pdf')
    pdf_spn = pdf.PdfPages(fig_dir + 'p_cal_pza_spn.pdf')
    pdf_dpn = pdf.PdfPages(fig_dir + 'p_cal_pza_dpn.pdf')
    
    # the average waveforms
    averages_bl = []
    averages_1p = []
    averages_2p = []
    
    # loop over the channels
    for channel in channels:
        # read in the tables
        wf_bl, _ = store.read_object(f'{channel}/wf_bl', files_ana)
        count, _ = store.read_object(f'{channel}/count', files_ana)
        
        # calculate the average waveform
        average_bl = np.zeros(db['tick_count'])
        for w in wf_bl.nda:
            average_bl += w
        average_bl /= sum(count.nda)
        
        # read in the time constant
        fit_tau0 = db['channels'][channel]['pole_zero']['fit_tau0']
        fit_tau1 = db['channels'][channel]['pole_zero']['fit_tau1']
        fit_tau2 = db['channels'][channel]['pole_zero']['fit_tau2']
        fit_frac = db['channels'][channel]['pole_zero']['fit_frac']

        # optimize the pole-zero cancellation
        tau0 = proc.optimize_1pz(
            average_bl, 
            np.average(average_bl[:1000]), 
            len(average_bl) - int(20 / db['tick_width']), 
            len(average_bl), 
            fit_tau0 / db['tick_width'])

        # optimize the pole-zero cancellation
        tau1, tau2, frac = proc.optimize_2pz(
            average_bl, 
            np.average(average_bl[:1000]), 
            len(average_bl) - int(20 / db['tick_width']), 
            len(average_bl), 
            fit_tau1 / db['tick_width'], fit_tau2 / db['tick_width'], fit_frac)

        # calculate the average waveforms
        average_1p = proc.pole_zero       (average_bl, tau0            )
        average_2p = proc.double_pole_zero(average_bl, tau1, tau2, frac)

        # include the waveforms
        averages_bl.append(average_bl)
        averages_1p.append(average_1p)
        averages_2p.append(average_2p)
        
        # convert the units
        tau0 *= db['tick_width']
        tau1 *= db['tick_width']
        tau2 *= db['tick_width']

        # include the information
        db['channels'][channel]['pole_zero']['avg_tau0'] = tau0
        db['channels'][channel]['pole_zero']['avg_tau1'] = tau1
        db['channels'][channel]['pole_zero']['avg_tau2'] = tau2
        db['channels'][channel]['pole_zero']['avg_frac'] = frac
        
        # status printout
        logging.info(f'Found parameters for {channel} of ' +\
            f'{tau0:.1f}, '     +\
            f'{tau1:.1f}, '     +\
            f'{tau2:.1f}, and ' +\
            f'{frac:.4f}')

    # the time axis
    time = np.arange(db['tick_count']) * db['tick_width']

    # plot the average waveform
    fig, ax = plt.subplots()
    for channel, average in zip(channels, averages_bl):
        ax.plot(time, average, label=channel)
    ax.set_xlabel(r'Time into waveform [$\mu$s]')
    ax.set_ylabel('Waveform value [ADU]')
    ax.set_xlim(0, db['tick_count'] * db['tick_width'])
    ax.legend(loc='upper left', ncol=int(np.ceil(len(channels) / 10)))
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_bls.savefig()
    plt.close()
    
    # plot the average waveform
    fig, ax = plt.subplots()
    for channel, average in zip(channels, averages_1p):
        den = np.average(average[len(average) - int(20 / db['tick_width']):])
        ax.plot(time, average / den, label=channel)
    ax.set_xlabel(r'Time into waveform [$\mu$s]')
    ax.set_ylabel('Waveform value [a.u.]')
    ax.set_xlim(25 + (db['tick_count'] * db['tick_width']) % 1, db['tick_count'] * db['tick_width'])
    ax.set_ylim(0.998, 1.002)
    ax.legend(loc='lower right', ncol=int(np.ceil(len(channels) / 4)))
    ax.axhline(1, ls='--', c='k', lw=1)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_spu.savefig()
    plt.close()

    # plot the average waveform
    fig, ax = plt.subplots()
    for channel, average in zip(channels, averages_2p):
        den = np.average(average[len(average) - int(20 / db['tick_width']):])
        ax.plot(time, average / den, label=channel)
    ax.set_xlabel(r'Time into waveform [$\mu$s]')
    ax.set_ylabel('Waveform value [a.u.]')
    ax.set_xlim(25 + (db['tick_count'] * db['tick_width']) % 1, db['tick_count'] * db['tick_width'])
    ax.set_ylim(0.998, 1.002)
    ax.legend(loc='lower right', ncol=int(np.ceil(len(channels) / 4)))
    ax.axhline(1, ls='--', c='k', lw=1)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_dpu.savefig()
    plt.close()

    # plot the average waveform
    for channel, average in zip(channels, averages_1p):
        fig, ax = plt.subplots()
        den = np.average(average[len(average) - int(20 / db['tick_width']):])
        ax.plot(time, average / den, label=channel)
        ax.set_xlabel(r'Time into waveform [$\mu$s]')
        ax.set_ylabel('Waveform value [a.u.]')
        ax.set_xlim(25 + (db['tick_count'] * db['tick_width']) % 1, db['tick_count'] * db['tick_width'])
        ax.set_ylim(
            min(average[int(30 / db['tick_width']):]) / den - 0.001, 
            max(average[int(30 / db['tick_width']):]) / den + 0.001)
        ax.legend(loc='upper right')
        ax.axhline(1, ls='--', c='k', lw=1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        pdf_spn.savefig()
        plt.close()

    # plot the average waveform
    for channel, average in zip(channels, averages_2p):
        fig, ax = plt.subplots()
        den = np.average(average[len(average) - int(20 / db['tick_width']):])
        ax.plot(time, average / den, label=channel)
        ax.set_xlabel(r'Time into waveform [$\mu$s]')
        ax.set_ylabel('Waveform value [a.u.]')
        ax.set_xlim(25 + (db['tick_count'] * db['tick_width']) % 1, db['tick_count'] * db['tick_width'])
        ax.set_ylim(
            min(average[int(30 / db['tick_width']):]) / den - 0.001, 
            max(average[int(30 / db['tick_width']):]) / den + 0.001)
        ax.legend(loc='upper right')
        ax.axhline(1, ls='--', c='k', lw=1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        pdf_dpn.savefig()
        plt.close()

    # close the output files
    pdf_bls.close()
    pdf_spu.close()
    pdf_dpu.close()
    pdf_spn.close()
    pdf_dpn.close()

    # update the database
    update_database(db)

# process based on the configuration
if proc_stage in ['', 'produce']: produce()
if proc_stage in ['', 'analyze']: analyze()
