# import python modules
from setup import *

# set which outputs to save to disk
processors['outputs'] = [
    'wf_bl']

# the energy bin information
e_bin_width = 50
e_bin_count = int(db['lines']['tlfep'] / e_bin_width) + 2

# process the events for the specified file and channel
def process_events(args):
    # interpret the arguments
    i_fi, i_ch = args

    # the mask
    all_dqc = find_mask('all_dqc', i_fi, channels[i_ch])
    
    # the events to process
    idx = np.arange(len(all_dqc))[all_dqc]
    
    # read in the tables
    waveform, _ = store.read_object(f'{channels[i_ch]}/raw/waveform', files_raw[i_fi], idx=idx)
    tftp_tp , _ = store.read_object(f'{channels[i_ch]}/tftp_tp'     , files_dsp[i_fi], idx=idx)
    
    # build the table for processing
    tb_data = lh5.Table(col_dict={'waveform': waveform})
    
    # copy the dictionary
    pcs = copy.deepcopy(processors)
    
    # this channel's parameters
    if pole_count == 1: tau = db['channels'][channels[i_ch]]['pole_zero']['avg_tau0']
    if pole_count == 2: tau = db['channels'][channels[i_ch]]['pole_zero']['avg_tau2']

    # update the processors
    pcs['processors']['wf_bl']['args'][2] = f'{tau}*us'

    # build the processing chain
    pc, tb_out = bpc.build_processing_chain(tb_data, pcs, verbosity=0)

    # process the events
    pc.execute()
    
    # extract the database parameters
    dsp_opt = db['channels'][channels[i_ch]]['calibration']['dsp_opt']
    fit_par = db['channels'][channels[i_ch]]['resolution' ]['fit_par']
    
    # extract the parameters for processing
    cen = tftp_tp.nda * (db['lines']['tlfep'] / dsp_opt)
    
#     # check that the parameters are valid
#     if np.all(fit_par != 0):
#         # the correction variable
#         var = db['channels'][channels[i_ch]]['optimization']['opt_var']
#         
#         # read in the tables
#         ct_corr, _ = store.read_object(f'{channels[i_ch]}/{var}', files_dsp[i_fi], idx=idx)
#         
#         # apply the correction
#         cen = cen - ct_corr.nda * sqrt_func(cen, *fit_par)
    
    # sort the events by their energy
    cen_sorted, iwf_sorted = zip(*sorted(zip(cen, np.arange(len(cen)))))
    
    # the result in each bin
    tot_binned = []
    cnt_binned = []
    
    # the event counter
    i_prev = 0
    
    # loop over the bins
    for i in range(e_bin_count):
        # find the index
        i_curr = np.argmax(np.array(cen_sorted) > i * e_bin_width)
        
        # the result
        wf = np.zeros(db['tick_count'])
        
        # loop over the events in this bin
        for iwf in iwf_sorted[i_prev:i_curr]:
            # include the waveform
            wf += tb_out['wf_bl'].nda[iwf]
            
        # include the result
        tot_binned.append(wf)
        cnt_binned.append(int(i_curr - i_prev))
            
        # update the index
        i_prev = i_curr
    
    # build the output table
    tb_ana = lh5.Table(col_dict={
        'wf_bl': lh5.Array(np.array(tot_binned)),
        'count': lh5.Array(np.array(cnt_binned))})

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
    pdf_bls = pdf.PdfPages(fig_dir + 'p_ana_pze_bls.pdf')
    pdf_wfc = pdf.PdfPages(fig_dir + 'p_ana_pze_wfc.pdf')
    pdf_wfs = pdf.PdfPages(fig_dir + 'p_ana_pze_wfs.pdf')
    pdf_vse = pdf.PdfPages(fig_dir + 'p_ana_pze_vse.pdf')
    pdf_rel = pdf.PdfPages(fig_dir + 'p_ana_pze_rel.pdf')
    
    # the result
    ch_avgs = []
    ch_pzcs = []
    ch_cens = []
    ch_taus = []
    
    # loop over the channels
    for channel in channels:
        # read in the tables
        wf_bl, _ = store.read_object(f'{channel}/wf_bl', files_ana)
        count, _ = store.read_object(f'{channel}/count', files_ana)
        
        # the events in each bin
        avg_binned = []
        pzc_binned = []
        cen_binned = []
        tau_binned = []

        # read in the time constant
        cal_base = db['channels'][channel]['pole_zero']['cal_base']

        # loop over the bins
        for i in range(1, e_bin_count):
            # calculate the average waveform
            avg = np.zeros(db['tick_count'])
            cnt = 0
            for j in range(i, len(files_ana) * e_bin_count, e_bin_count):
                avg += wf_bl.nda[j]
                cnt += count.nda[j]
            
            # check if there are enough events to process
            if cnt > 30:
                # normalize the average waveform
                avg /= cnt
                
                # optimize the pole-zero cancellation
                tau0 = proc.optimize_1pz(
                    avg, 
                    np.average(avg[:1000]), 
                    len(avg) - int(20 / db['tick_width']), 
                    len(avg), 
                    cal_base / db['tick_width'])
                
                # include the result
                avg_binned.append(avg                      )
                pzc_binned.append(proc.pole_zero(avg, tau0))
                cen_binned.append(i                        )
                tau_binned.append(tau0 * db['tick_width']  )
                
        # status printout
        logging.info(f'Finished processing {channel}')

        # select only reasonable results
        avg_binned = np.array(avg_binned)[np.array(tau_binned) < 1000]
        pzc_binned = np.array(pzc_binned)[np.array(tau_binned) < 1000]
        cen_binned = np.array(cen_binned)[np.array(tau_binned) < 1000]
        tau_binned = np.array(tau_binned)[np.array(tau_binned) < 1000]
        
        # remove the bins close to the energy threshold
        avg_binned = avg_binned[2:]
        pzc_binned = pzc_binned[2:]
        cen_binned = cen_binned[2:]
        tau_binned = tau_binned[2:]

        # include the result
        ch_avgs.append(avg_binned)
        ch_pzcs.append(pzc_binned)
        ch_cens.append(cen_binned)
        ch_taus.append(tau_binned)
        
    # the time axis
    time = np.arange(db['tick_count']) * db['tick_width']

    # plot the average waveform
    for i in range(e_bin_count):
        cnt = 0
        for j, channel in enumerate(channels):
            for k, cen in enumerate(ch_cens[j]):
                if cen == i:
                    cnt += 1
        if cnt == 0:
            continue
        fig, ax = plt.subplots()
        for j, channel in enumerate(channels):
            for k, cen in enumerate(ch_cens[j]):
                if cen == i:
                    ax.plot(time, ch_avgs[j][k], label=channel)
        ax.set_xlabel(r'Time into waveform [$\mu$s]')
        ax.set_ylabel('Waveform value [ADU]')
        ax.set_title(f'{(i - 0.5) * e_bin_width} keV')
        ax.set_xlim(0, db['tick_count'] * db['tick_width'])
        ax.legend(loc='upper left', ncol=int(np.ceil(len(channels) / 10)))
        cutoff = (db['tick_count'] - int(20 / db['tick_width'])) * db['tick_width']
        ax.axvline(cutoff, ls='--', c='k', lw=1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        pdf_bls.savefig()
        plt.close()
        
    # plot the average waveform
    for i in range(e_bin_count):
        cnt = 0
        for j, channel in enumerate(channels):
            for k, cen in enumerate(ch_cens[j]):
                if cen == i:
                    cnt += 1
        if cnt == 0:
            continue
        fig, ax = plt.subplots()
        for j, channel in enumerate(channels):
            for k, cen in enumerate(ch_cens[j]):
                if cen == i:
                    den = np.average(ch_pzcs[j][k][len(ch_pzcs[j][k]) - int(20 / db['tick_width']):])
                    ax.plot(time, ch_pzcs[j][k] / den, label=channel)
        ax.set_xlabel(r'Time into waveform [$\mu$s]')
        ax.set_ylabel('Waveform value [a.u.]')
        ax.set_title(f'{(i - 0.5) * e_bin_width} keV')
        ax.set_xlim(25 + (db['tick_count'] * db['tick_width']) % 1, db['tick_count'] * db['tick_width'])
        ax.set_ylim(0.998, 1.002)
        ax.legend(loc='lower right', ncol=int(np.ceil(len(channels) / 4)))
        cutoff = (db['tick_count'] - int(20 / db['tick_width'])) * db['tick_width']
        ax.axvline(cutoff, ls='--', c='k', lw=1)
        ax.axhline(1, ls='--', c='k', lw=1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        pdf_wfc.savefig()
        plt.close()

    # plot the average waveform
    for channel, cen_binned, pzc_binned in zip(channels, ch_cens, ch_pzcs):
        for cen, pzc in zip(cen_binned, pzc_binned):
            fig, ax = plt.subplots()
            den = np.average(pzc[len(pzc) - int(20 / db['tick_width']):])
            ax.plot(time, pzc / den, label=channel)
            ax.set_xlabel(r'Time into waveform [$\mu$s]')
            ax.set_ylabel('Waveform value [a.u.]')
            ax.set_title(f'{(cen - 0.5) * e_bin_width} keV')
            ax.set_xlim(25 + (db['tick_count'] * db['tick_width']) % 1, db['tick_count'] * db['tick_width'])
            ax.set_ylim(
                min(pzc[int(30 / db['tick_width']):]) / den - 0.001, 
                max(pzc[int(30 / db['tick_width']):]) / den + 0.001)
            ax.legend(loc='upper left')
            cutoff = (db['tick_count'] - int(20 / db['tick_width'])) * db['tick_width']
            ax.axvline(cutoff, ls='--', c='k', lw=1)
            ax.axhline(1, ls='--', c='k', lw=1)
            ax.get_yaxis().set_label_coords(-0.1, 0.5)
            pdf_wfs.savefig()
            plt.close()
    
    # plot the result
    for i, channel in enumerate(channels):
        if len(ch_cens[i]) > 0:
            fig, ax = plt.subplots()
            x = [(c - 0.5) * e_bin_width for c in ch_cens[i]]
            ax.errorbar(
                x, ch_taus[i], 
                marker='.', ms=10, ls='-', lw=1, label=channel)
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel(r'Time constant [$\mu$s]')
            ax.set_xlim(0, e_bin_count * e_bin_width)
            ax.legend()
            ax.get_yaxis().set_label_coords(-0.1, 0.5)
            pdf_vse.savefig()
            plt.close()

    # plot the result
    fig, ax = plt.subplots()
    for i, channel in enumerate(channels):
        if len(ch_cens[i]) > 0:
            count = min(5, len(ch_taus[i]))
            x = [(c - 0.5) * e_bin_width for c in ch_cens[i]]
            ax.errorbar(
                x, ch_taus[i] - np.average(ch_taus[i][-count:]), 
                marker='.', ms=10, ls='-', lw=1, label=channel)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel(r'Relative time constant [$\mu$s]')
    ax.set_xlim(0, e_bin_count * e_bin_width)
    ax.legend(ncol=int(np.ceil(len(channels) / 4)))
    ax.axhline(0, ls='--', color='k', lw=1)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_rel.savefig()
    plt.close()

    # close the output files
    pdf_bls.close()
    pdf_wfc.close()
    pdf_wfs.close()
    pdf_vse.close()
    pdf_rel.close()

# process based on the configuration
if proc_stage in ['', 'produce']: produce()
if proc_stage in ['', 'analyze']: analyze()
