# import python modules
from setup import *

# the linear-constrained model
def linear_constrained(beta, x):
    return beta[0] * x

# find the two-channel match candidates
@numba.jit(nopython=True)
def find_candidates(ps_idx, ch_pps, ch_tks, i_ch_neg, idx, ts_pps, ts_tks):
    # the candidates
    v_evt_neg  = np.full(len(idx), -1, dtype=np.int32)
    v_i_ch_pos = np.full(len(idx), -1, dtype=np.int32)
    v_evt_pos  = np.full(len(idx), -1, dtype=np.int32)
    
    # loop over the events
    for i_idx in range(len(idx)):
        # extract the information
        evt_neg = idx   [i_idx]
        pps_neg = ts_pps[i_idx]
        tks_neg = ts_tks[i_idx]
        
        # the event information
        i_ch_pos = -1
        ch_pos   = -1
        evt_pos  = -1
        
        # a flag for a multiple-channel match
        skip = False

        # loop over the channels
        for i in range(len(ch_pps)):            
            # skip if the same channel or an empty channel
            if i == i_ch_neg or\
               ps_idx[pps_neg-1][i] == -1:
                continue
                
            # the matching event
            evt = -1
            
            # loop over the events
            for j in range(ps_idx[pps_neg-1][i], len(ch_pps[i])):
                # skip the rest if the pps is larger
                if ch_pps[i][j] > pps_neg:
                    break
                
                # check if the timestamps match
                if ch_pps[i][j] == pps_neg and\
                   ch_tks[i][j] == tks_neg:
                    # update the event index
                    evt = j
                        
            # check if a match was found
            if evt > 0:
                # skip the rest if more than one match was found
                if i_ch_pos > 0:
                    skip = True
                    break
                
                # update the matching event
                i_ch_pos = i
                evt_pos  = evt
        
        # skip if no match was found
        if i_ch_pos == -1:
            skip = True
        
        # check if only one match was found
        if not skip:
            # include the candidate
            v_evt_neg [i_idx] = evt_neg
            v_i_ch_pos[i_idx] = i_ch_pos
            v_evt_pos [i_idx] = evt_pos
                
    # return the candidates
    return v_evt_neg, v_i_ch_pos, v_evt_pos
    
# process the events for the specified channel
def process_events(args):
    # interpret the arguments
    i_fi, i_ch_neg, l_ch_pps, l_ch_tks, l_ps_idx = args
        
    # this channel
    ch_neg = channels[i_ch_neg]
    
    # the mask
    all_xtk = find_mask('all_xtk', i_fi, ch_neg)
    
    # the events to process
    idx = np.arange(len(all_xtk))[all_xtk]
    
    # the event information
    ch_pps = numba.typed.List()
    ch_tks = numba.typed.List()
    ps_idx = numba.typed.List()
    
    # include the event information
    for l in l_ch_pps: ch_pps.append(l)
    for l in l_ch_tks: ch_tks.append(l)
    for l in l_ps_idx: ps_idx.append(l)
    
    # read in the tables
    ts_pps = ch_pps[i_ch_neg][idx]
    ts_tks = ch_tks[i_ch_neg][idx]
    
    # find the candidates
    candidates = find_candidates(ps_idx, ch_pps, ch_tks, i_ch_neg, idx, ts_pps, ts_tks)
    
    # the two-channel matches
    matches = [[] for channel in channels]

    # loop over the candidates
    for i in range(len(candidates)):
        # extract the information
        evt_neg  = candidates[0][i]
        i_ch_pos = candidates[1][i]
        evt_pos  = candidates[2][i]
        
        # skip if not a candidate
        if evt_neg == -1:
            continue
        
        # the channel
        ch_pos = channels[i_ch_pos]
        
        # read in the tables
        raw_neg, _ = store.read_object(f'{ch_neg}/raw/waveform/values', files_raw[i_fi], idx=np.array([evt_neg]))
        raw_pos, _ = store.read_object(f'{ch_pos}/raw/waveform/values', files_raw[i_fi], idx=np.array([evt_pos]))
        std_neg, _ = store.read_object(f'{ch_neg}/bl_std'             , files_dsp[i_fi], idx=np.array([evt_neg]))
        std_pos, _ = store.read_object(f'{ch_pos}/bl_std'             , files_dsp[i_fi], idx=np.array([evt_pos]))
        fta_neg, _ = store.read_object(f'{ch_neg}/tf_avg'             , files_dsp[i_fi], idx=np.array([evt_neg]))
        fta_pos, _ = store.read_object(f'{ch_pos}/tf_avg'             , files_dsp[i_fi], idx=np.array([evt_pos]))
        fts_neg, _ = store.read_object(f'{ch_neg}/tf_std'             , files_dsp[i_fi], idx=np.array([evt_neg]))
        fts_pos, _ = store.read_object(f'{ch_pos}/tf_std'             , files_dsp[i_fi], idx=np.array([evt_pos]))
        ftp_neg, _ = store.read_object(f'{ch_neg}/tftp_31'            , files_dsp[i_fi], idx=np.array([evt_neg]))
        ftp_pos, _ = store.read_object(f'{ch_pos}/tftp_31'            , files_dsp[i_fi], idx=np.array([evt_pos]))

        # check the number of poles
        if pole_count == 1:
            # this channel's parameters
            tau_neg = db['channels'][ch_neg]['pole_zero']['avg_tau0']
            tau_pos = db['channels'][ch_pos]['pole_zero']['avg_tau0']

        # check the number of poles
        if pole_count == 2:
            # this channel's parameters
            tau_neg = db['channels'][ch_neg]['pole_zero']['avg_tau2']
            tau_pos = db['channels'][ch_pos]['pole_zero']['avg_tau2']

        # the baseline-subtracted waveforms
        wbl_neg = proc.soft_pileup_corr(raw_neg.nda[0], 1000, int(tau_neg / db['tick_width']))
        wbl_pos = proc.soft_pileup_corr(raw_pos.nda[0], 1000, int(tau_pos / db['tick_width']))

        # the rise times
        rise_neg = db['channels'][ch_neg]['optimization']['opt_par'][0] / db['tick_width']
        rise_pos = db['channels'][ch_pos]['optimization']['opt_par'][0] / db['tick_width']

        # the noise levels
        noise_neg = std_neg.nda[0] * np.sqrt(2) / np.sqrt(rise_neg)
        noise_pos = std_pos.nda[0] * np.sqrt(2) / np.sqrt(rise_pos)

        # check if this event pair passes the cuts
        if fta_neg.nda[0] < -noise_neg and ftp_neg.nda[0] < 0 and wbl_neg[0] > -100 and\
           fta_pos.nda[0] >  noise_pos and ftp_pos.nda[0] > 0 and wbl_pos[0] > -100:
            # include the two-channel match
            matches[i_ch_pos].append([
                [i_fi, evt_neg, fta_neg.nda[0], fts_neg.nda[0]],
                [i_fi, evt_pos, fta_pos.nda[0], fts_pos.nda[0]]])
    
    # build the output table
    tb_ana = lh5.Table(col_dict={
        'matches': lh5.Array(np.array(matches))})

    # write the output to disk
    lock .acquire()
    store.write_object(tb_ana, ch_neg, files_ana[i_fi])
    lock .release()

    # status printout
    logging.info(f'Finished processing {ch_neg} in {os.path.basename(files_raw[i_fi])}')

# the stage that produces the data
def produce():
    # loop over the files
    for i_fi, file_raw in enumerate(files_raw):
        # skip if not the selected file
        if proc_index and i_fi not in proc_index:
            continue
        
        # delete the output file if it exists
        if os.path.isfile(files_ana[i_fi]):
            os.remove(files_ana[i_fi])
        
        # the timestamp information per channel
        ch_pps = []
        ch_tks = []

        # loop over the channels
        for channel in channels:
            # read in the tables
            ts_pps, _ = store.read_object(f'{channel}/raw/ts_pps'  , file_raw)
            ts_tks, _ = store.read_object(f'{channel}/raw/ts_ticks', file_raw)

            # append the tables
            ch_pps.append(ts_pps.nda)
            ch_tks.append(ts_tks.nda)

        # the maximum pps signal
        pps_max = max([pps[-1] for pps in ch_pps])

        # the first index for each pps signal
        ps_idx = []

        # loop over the pps signals
        for pps in range(1, pps_max + 1):
            # the index list
            idx = []

            # loop over the channels
            for i in range(len(channels)):
                # find the first index for this pps signal
                j = np.argmax(ch_pps[i] == pps)
                    
                # include the index
                idx.append(j if ch_pps[i][j] == pps else -1)

            # include the index list
            ps_idx.append(np.array(idx, dtype='int32'))
            
        # the argument list
        args = [[
            i_fi  , 
            i_ch  ,
            ch_pps,
            ch_tks,
            ps_idx] for i_ch in range(len(channels))]
        
        # launch the parallel processes
        with mp.Pool(proc_count) as p:
            p.map(process_events, args)

# the stage that analyzes the data
def analyze():
    # the output files
    pdf_sca = pdf.PdfPages(fig_dir + 'p_ana_cro_sca.pdf')
    pdf_avg = pdf.PdfPages(fig_dir + 'p_ana_cro_avg.pdf')
    pdf_mat = pdf.PdfPages(fig_dir + 'p_ana_cro_mat.pdf')
    pdf_adu = pdf.PdfPages(fig_dir + 'p_ana_cro_adu.pdf')
    pdf_kev = pdf.PdfPages(fig_dir + 'p_ana_cro_kev.pdf')
    
    # the two-channel matches
    ch_matches = [[[] for j in channels] for i in channels]
    
    # note: cannot use store.read_object due to a bug in h5py
    # note: for details, see https://github.com/h5py/h5py/issues/1792
    
    # loop over the files
    for file_ana in files_ana:
        # open this file
        with h5py.File(file_ana, 'r') as f:
            # loop over the channels
            for i, channel in enumerate(channels):
                # loop over the channels
                for j in range(len(channels)):
                    # the results
                    results = f[f'{channel}/matches'][j]
                    
                    # include the results
                    ch_matches[i][j] = np.concatenate((ch_matches[i][j], results))
    
    # the average waveform per channel
    ch_average = []
    
    # loop over the channels
    for i, channel in enumerate(channels):
        # extract the results
        events = [k_m[0][:2] for i_m in ch_matches[i] for j_m in i_m for k_m in j_m]
        
        # status printout
        logging.info(f'Found {len(events)} matches for {channel}')

        # the average waveform
        average = np.zeros(db['tick_count'])

        # the waveform count
        count = 0

        # this channel's parameters
        if pole_count == 1: tau_neg = db['channels'][channel]['pole_zero']['avg_tau0']
        if pole_count == 2: tau_neg = db['channels'][channel]['pole_zero']['avg_tau2']
        
        # loop over the events
        for i_fi, i_ev in events:
            # the mask
            all_dqc = find_mask('all_dqc', i_fi, channel)
            
            # check if this event is of interest
            if all_dqc[i_fi][i_ev]:
                # read in the tables
                raw_neg, _ = store.read_object(f'{channel}/raw/waveform/values', files_raw[i_fi], idx=np.array([i_ev]))

                # update the waveform average
                average += proc.soft_pileup_corr(raw_neg.nda[0], 1000, int(tau_neg / db['tick_width']))
                count   += 1

        # normalize the waveform average
        if count > 0:
            average /= count

        # append the results
        ch_average.append(average)

    # plot the average waveform
    fig, ax = plt.subplots()
    time = np.arange(db['tick_count']) * db['tick_width']
    plotted = False
    for channel, average in zip(channels, ch_average):
        if not np.all(average == 0):
            ax.plot(time, average, label=channel)
            plotted = True
    ax.set_xlabel(r'Time into waveform [$\mu$s]')
    ax.set_ylabel('Waveform value [ADU]')
    ax.set_xlim(0, db['tick_count'] * db['tick_width'])
    if plotted:
        ax.legend()
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_avg.savefig()
    plt.close()

    # the amplitude fractions
    ch_fit_adu = [np.full(len(channels), np.nan) for channel in channels]
    ch_fit_kev = [np.full(len(channels), np.nan) for channel in channels]

    # loop over the channels
    for i in range(len(channels)):
        # loop over the channels
        for j in range(len(channels)):
            # extract the negative amplitudes
            avg_neg = [m[0][2] for m in ch_matches[i][j]]
            std_neg = [m[0][3] for m in ch_matches[i][j]]

            # extract the positive amplitudes
            avg_pos = [m[1][2] for m in ch_matches[i][j]]
            std_pos = [m[1][3] for m in ch_matches[i][j]]

            # check that there are enough data for a fit
            if len(avg_neg) > 30:
                # set up the fit function
                model = odr.Model(linear_constrained)
                data  = odr.RealData(avg_pos, avg_neg, sx=std_pos, sy=std_neg)
                i_odr = odr.ODR(data, model, beta0=[-0.001])

                # fit the data
                i_odr.set_job(fit_type=0)
                output = i_odr.run()

                # include the results
                ch_fit_adu[i][j] = output.beta[0]
                ch_fit_kev[i][j] = output.beta[0] * (\
                    db['channels'][channels[j]]['calibration']['dsp_opt'] /\
                    db['channels'][channels[i]]['calibration']['dsp_opt'])

    # plot the data
    for i in range(len(channels)):
        fig, ax = plt.subplots()
        plotted = False
        for j in range(len(channels)):
            if not np.isnan(ch_fit_adu[i][j]):
                p = ax.errorbar(
                    x   =[m[0][2] for m in ch_matches[i][j]],
                    y   =[m[1][2] for m in ch_matches[i][j]], 
                    xerr=[m[0][3] for m in ch_matches[i][j]], 
                    yerr=[m[1][3] for m in ch_matches[i][j]], 
                    marker='.', ms=10, ls='None', lw=1, label=channels[j])
                x = np.linspace(-65, 0, 10)
                y = linear_constrained([*ch_fit_adu[i][j]], x)
                ax.plot(x, y, c=p.get_facecolors()[0].tolist(), ls='--', lw=1)
                plotted = True
        ax.set_xlabel(f'Amplitude in {channels[i]} [ADU]')
        ax.set_ylabel('Amplitude [ADU]')
        ax.set_xlim(-65, 0)
        ax.set_ylim(0, 30000)
        if plotted:
            ax.legend()
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        pdf_sca.savefig()
        plt.close() 

    # the data for plotting
    w_cnt = []
    w_adu = []
    w_kev = []

    # loop over the channels
    for i in range(len(channels)):
        # loop over the channels
        for j in range(len(channels)):
            # include the data
            w_cnt.append(len(ch_matches[i][j]))
            w_adu.append(ch_fit_adu[i][j] * 100)
            w_kev.append(ch_fit_kev[i][j] * 100)

    # extract the channel index
    idx = np.array([int(channel[1:]) for channel in channels])

    # the axis labels
    xlabel = 'Channel with negative signal'
    ylabel = 'Channel with positive signal'

    # plot the matrices
    plot_matrix(idx, idx, w_cnt, pdf_mat, xlabel, ylabel, 'Number of events'      )
    plot_matrix(idx, idx, w_adu, pdf_adu, xlabel, ylabel, 'Amplitude fraction [%]')
    plot_matrix(idx, idx, w_kev, pdf_kev, xlabel, ylabel, 'Energy fraction [%]'   )

    # close the output files
    pdf_sca.close()
    pdf_avg.close()
    pdf_mat.close()
    pdf_adu.close()
    pdf_kev.close()

# process based on the configuration
if proc_stage in ['', 'produce']: produce()
if proc_stage in ['', 'analyze']: analyze()
