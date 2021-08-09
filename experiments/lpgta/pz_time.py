# import python modules
from setup import *

# the window of interest
window = [100, 1100]

# the double pole-zero cancellation model
def dpzc_func(x, a, b, c):
    return\
        ((1 - c)     * np.exp(-x / a) + c     * np.exp(-x / b)) /\
        ((1 - c) / a * np.exp(-x / a) + c / b * np.exp(-x / b))

# the time-bin width
t_bin_width = 4

# the number of time bins
t_bin_start_1 = 34
t_bin_limit_1 = db['tick_count'] * db['tick_width']
t_bin_count_1 = math.floor((t_bin_limit_1 - t_bin_start_1) / t_bin_width)

# the number of time bins
t_bin_start_2 = t_bin_start_1 + (t_bin_count_1 + 1) * t_bin_width
t_bin_limit_2 = 200
t_bin_count_2 = math.floor((t_bin_limit_2 - t_bin_start_2) / t_bin_width)

# process the events for the specified file and channel
def process_events(args):
    # interpret the arguments
    i_fi, i_ch = args

    # the mask
    cal_dqc = find_mask('cal_dqc', i_fi, channels[i_ch])
    
    # the events to process
    idx = np.arange(len(cal_dqc))[cal_dqc][:-1]
    
    # read in the tables
    wf_1, _ = store.read_object(f'{channels[i_ch]}/raw/waveform', files_raw[i_fi], idx=idx  )
    bl_1, _ = store.read_object(f'{channels[i_ch]}/raw/baseline', files_raw[i_fi], idx=idx  )
    ps_1, _ = store.read_object(f'{channels[i_ch]}/raw/ts_pps'  , files_raw[i_fi], idx=idx  )
    tk_1, _ = store.read_object(f'{channels[i_ch]}/raw/ts_ticks', files_raw[i_fi], idx=idx  )
    wf_2, _ = store.read_object(f'{channels[i_ch]}/raw/waveform', files_raw[i_fi], idx=idx+1)
    ps_2, _ = store.read_object(f'{channels[i_ch]}/raw/ts_pps'  , files_raw[i_fi], idx=idx+1)
    tk_2, _ = store.read_object(f'{channels[i_ch]}/raw/ts_ticks', files_raw[i_fi], idx=idx+1)
    
    # extract the clock unit
    clock_unit = wf_1['dt'].nda[0] * units.unit_parser.parse_unit(wf_1['dt'].attrs['units'])
    
    # loop over the time bins
    for i in range(t_bin_count_1):
        # the processing chain
        pc = ProcessingChain.ProcessingChain(block_width=16, clock_unit=clock_unit, verbosity=0)

        # add the input buffers
        pc.add_input_buffer('waveform', wf_1['values'].nda, dtype='float32')
        pc.add_input_buffer('baseline', bl_1          .nda, dtype='uint16' )
    
        # add the processors
        pc.add_processor(
            proc.optimize_1pz, 
            'waveform', 
            'baseline', 
            f'{t_bin_start_1+(i  )*t_bin_width}*us', 
            f'{t_bin_start_1+(i+1)*t_bin_width}*us', 
            '400*us', 
            'tau0')
    
        # get the output buffers
        tb_out = lh5.Table(size=pc._buffer_len, col_dict={
            'tau0': lh5.Array(pc.get_output_buffer('tau0', unit=units.us))})

        # process the events
        pc.execute()
        
        # build the output table
        tb_ana = lh5.Table(col_dict={
            f'bin_{i}': lh5.Array(tb_out['tau0'].nda)})

        # write the output to disk
        lock .acquire()
        store.write_object(tb_ana, f'{channels[i_ch]}/tau_1', files_ana[i_fi])
        lock .release()
    
    # loop over the time bins
    for i in range(t_bin_count_2):
        # the processing chain
        pc = ProcessingChain.ProcessingChain(block_width=16, clock_unit=clock_unit, verbosity=0)

        # this bin's time range
        bin_lo = t_bin_start_2 + i * t_bin_width
        bin_hi = bin_lo + t_bin_width
        
        # the events to process
        idx_sel = []
        
        # the time ranges to process
        opt_beg = []
        opt_end = []
        
        # loop over the events
        for j in range(len(idx)):
            # if the pps signal is the same
            if ps_1.nda[j] == ps_2.nda[j]:
                # the tick count at the beginning of the event
                start_1 = int(tk_1.nda[j] / 4)
                start_2 = int(tk_2.nda[j] / 4)
                
                # the time range
                beg = (start_1 - start_2) + int((t_bin_start_2 + i * t_bin_width) / db['tick_width'])
                end = beg + int(t_bin_width / db['tick_width'])

                # check the time range
                if beg > 0 and end < min(int(20 / db['tick_width']), db['tick_count']):
                    # append the event
                    idx_sel.append(j)
                    
                    # append the time range
                    opt_beg.append(beg)
                    opt_end.append(end)
                    
        # the output array
        ar_ana = lh5.Array(np.array([], dtype='float32'))
        
        # check that there are events to process
        if len(idx_sel) > 0:
            # add the input buffers
            pc.add_input_buffer('waveform', wf_2['values'].nda[idx_sel], dtype='float32')
            pc.add_input_buffer('baseline', bl_1          .nda[idx_sel], dtype='uint16' )
            pc.add_input_buffer('beg'     , np.array(opt_beg)          , dtype='int32'  )
            pc.add_input_buffer('end'     , np.array(opt_end)          , dtype='int32'  )

            # add the processors
            pc.add_processor(
                proc.optimize_1pz, 
                'waveform', 
                'baseline', 
                'beg', 
                'end', 
                '400*us', 
                'tau0')

            # get the output buffers
            tb_out = lh5.Table(size=pc._buffer_len, col_dict={
                'tau0': lh5.Array(pc.get_output_buffer('tau0', unit=units.us))})

            # process the events
            pc.execute()
            
            # update the output array
            ar_ana = lh5.Array(tb_out['tau0'].nda)

        # build the output table
        tb_ana = lh5.Table(col_dict={
            f'bin_{i}': ar_ana})

        # write the output to disk
        lock .acquire()
        store.write_object(tb_ana, f'{channels[i_ch]}/tau_2', files_ana[i_fi])
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
def analyze_events(channel, label, bin_start, bin_count, pdf):
    # the extracted information
    time = []
    peak = []

    # loop over the time bins
    for i in range(bin_count):
        # read in the tables
        taus, _ = store.read_object(f'{channel}/{label}/bin_{i}', files_ana)
            
        # the data in the window
        data = taus.nda[
            (taus.nda > window[0]) &\
            (taus.nda < window[1])]
        
        # skip if there is not enough to analyze
        if len(data) == 0:
            continue
        
        # find the bin width
        bin_width = find_bin_width(data)
        
        # check that the bin width is reasonable
        if bin_width > 0 and bin_width < 100:
            # find the binning
            bins = np.arange(*window, bin_width)
        
            # include the results
            time.append(bin_start + (i + 0.5) * t_bin_width)
            peak.append(find_peak(data, bins)[0])

            # plot the results
            fig, ax = plt.subplots()
            ax.hist(data, bins, histtype='stepfilled', alpha=0.5, 
                label=f'{channel}, {label}, {bin_start + (i + 0.5) * t_bin_width:.1f}')
            ax.set_xlabel(r'Time constant [$\mu$s]')
            ax.set_ylabel('Number of events')
            ax.set_xlim(*window)
            ax.legend()
            ax.axvline(peak[-1], ls='--', color='k', lw=1)
            ax.get_yaxis().set_label_coords(-0.1, 0.5)
            pdf.savefig()
            plt.close()
        
    # return the results
    return time, peak

# the stage that analyzes the data
def analyze():
    # the output files
    pdf_vst = pdf.PdfPages(fig_dir + 'p_cal_pzt_vst.pdf')
    pdf_hst = pdf.PdfPages(fig_dir + 'p_cal_pzt_hst.pdf')

    # loop over the channels
    for channel in channels:
        # the extracted information
        time_1, peak_1 = analyze_events(channel, 'tau_1', t_bin_start_1, t_bin_count_1, pdf_hst)
        time_2, peak_2 = analyze_events(channel, 'tau_2', t_bin_start_2, t_bin_count_2, pdf_hst)
        
        # the data to fit
        x = np.concatenate((time_1, time_2))
        y = np.concatenate((peak_1, peak_2))

        # the default result
        popt_1pz = np.average(peak_1)
        popt_2pz = np.zeros(3)
        
        # read in the time constant
        cal_base = db['channels'][channel]['pole_zero']['cal_base']
        
        # the model
        def model(x, b, c):
            return dpzc_func(x, cal_base, b, c)

        try:
            # fit the data
            popt_2pz, _ = optimize.curve_fit(model, x, y, p0=[20, 0.04])
        except:
            # continue processing
            pass

        # plot the result
        fig, ax = plt.subplots()
        ax.errorbar(time_1, peak_1, marker='.', ms=10, ls='None', lw=1, label=f'Event 1 in {channel}')
        ax.errorbar(time_2, peak_2, marker='.', ms=10, ls='None', lw=1, label=f'Event 2 in {channel}')
        ax.set_xlabel(r'Time into waveform [$\mu$s]')
        ax.set_ylabel(r'Time constant [$\mu$s]')
        ax.set_xlim(t_bin_start_1, t_bin_start_2 + (t_bin_count_2 + 1) * t_bin_width)
        ax.legend(loc='upper right')
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        ax.axhline(cal_base, ls='--', color='k', lw=1)
        if np.any(popt_2pz != 0):
            x = np.linspace(*ax.get_xlim(), 1000)
            ax.plot(x, model(x, *popt_2pz), c='k', ls='-', lw=1)
        pdf_vst.savefig()
        plt.close()

        # include the information
        db['channels'][channel]['pole_zero']['fit_tau0'] = popt_1pz
        db['channels'][channel]['pole_zero']['fit_tau1'] = cal_base
        db['channels'][channel]['pole_zero']['fit_tau2'] = popt_2pz[0]
        db['channels'][channel]['pole_zero']['fit_frac'] = popt_2pz[1]

        # status printout
        if np.any(popt_2pz != 0):
            logging.info(f'Found parameters for {channel} of ' +\
                f'{cal_base   :.1f}, '     +\
                f'{popt_2pz[0]:.1f}, and ' +\
                f'{popt_2pz[1]:.4f}')
        else:
            logging.info(f'Found no parameters for {channel}')
            
    # close the output files
    pdf_vst.close()
    pdf_hst.close()

    # update the database
    update_database(db)

# process based on the configuration
if proc_stage in ['', 'produce']: produce()
if proc_stage in ['', 'analyze']: analyze()