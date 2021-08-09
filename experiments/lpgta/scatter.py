# import python modules
from setup import *

# process the events for the specified file and channel
def process_events(args):
    # interpret the arguments
    i_fi, i_ch = args

    # the mask
    cal_dqc = find_mask('cal_dqc', i_fi, channels[i_ch])
    cal_def = find_mask('cal_def', i_fi, channels[i_ch])

    # the events to process
    idx_raw = np.arange(len(cal_dqc))[cal_dqc][1:-1]
    idx_def = np.arange(len(cal_def))[cal_def]
    *_, idx_cal = np.intersect1d(idx_raw, idx_def, return_indices=True)

    # read in the tables
    ts_0, _ = store.read_object(f'{channels[i_ch]}/raw/timestamp'      , files_raw[i_fi], idx=idx_raw-1)
    ps_1, _ = store.read_object(f'{channels[i_ch]}/raw/ts_pps'         , files_raw[i_fi], idx=idx_raw  )
    tk_1, _ = store.read_object(f'{channels[i_ch]}/raw/ts_ticks'       , files_raw[i_fi], idx=idx_raw  )
    ts_1, _ = store.read_object(f'{channels[i_ch]}/raw/timestamp'      , files_raw[i_fi], idx=idx_raw  )
    tp_1, _ = store.read_object(f'{channels[i_ch]}/tp_00'              , files_cal[i_fi], idx=idx_cal  )
    wf_2, _ = store.read_object(f'{channels[i_ch]}/raw/waveform/values', files_raw[i_fi], idx=idx_raw+1)
    ps_2, _ = store.read_object(f'{channels[i_ch]}/raw/ts_pps'         , files_raw[i_fi], idx=idx_raw+1)
    tk_2, _ = store.read_object(f'{channels[i_ch]}/raw/ts_ticks'       , files_raw[i_fi], idx=idx_raw+1)

    # identify the events with the same pps
    idx_sel = np.where(
        (ps_2.nda - ps_1.nda == 0    ) &
        (ts_1.nda - ts_0.nda >  0.005))[0]
    
    # map out the baseline
    bl = np.array([wf_2.nda[i][0] for i in idx_sel])
    dt = np.array([
        tk_2.nda[i] * 0.004 / 1000 -\
        tk_1.nda[i] * 0.004 / 1000 -\
        tp_1.nda[i] / 1000  / 1000 for i in idx_sel])

    # select the events of interest
    bl = bl[(dt > 0) & (dt < 10)]
    dt = dt[(dt > 0) & (dt < 10)]

    # build the output table
    tb_ana = lh5.Table(col_dict={
        'amplitude': lh5.Array(bl),
        'timestamp': lh5.Array(dt)})

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
    pdf_unz = pdf.PdfPages(fig_dir + 'p_cal_sca_unz.pdf')
    pdf_zed = pdf.PdfPages(fig_dir + 'p_cal_sca_zed.pdf')
    pdf_hst = pdf.PdfPages(fig_dir + 'p_cal_sca_hst.pdf')
    
    # loop over the channels
    for channel in channels:
        # read in the tables
        amplitude, _ = store.read_object(f'{channel}/amplitude', files_ana)
        timestamp, _ = store.read_object(f'{channel}/timestamp', files_ana)

        # extract the data
        bl = amplitude.nda
        dt = timestamp.nda
        
        # extract the mode in time ranges
        bin_width = 0.1
        bins = np.arange(0, 10 + bin_width, bin_width)
        x = []
        y = []
        for i in bins:
            bl_sel = bl[(i < dt) & (dt <= i + bin_width)]
            if len(bl_sel) > 0:
                x.append(i + (bin_width / 2))
                y.append(find_peak(bl_sel, np.arange(0, np.power(2, 16), 10))[0])

        # fit the data to an exponential
        popt, _ = optimize.curve_fit(expl_func, x, y, p0=[min(y), max(y) - min(y), 0.4])

        # plot the result
        fig, ax = plt.subplots()
        ax.scatter(dt, bl, s=5, label=channel, rasterized=True)
        ax.set_xlabel('Time into decay [ms]')
        ax.set_ylabel('Waveform value [ADU]')
        ax.set_xlim(0, 10)
        ax.legend()
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        pdf_unz.savefig()
        x = np.arange(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1, 0.01)
        ax.plot(x, expl_func(x, *popt), c='k', lw=1, ls='-')
        ax.axhline(popt[0], ls='--', c='k', lw=1)
        ax.set_ylim(popt[0] - 250, popt[0] + 250)
        pdf_zed.savefig()
        plt.close()
        
        # plot the result
        fig, ax = plt.subplots()
        bins = (int(100 * (fig.get_size_inches()[0] / fig.get_size_inches()[1])), 100)
        *_, image = ax.hist2d(dt, bl, bins=bins, cmap='plasma_r', norm=matplotlib.colors.LogNorm())
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.03, ax.get_position().height])
        cax.get_yaxis().set_label_coords(3, 0.5)
        cbar = plt.colorbar(image, cax=cax)
        cbar.set_label('Number of events')
        ax.set_xlabel('Time into decay [ms]')
        ax.set_ylabel('Waveform value [ADU]')
        ax.set_xlim(0, 10)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        pdf_hst.savefig(bbox_inches='tight')
        plt.close()

    # close the output files
    pdf_unz.close()
    pdf_zed.close()
    pdf_hst.close()

# process based on the configuration
if proc_stage in ['', 'produce']: produce()
if proc_stage in ['', 'analyze']: analyze()
