# import python modules
from setup import *

# the binning information
bin_min =  0
bin_max = 50
bin_wid =  0.01

# the binning
bins = np.arange(bin_min, bin_max, bin_wid)

# process the events for the specified file and channel
def process_events(args):
    # interpret the arguments
    i_fi, i_ch = args

    # the mask
    all_dqc = find_mask('all_dqc', i_fi, channels[i_ch])
    
    # the events to process
    idx = np.arange(len(all_dqc))[all_dqc]
    
    # read in the tables
    bl_std, _ = store.read_object(f'{channels[i_ch]}/bl_std', files_dsp[i_fi], idx=idx)
    
    # histogram the energy
    n = numba_histogram(bl_std.nda, bins)
    
    # build the output table
    tb_ana = lh5.Table(col_dict={
        'n': lh5.Array(np.array([n]))})

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
    pdf_nls = pdf.PdfPages(fig_dir + 'p_ana_noi_hst.pdf')
    
    # loop over the channels
    for channel in channels:
        # read in the tables
        n_list, _ = store.read_object(f'{channel}/n', files_ana)
        
        # the result
        n = np.zeros(len(bins) - 1)
        b = bins
        
        # calculate the total
        for l in n_list.nda:
            n += l
        
        # calculate the cumulative sum
        cs = np.cumsum(n)
        
        # calculate the percentiles
        q1 = b[np.argmax(cs > 0.25 * cs[-1])]
        q3 = b[np.argmax(cs > 0.75 * cs[-1])]

        # calculate the bin width
        width = 10 * (q3 - q1) / np.cbrt(cs[-1])
        
        # the number of bins to combine together
        factor = math.floor(width / (b[1] - b[0]))
                
        # check whether to re-bin the data
        if factor > 1:
            # the new result
            new_n = []
            new_b = []
            
            # the number of bins
            length = math.floor(len(b) / factor)
            
            # include the bin
            for i in range(length):
                new_n.append(np.sum(n[i*factor:(i+1)*factor]))
                new_b.append(bin_min + i * (bin_wid * factor))
            
            # include the bin
            new_b.append(new_b[-1] + bin_wid * factor)
            
            # update the result
            n = np.array(new_n)
            b = np.array(new_b)

        # find the largest bin
        i_max = np.argmax(n)
        
        # find the bin over threshold
        b_lo = b[i_max - np.argmax(n[:i_max+1][::-1] < n[i_max] / 2) + 1]
        b_hi = b[i_max + np.argmax(n[i_max:  ]       < n[i_max] / 2) - 1]

        # calculate the window information
        wid = (b_hi - b_lo) / 2
        
        # calculate the bin centers
        c = (b[:-1] + b[1:]) / 2

        # calculate the fit range
        i_lo = i_max - np.argmax(c[:i_max+1][::-1] < c[i_max] - 1 * wid) + 1
        i_hi = i_max + np.argmax(c[i_max:  ]       > c[i_max] + 1 * wid) - 1

        # the data to be fit
        x = c[i_lo:i_hi]
        y = n[i_lo:i_hi]

        # the default fit result
        popt = np.zeros(3)

        # check that there are enough data to fit
        if len(x) >= 4:
            try:
                # fit the data
                popt, _ = optimize.curve_fit(gaus_func, x, y, sigma=np.sqrt(y), 
                    p0=[n[i_max], c[i_max], 0.1])
            except:
                print('fit failed')
                # continue processing
                pass

        # plot the results
        fig, ax = plt.subplots()
        ax.hist(b[:-1], b, weights=n, histtype='stepfilled', alpha=0.5, log=True, label=channel)
        ax.set_xlabel('Standard deviation [ADU]')
        ax.set_ylabel('Number of events')
        ax.set_xlim(b[np.argmax(n)] - 10, b[np.argmax(n)] + 10)
        ax.legend()
        if np.all(popt != 0):
            xf = np.linspace(c[i_lo], c[i_hi], 1000)
            ax.plot(xf, gaus_func(xf, *popt), c='k', ls='-', lw=1)
            ax.axvline(popt[1], ls='--', color='k', lw=1)
        else:
            ax.axvspan(c[i_lo], c[i_hi], color='k', alpha=0.1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        pdf_nls.savefig()
        plt.close()

        # include the information
        db['channels'][channel]['noise_level'] = popt[1]

        # status printout
        if np.all(popt != 0):
            logging.info(f'Found noise level for {channel} of {popt[1]:.1f} ADU')
        else:
            logging.info(f'Found no noise level for {channel}')    

    # close the output files
    pdf_nls.close()

    # update the database
    update_database(db)

# process based on the configuration
if proc_stage in ['', 'produce']: produce()
if proc_stage in ['', 'analyze']: analyze()
