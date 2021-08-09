# import python modules
from setup import *

# the output files
pdf_mts = pdf.PdfPages(fig_dir + 'p_cal_cal_mts.pdf')
pdf_unc = pdf.PdfPages(fig_dir + 'p_cal_cal_uen.pdf')
pdf_cal = pdf.PdfPages(fig_dir + 'p_cal_cal_cen.pdf')

# status printout
logging.info(f'Found {len(files_raw)} files to process')

# the channels shared across the files
channels = []

# loop over the files
for file_raw in files_raw:
    # open this file
    with h5py.File(file_raw, 'r') as f:
        # find the shared channels
        channels = list(f.keys()) if len(channels) == 0 else list(set(channels) & set(f.keys()))

# check that there are channels to process
if len(channels) > 0:
    # status printout
    logging.info(f'Found {len(channels)} shared channels')
else:
    # exit if no channels to process
    sys.exit('Found no channels to process')
    
# the channels to process
channels = np.sort(channels)

# the dictionary labels for each channel
labels = [
    'calibration',
    'optimization',
    'pole_zero',
    'resolution']

# create a new dictionary
db['channels'] = {}

# loop over the channels
for channel in channels:
    # create a new dictionary
    db['channels'][channel] = {}
    
    # loop over the dictionary labels
    for label in labels:
        # create a new dictionary
        db['channels'][channel][label] = {}

# the event count per channel
event_count = [[] for channel in channels]

# loop over the files
for i_fi, file_raw in enumerate(files_raw):
    # open this file
    with h5py.File(file_raw, 'r') as f:
        # initialize the event information
        if i_fi == 0:
            tick_width = f[f'{channels[0]}/raw/waveform/dt'    ]      [0]
            tick_count = f[f'{channels[0]}/raw/waveform/values'].shape[1]
        
        # loop over the channels
        for i_ch, channel in enumerate(channels):
            # exit if they are not the same for all events
            if not np.all(np.diff(f[f'{channel}/raw/waveform/dt']) == 0)   or\
               f[f'{channel}/raw/waveform/dt'    ]      [0] != tick_width  or\
               f[f'{channel}/raw/waveform/values'].shape[1] != tick_count:
                sys.exit('Found waveforms with different configurations')
             
            # include the count of events
            event_count[i_ch].append(f[f'{channel}/raw/waveform/values'].shape[0])

# include the information
db['tick_width'] = tick_width / 1000
db['tick_count'] = tick_count

# loop over the channels
for i_ch, channel in enumerate(channels):
    # include the information
    db['channels'][channel]['event_count'] = event_count[i_ch]
        
# status printout
logging.info(f'Found a sampling period of {tick_width} ns'  )
logging.info(f'Found a waveform length of {tick_count} c.t.')

# list of the pps periods
maxticks = []

# loop over the files
for file_raw in files_raw:
    # open this file
    with h5py.File(file_raw, 'r') as f:
        # the channel with the most pps signals
        key_max = channels[0]
        idx_max = np.where(np.diff(f[f'{key_max}/raw/ts_pps']) != 0)[0]
        
        # loop over the remaining channels
        for i in range(1, len(channels)):
            # the indices of the pps signals
            idx = np.where(np.diff(f[f'{channels[i]}/raw/ts_pps']) != 0)[0]
            
            # update the channel information
            if len(idx) > len(idx_max):
                key_max = channels[i]
                idx_max = idx

        # loop over the pps signals
        for i in idx:
            # append the pps period if in range
            if i < f[f'{key_max}/raw/ts_maxticks'].size - 1:
                maxticks.append(int(f[f'{key_max}/raw/ts_maxticks'][i+1]))

# the period deviation
dev = np.array(maxticks) - 250000000

# histogram the period deviation
fig, ax = plt.subplots()
_, b, _ = ax.hist(dev, 100, histtype='stepfilled', log=True)
ax.set_xlabel('Period deviation [c.t.]')
ax.set_ylabel(f'N / {b[1] - b[0]:.0f} c.t.')
xlim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
ax.set_xlim(-xlim, xlim)
ax.get_yaxis().set_label_coords(-0.1, 0.5)
pdf_mts.savefig()
plt.close()

# the period-deviation threshold
pdt = 100000

# status printout
logging.info(f'Found {len(dev[dev < -pdt]) / len(dev) * 100:.1f} % of signals came early')
logging.info(f'Found {len(dev[dev >  pdt]) / len(dev) * 100:.1f} % of signals came late' )

# the clock coupling
clock_coupling = 'loose' if len(dev[(dev < -pdt) | (dev > pdt)]) > 0 else 'tight'

# include the information
db['clock_coupling'] = clock_coupling

# status printout
logging.info(f'Found the clock to be {clock_coupling}ly coupled')

# the window of interest
window_size = 100
window = [
    db['lines']['tlfep'] - window_size,
    db['lines']['tlfep'] + window_size]

# loop over the channels
for channel in channels:
    # read in the energies
    energy, _ = store.read_object(f'{channel}/raw/energy', files_raw)
    
    # the default peak window
    y_min = 15000
    y_max = 35000
    
    # find the end of the spectrum
    b = np.arange(min(energy.nda), max(energy.nda), find_bin_width(energy.nda) / 10)
    n = numba_histogram(energy.nda, b)
    for i in range(np.argmax(n > 0), len(n)):
        if n[i] == 0:
            y_min = b[int(i/2)]
            y_max = b[i]
            break
    
    # plot the distribution
    fig, ax = plt.subplots()
    bins = np.arange(0, np.power(2, 16), 100)
    c = 'tab:red' if y_max - y_min < 100 else 'tab:blue'
    ax.hist(energy.nda, bins, histtype='stepfilled', color=c, alpha=0.5, log=True, label=channel)
    ax.set_xlabel('Energy [ADU]')
    ax.set_ylabel('Number of events')
    ax.set_xlim(bins[0], bins[-1])
    ax.legend(loc='upper right')
    ax.axvspan(y_min, y_max, color='k', alpha=0.1)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_unc.savefig()
    plt.close()
    
    # check if the distribution is unreasonable
    if y_max - y_min < 100:
        # remove this channel
        db['channels'].pop(channel)
        
        # status printout
        logging.info(f'Found no calibration line for {channel}')
        
        # skip to the next channel
        continue

    # find the calibration peak
    peak, _ = find_peak(energy.nda, np.arange(y_min, y_max, 2))
    
    # the calibration factor
    adu_to_kev = db['lines']['tlfep'] / peak

    # plot the distribution
    fig, ax = plt.subplots()
    bins = np.arange(0, 3500, 100 * adu_to_kev)
    ax.hist(energy.nda * adu_to_kev, bins, histtype='stepfilled', alpha=0.5, log=True, label=channel)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Number of events')
    ax.set_xlim(bins[0], bins[-1])
    ax.legend(loc='upper right')
    ax.axvline(db['lines']['tlfep'], ls='--', c='k', lw=1)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    pdf_cal.savefig()
    plt.close()
    
    # select events about the calibration peak
    idx = np.where(
        (window[0] / adu_to_kev < energy.nda) & 
        (window[1] / adu_to_kev > energy.nda))[0]

    # write the index mask to disk
    wr_mask(len(energy.nda), idx, f'{channel}_cal_def')
    
    # include the information
    db['channels'][channel]['calibration']['raw_def'] = peak

    # status printout
    logging.info(f'Found calibration line for {channel} at {peak:.1f} ADU')

# close the output files
pdf_mts.close()
pdf_unc.close()
pdf_cal.close()

# update the database
update_database(db)
