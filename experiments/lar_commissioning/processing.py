# Using the code David Sweigart developed to multi-thread the Germanium analysis 
# import python modules
from setup import *


# set the channels you want to analyze sipm data from
channels = [18, 19, 20, 21]

# set the files you want to analyze
files_raw = glob.glob("/global/project/projectdirs/legend/data/lngs/pgt/raw/spms/LPGTA_r0030*_phys_*.lh5")

# the lock to synchronize the processes
lock = mp.Lock()

# create a store object
store = lh5.Store()

# read in the processors
with open('processors.json') as f:
    processors = json.load(f, object_pairs_hook=collections.OrderedDict)

# process the events for the specified file and channel
def process_events(args):
    # interpret the arguments
    i_fi, i_ch, idx = args
    
    # read in the tables
    waveform, _ = store.read_object(f'spms/raw/waveform', files_raw[i_fi], idx=idx)

    # build the table for processing
    tb_data = lh5.Table(col_dict={'waveform': waveform})
    
    # build the processing chain
    pc, tb_out = bpc.build_processing_chain(tb_data, processors, verbosity=0)

    # process the events
    pc.execute()
    
    

    # write the output to disk
    lock .acquire()
    store.write_object(tb_out, f'{channels[i_ch]}', f'sipm_outputs.lh5')
    lock .release()
    

# loop over the files
for i_fi in range(len(files_raw)):
        
    channel, _ = store.read_object(f'spms/raw/channel', files_raw[i_fi])

    
    # the argument list
    args = [[
        i_fi, 
        i_ch,
        np.where(channel.nda == channels[i_ch])[0]] for i_ch in range(len(channels))]
    
    # launch the parallel processes
    with mp.Pool(len(channels)) as p:
        p.map(process_events, args)
        