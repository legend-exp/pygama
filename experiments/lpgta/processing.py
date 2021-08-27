# import python modules
from setup import *

# the outputs to save to disk
outputs = []

# select the outputs
if proc_label in ['cal',        'dsp']: outputs.append('sat_lo'   )
if proc_label in ['cal',        'dsp']: outputs.append('sat_hi'   )
if proc_label in [              'dsp']: outputs.append('bl_std'   )
if proc_label in ['cal',        'dsp']: outputs.append('pftp_lo'  )
if proc_label in ['cal',        'dsp']: outputs.append('pftp_hi'  )
if proc_label in [              'dsp']: outputs.append('tf_avg'   )
if proc_label in [              'dsp']: outputs.append('tf_std'   )
if proc_label in ['cal', 'opt', 'dsp']: outputs.append('tftp_tp'  )
if proc_label in [              'dsp']: outputs.append('tftp_31'  )
if proc_label in ['cal',        'dsp']: outputs.append('tp_00'    )
if proc_label in [       'opt', 'dsp']: outputs.append('drift_t'  )
if proc_label in [       'opt', 'dsp']: outputs.append('dcr_raw'  )
if proc_label in [       'opt', 'dsp']: outputs.append('q_drift'  )
if proc_label in ['cal',        'dsp']: outputs.append('tp_00_dqc')

# set the outputs
processors['outputs'] = outputs

# check the processing label
if proc_label == 'opt':
    # the parameter grid
    grid = dspo.ParGrid()

    # add the parameter grid dimensions
    grid.add_dimension('wf_tf', [1, 2], trap, companions=[('tftp_tp', 1, pick)])
    
# process the events for the specified file and channel
def process_events(args):
    # interpret the arguments
    i_fi, i_ch = args
    
    # the mask
    idx = None
    
    # select the events
    if proc_label == 'cal': idx = find_mask('cal_def', i_fi, channels[i_ch])
    if proc_label == 'opt': idx = find_mask('cal_dqc', i_fi, channels[i_ch])
    
    # the events to process
    if idx is not None:
        idx = np.arange(len(idx))[idx]
    
    # read in the tables
    waveform, _ = store.read_object(f'{channels[i_ch]}/raw/waveform', files_raw[i_fi], idx=idx)

    # build the table for processing
    tb_data = lh5.Table(col_dict={'waveform': waveform})
    
    # copy the dictionary
    pcs = copy.deepcopy(processors)

    # check the number of poles
    if pole_count == 1:
        # this channel's parameters
        tau_bl = db['channels'][channels[i_ch]]['pole_zero']['cal_base']
        tau_pz = db['channels'][channels[i_ch]]['pole_zero']['cal_tail']

        # update the processors
        pcs['processors']['wf_bl'    ]['args'][2] = f'{tau_bl}*us'
        pcs['processors']['wf_pz'    ]['args'][1] = f'{tau_pz}*us'
        pcs['processors']['wf_pz_dqc']['args'][1] = f'{tau_pz}*us'

    # check the number of poles
    if pole_count == 2:
        # this channel's parameters
        tau1 = db['channels'][channels[i_ch]]['pole_zero']['avg_tau1']
        tau2 = db['channels'][channels[i_ch]]['pole_zero']['avg_tau2']
        frac = db['channels'][channels[i_ch]]['pole_zero']['avg_frac']

        # the new argument list
        args     = ['wf_bl'    , f'{tau1}*us', f'{tau2}*us', f'{frac}', 'wf_pz'    ]
        args_dqc = ['wf_bl_dqc', f'{tau1}*us', f'{tau2}*us', f'{frac}', 'wf_pz_dqc']

        # update the processors
        pcs['processors']['wf_bl'    ]['args'    ][2] = f'{tau2}*us'
        pcs['processors']['wf_pz'    ]['function']    = 'double_pole_zero'
        pcs['processors']['wf_pz'    ]['args'    ]    = args
        pcs['processors']['wf_pz_dqc']['function']    = 'double_pole_zero'
        pcs['processors']['wf_pz_dqc']['args'    ]    = args_dqc

    # check the processing label
    if proc_label == 'dsp':
        # this channel's parameters
        rise = db['channels'][channels[i_ch]]['optimization']['opt_par'][0]
        flat = db['channels'][channels[i_ch]]['optimization']['opt_par'][1]

        # update the processors
        pcs['processors']['wf_tf'                         ]['args'][1] = f'{rise}'
        pcs['processors']['wf_tf'                         ]['args'][2] = f'{flat}'
        pcs['processors']['tf_avg, tf_std, tf_ftm, tf_ftb']['args'][0] = f'wf_tf[{1702+rise}:{1702+rise+flat}]'
        pcs['processors']['tftp_tp'                       ]['args'][1] = f'tp_00+{rise+flat}-0.496*us'

    # check the processing label
    if proc_label == 'opt':
        # the index
        idx = grid.get_zero_indices()
        
        # loop over the grid
        while True:
            # update the parameters
            grid.set_dsp_pars(pcs, idx)
            
            # process the events
            tb_out = dspo.run_one_dsp(tb_data, pcs)
            
            # the datagroup label
            label = '_'.join(str(i) for i in idx)
            
            # write the output to disk
            lock .acquire()
            store.write_object(tb_out, label, files_arg[i_fi], group=channels[i_ch])
            lock .release()
            
            # update the index
            if not grid.iterate_indices(idx):
                break
    else:
        # build the processing chain
        pc, tb_out = bpc.build_processing_chain(tb_data, pcs, verbosity=0)

        # process the events
        pc.execute()

        # write the output to disk
        lock .acquire()
        store.write_object(tb_out, channels[i_ch], files_arg[i_fi])
        lock .release()

    # status printout
    logging.info(f'Finished processing {channels[i_ch]} in {os.path.basename(files_raw[i_fi])}')

# loop over the files
for i_fi in range(len(files_raw)):
    # skip if not the selected file
    if proc_index and i_fi not in proc_index:
        continue
        
    # delete the output file if it exists
    if os.path.isfile(files_arg[i_fi]):
        os.remove(files_arg[i_fi])
    
    # the argument list
    args = [[i_fi, i_ch] for i_ch in range(len(channels))]
    
    # launch the parallel processes
    with mp.Pool(proc_count) as p:
        p.map(process_events, args)
