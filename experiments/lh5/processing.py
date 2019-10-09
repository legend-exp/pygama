#!/usr/bin/env python3
import json
from abc import ABC
from pprint import pprint

from pygama import DataSet
import fcutils
from flashcam import FlashCamDecoder


def main():
    """
    testing Yoann's Tier 0 (daq_to_raw) FlashCam parser.
    Requires 'fcutils' python module.  Uses local copies of the Digitizer
    and DataTaker class s/t it's easier to see how to remove the pandas stuff
    and replace with the desired .lh5 format.
    """
    d2r_wrapper()
    
    
def d2r_wrapper():
    """
    this is the high-level part of the code, something that a user might
    write for processing with a specific config file.
    """
    # move these to cmd line args
    overwrite = True
    test = False

    ds = DataSet(run=0, config="config.json")
    ds.daq_to_raw(overwrite, test)
    
    

def daq_to_raw(t0_file, run, config):
    """
    formerly "ProcessRaw", aka the loop over the raw data file.
    no run info in FC header yet
    """
    n_event_limit = 5e4

    io = fcutils.fcio(t0_file)
    
    print(f"Run number: {run}")
    
    decoders = [FlashCamDecoder(),]
    decoder_names = ['FlashCam']

    # #Currently no decoder name provide by the FC file header
    # #Set by hand - get_decoder()
    # if decoders is None:
    #   decoders = []
    #   decoders.append(FlashCamDecoder())
    #   decoder_names = []
    #   decoder_names.append('FlashCam')
    # 
    # final_decoder_list = list(set(decoder_names).intersection(used_decoder_names))
    # decoder_to_id = {d.decoder_name: d for d in decoders}
    # 
    # # pass in specific decoder options (windowing, multisampling, etc.)
    # for d in decoders:
    #   d.set_config(config)
    # 
    # if os.path.isfile(t1_file):
    #   if overwrite:
    #     print("Overwriting existing file...")
    #     os.remove(t1_file)
    #   else:
    #     print("File already exists, continuing ...")
    #     return
    # 
    # packet_id = 0 # number of events decoded
    # unrecognized_data_ids = []
    # 
    # while io.next_event() and packet_id<n_max:
    #   packet_id += 1
    #   if packet_id % 10000 == 0:
    #     print(packet_id,io.eventtime)
    #   #if verbose and packet_id % 1000 == 0:
    #   #update_progress(float(io.telid) / file_size)
    # 
    #   # write periodically to the output file instead of writing all at once
    #   if packet_id % n_event_limit == 0:
    #     for d in decoders:
    #       d.to_file(t1_file, verbose=True)
    # 
    #   # sends data to the pandas dataframe
    #   # specific FlashCam formatting
    #   for d in decoders:
    #     if d.decoder_name == 'FlashCam':
    #       d.decode_event(io, packet_id)
    #     else:
    #       print("ERROR: Specific FlashCam event decoder provided to ",d.decoder_name)
    #       sys.exit()
    # 
    # ##########################################
    # # End of FlashCam data specific decoding #
    # ##########################################
    # 
    # # final write to file
    # for d in decoders:
    #   if verbose:
    #     print("daq_to_raw - write in decoder ",d.decoder_name)
    #   d.to_file(t1_file, verbose=True)
    # 
    # if verbose:
    #   update_progress(1)
    # 
    # if len(unrecognized_data_ids) > 0:
    #   print("WARNING, Found the following unknown data IDs:")
    #   for id in unrecognized_data_ids:
    #     print(" {}".format(id))
    #   print("hopefully they weren't important!\n")
    # 
    # # --------- summary ------------
    # 
    # print("Wrote: Tier 1 File:\n    {}\nFILE INFO:".format(t1_file))
    # with pd.HDFStore(t1_file,'r') as store:
    #   print(store.keys())
    #   # print(store.info())
    # 
    # statinfo = os.stat(t1_file)
    # print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
    # elapsed = time.time() - start
    # print("Time elapsed: {:.2f} sec".format(elapsed))
    # print("Done.\n")

    
    
    
if __name__=="__main__":
    main()