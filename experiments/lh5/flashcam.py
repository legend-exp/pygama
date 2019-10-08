"""
functions needed to convert FlashCam data to lh5.  Keeping them separate
from the main pygama functions for now s/t it's easier to see what is essential,
for converting to the .lh5 format and doing some simplification of the core code.
"""
from abc import ABC

class DataLoader(ABC):
    """
    when you initialize this from a derived class with 'super',
    you need to have already declared:
        self.digitizer_type, self.class_name, and self.decoded_values.
    """
    def __init__(self, config=None):
        """
        """
        self.total_count = 0
        self.garbage_count = 0 # never leave any data behind
        self.garbage_values = {key:[] for key in self.decoded_values}
        self.h5_format = "lh5"

        with open(config) as f:
            self.config = json.load(f)
        
        
    def format_data(self, vals, is_garbage=False):
        """
        for every event in the raw DAQ data, we send any variable with a name in
        "self.decoded_values" to the output with: self.format_data(locals())
        """
        self.total_count += 1
        if is_garbage:
            self.garbage_count += 1

        for key in vals:
            if key is not "self" and key in self.decoded_values:
                if is_garbage:
                    self.garbage_values[key].append(vals[key])
                else:
                    # Need to make this fix, otherwise waveform array 
                    # overwritten at each event
                    if type(vals[key])==np.ndarray:
                        self.decoded_values[key].append(vals[key].copy())
                    else:
                        self.decoded_values[key].append(vals[key])


    def create_df(self, get_garbage=False):
        """
        Base dataframe creation method. Classes inheriting from DataLoader 
        (like digitizers or pollers) can overload this if necessary for more 
        complicated use cases. Try to avoid pickling to 'object' types.
        """
        if not get_garbage:
            raw_values = self.decoded_values
        else:
            raw_values = self.garbage_values
            df = pd.DataFrame.from_dict(self.garbage_values).infer_objects()
            return df

        print("STOP.  LH5 and listen.")
        exit()

        # # 'flatten' the values (each wf sample gets a column, h5type = 'table')
        # new_cols = []
        # for col in sorted(raw_values.keys()):
        # 
        #     # unzip waveforms into an ndarray
        #     if col == "waveform":
        # 
        #         wf_lens = [len(wf) for wf in raw_values["waveform"]]
        #         wf_dims = set(wf_lens)
        # 
        #         # error checking
        #         if len(wf_dims) > 1:
        #             print("ERROR, wfs must all be same size, not ",wf_dims)
        #             pytables_error = True
        #             break
        #         if any(d > self.pytables_col_limit for d in wf_dims):
        #             print("ERROR, exceeded pytables_col_limit")
        #             print(wf_dims)
        #             self.h5_format = "fixed"
        #             pytables_error = True
        #             break
        #             # sys.exit()
        # 
        #         # create the ndarray and a new dataframe
        #         if not pytables_error and len(raw_values["waveform"]) > 0:
        #             wfs = np.vstack(raw_values["waveform"])
        #             new_cols.append(pd.DataFrame(wfs, dtype='uint16'))
        # 
        #     # everything else is single-valued
        #     else:
        #         if not pytables_error:
        #             vals = pd.Series(raw_values[col], name=col)
        #             new_cols.append(vals)
        # 
        # # this is our flattened output (can be chunked with hdf5)
        # df = pd.concat(new_cols, axis=1)

        if len(df) == 0:
            return None
        else:
            return df


    def to_file(self, file_name, verbose=False, write_option="a"):
        """ 
        save primary data, garbage data, and metadata to hdf5 
        """
        if verbose:
            print("Writing {}".format(self.digitizer_type))

        # we can only fast-append to the file if it's in table format
        append = True if self.h5_format == "table" else False

        hdf_kwargs = {"mode":write_option,
                      "format":self.h5_format,
                      "append":append,
                      "complib":"blosc:snappy", # idk, sounds fast
                      "complevel":2, # compresses raw by ~0.5
                      "data_columns":["ievt"]} # cols for hdf5 fast file indexing

        def check_and_append(file_name, key, df_data):
            with pd.HDFStore(file_name, 'r') as store:
                try:
                    extant_df = store.get(key)
                    df_data = pd.concat([extant_df, df_data]).reset_index(drop=True)
                    if verbose:
                        print(key, df_data.shape)
                except KeyError:
                    pass
            return df_data

        #  ------------- save primary data -------------
        df_data = self.create_df()

        if not append:
            # make a copy of the df already in the file and manually append
            df_data = check_and_append(file_name, self.digitizer_type, df_data)

        # write to hdf5 file
        df_data.to_hdf(file_name, key=self.digitizer_type, **hdf_kwargs)

        # ------------- save metadata -------------
        if self.config is not None:

            if self.class_name == self.digitizer_type:
                raise ValueError(
                    "Can't write config if it has the same "
                    "ORCA decoder and class names, you would overwrite the data."
                    "Class: {}\nDecoder: {}".format(self.class_name,
                                                    self.digitizer_type))
                sys.exit()

            if not append:
                self.config = check_and_append(file_name,
                                                    self.class_name,
                                                    self.config).infer_objects()

            self.config.to_hdf(file_name, key=self.class_name,
                                    mode='a',
                                    format="fixed")

        # # ------------- save garbage data -------------
        # if self.garbage_count > 0:
        #     print("Saving garbage: {} of {} total events"
        #           .format(self.garbage_count, self.total_count))
        # 
        #     df_garbage = self.create_df(get_garbage=True)
        # 
        #     # garbage is always in fixed format since the wfs may be different lengths
        #     hdf_kwargs["format"] = "fixed"
        #     hdf_kwargs["append"] = False
        # 
        #     df_garbage = check_and_append(file_name,
        #                                   self.digitizer_type+"_Garbage",
        #                                   df_garbage)#.infer_objects()
        # 
        #     df_garbage.to_hdf(file_name, key=self.digitizer_type+"_Garbage",
        #                       **hdf_kwargs)

        # finally, clear out existing data (relieve memory pressure)
        self.clear_data()


    def clear_data(self):
        """ 
        clear out standard objects when we do a write to file 
        """
        self.total_count = 0
        self.garbage_count = 0
        self.decoded_values = {key:[] for key in self.decoded_values}
        self.garbage_values = {key:[] for key in self.garbage_values}


# === DIGITIZER SPECIFIC STUFF =================================================
# class Digitizer(DataLoader):
#     """ handle any data loader which contains waveform data """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
# 
#     def apply_settings(self, settings):
#         """ apply user settings specific to this card and run """
# 
#         if settings["digitizer"] == self.digitizer_type:
# 
#             self.window = False
#             sk = settings.keys()
#             if "window" in sk:
#                 self.window = True
#                 self.win_type = settings["window"]
#             if "n_samp" in sk:
#                 self.n_samp = settings["n_samp"]
#             if "n_blsamp" in sk:
#                 self.n_blsamp = settings["n_blsamp"]


class FlashCamDecoder(DataLoader):
  """ 
  load data from FlashCam using Yoann's `pyflashcam` code.
  """
  def __init__(self, *args, **kwargs):

    self.digitizer_type = 'FlashCam'
    self.class_name = 'FlashCam'

    # store an entry for every event
    self.decoded_values = {"packet_id":[], "ievt":[], "energy":[], "bl":[], 
                           "bl0":[], "bl1" :[], "timestamp":[], "channel":[],
                           "waveform":[]}
    super().__init__(*args, **kwargs) # also initializes the garbage df
    
    self.event_header_length = 1
    self.sample_period = 16 # ns
    self.n_blsamp = 500
    self.ievt = 0
    self.ievtg = 0
    self.h5_format = "lh5" # <---this is new  
          
  def decode_event(self, io, packet_id, verbose=False):
      """
      see README for the 32-bit data word diagram
      """
      crate = 0
      card = 0
      channel = 0
      energy = 0 # currently not stored but can be in the future?
      wf_length_32 = io.nsamples
      timestamp = io.eventtime
      bl = float(io.average_prebaselines)
      bl0 = int(io.prebaselines0)
      bl1 = int(io.prebaselines1)

      # get raw wf array
      waveform = io.traces
      waveform.astype(float)

      # set the event number (searchable HDF5 column)
      ievt = self.ievt
      self.ievt += 1

      # send any variable with a name in "decoded_values" to the pandas output
      self.format_data(locals())  
