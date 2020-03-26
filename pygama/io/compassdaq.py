import numpy as np
import pandas as pd
from .io_base import DataTaker

class CAENDT57XX(DataTaker):
    """
    decode CAENDT5725 or CAENDT5730 digitizer data.
    
    Setting the model_name will set the appropriate sample_rate
    Use the input_config function to set certain variables by passing
    a dictionary, this will most importantly assemble the file header used
    by CAEN CoMPASS to label output files.
    """
    def __init__(self, *args, **kwargs):
        self.id = None
        self.model_name = "DT5725" # hack -- can't set the model name in the init
        self.decoder_name = "caen"
        self.file_header = None
        self.adc_bitcount = 14
        self.sample_rates = {"DT5725": 250e6, "DT5730": 500e6}
        self.sample_rate = None
        if self.model_name in self.sample_rates.keys():
            self.sample_rate = self.sample_rates[self.model_name]
        else:
            raise TypeError("Unidentified digitizer type: "+str(model_name))
        self.v_range = 2.0

        self.e_cal = None
        self.e_type = None
        self.int_window = None
        self.parameters = ["TIMETAG", "ENERGY", "E_SHORT", "FLAGS"]

        self.decoded_values = {
            "board": None,
            "channel": None,
            "timestamp": None,
            "energy": None,
            "energy_short": None,
            "flags": None,
            "num_samples": None,
            "waveform": []
        }
        super().__init__(*args, **kwargs)


    def input_config(self, config):
        self.id = config["id"]
        self.v_range = config["v_range"]
        self.e_cal = config["e_cal"]
        self.e_type = config["e_type"]
        self.int_window = config["int_window"]
        self.file_header = "CH_"+str(config["channel"])+"@"+self.model_name+"_"+str(config["id"])+"_Data_"


    def get_event_size(self, t0_file):
        with open(t0_file, "rb") as file:
            if self.e_type == "uncalibrated":
                first_event = file.read(24)
                [num_samples] = np.frombuffer(first_event[20:24], dtype=np.uint16)
                return 24 + 2*num_samples
            elif self.e_type == "calibrated":
                first_event = file.read(30)
                [num_samples] = np.frombuffer(first_event[26:30], dtype=np.uint32)
                return 30 + 2 * num_samples  # number of bytes / 2
            else:
                raise TypeError("Invalid e_type! Valid e_type's: uncalibrated, calibrated")


    def get_event(self, event_data_bytes):
        self.decoded_values["board"] = np.frombuffer(event_data_bytes[0:2], dtype=np.uint16)[0]
        self.decoded_values["channel"] = np.frombuffer(event_data_bytes[2:4], dtype=np.uint16)[0]
        self.decoded_values["timestamp"] = np.frombuffer(event_data_bytes[4:12], dtype=np.uint64)[0]
        if self.e_type == "uncalibrated":
            self.decoded_values["energy"] = np.frombuffer(event_data_bytes[12:14], dtype=np.uint16)[0]
            self.decoded_values["energy_short"] = np.frombuffer(event_data_bytes[14:16], dtype=np.uint16)[0]
            self.decoded_values["flags"] = np.frombuffer(event_data_bytes[16:20], np.uint32)[0]
            self.decoded_values["num_samples"] = np.frombuffer(event_data_bytes[20:24], dtype=np.uint32)[0]
            self.decoded_values["waveform"] = np.frombuffer(event_data_bytes[24:], dtype=np.uint16)
        elif self.e_type == "calibrated":
            self.decoded_values["energy"] = np.frombuffer(event_data_bytes[12:20], dtype=np.float64)[0]
            self.decoded_values["energy_short"] = np.frombuffer(event_data_bytes[20:22], dtype=np.uint16)[0]
            self.decoded_values["flags"] = np.frombuffer(event_data_bytes[22:26], np.uint32)[0]
            self.decoded_values["num_samples"] = np.frombuffer(event_data_bytes[26:30], dtype=np.uint32)[0]
            self.decoded_values["waveform"] = np.frombuffer(event_data_bytes[30:], dtype=np.uint16)
        else:
            raise TypeError("Invalid e_type! Valid e_type's: uncalibrated, calibrated")
        return self.assemble_data_row()


    def assemble_data_row(self):
        timestamp = self.decoded_values["timestamp"]
        energy = self.decoded_values["energy"]
        energy_short = self.decoded_values["energy_short"]
        flags = self.decoded_values["flags"]
        waveform = self.decoded_values["waveform"]
        return [timestamp, energy, energy_short, flags], waveform


    def create_dataframe(self, array):
        waveform_labels = [str(item) for item in list(range(self.decoded_values["num_samples"]-1))]
        column_labels = self.parameters + waveform_labels
        dataframe = pd.DataFrame(data=array, columns=column_labels, dtype=float)
        return dataframe



def process_compass(daq_filename, raw_filename, digitizer, output_dir=None):
    """
    Takes an input .bin file name as daq_filename from CAEN CoMPASS and outputs raw_filename
    daq_filename: input file name, string type
    raw_filename: output file name, string type
    digitizer: CAEN digitizer, Digitizer type
    options are uncalibrated or calibrated.  Select the one that was outputted by CoMPASS, string type
    output_dir: path to output directory string type
    """
    start = time.time()
    f_in = open(daq_filename.encode("utf-8"), "rb")
    if f_in is None:
        raise LookupError("Couldn't find the file %s" % daq_filename)
    SEEK_END = 2
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)
    file_size_MB = file_size / 1e6
    print("Total file size: {:.3f} MB".format(file_size_MB))

    # ------------- scan over raw data starts here ----------------

    print("Beginning Tier 0 processing ...")

    event_rows = []
    waveform_rows = []
    event_size = digitizer.get_event_size(daq_filename)
    with open(daq_filename, "rb") as metadata_file:
        event_data_bytes = metadata_file.read(event_size)
        while event_data_bytes != b"":
            event, waveform = digitizer.get_event(event_data_bytes)
            event_rows.append(event)
            waveform_rows.append(waveform)
            event_data_bytes = metadata_file.read(event_size)
    all_data = np.concatenate((event_rows, waveform_rows), axis=1)
    output_dataframe = digitizer.create_dataframe(all_data)
    f_in.close()

    output_dataframe.to_hdf(path_or_buf=output_dir+"/"+raw_filename, key="dataset", mode="w", table=True)
    print("Wrote Tier 1 File:\n  {}\nFILE INFO:".format(raw_filename))

    # --------- summary -------------

    with pd.HDFStore(raw_filename, "r") as store:
        print(store.keys())


