"""another docstring, as required by schpinxx"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import datetime as dt

from pygama.processing import process_tier_0
import pygama.decoders as dl
from pygama.calibration import *

def main():
    runNumber = 35366
    n_max = np.inf

    process(runNumber, n_max=500)
    # plot_baselines("t1_run{}.h5".format(runNumber))
    # plot_waveforms("t1_run{}.h5".format(runNumber), num_waveforms=50)

    plt.show()

def plot_baselines(file_name, draw_non_detectors=False):
    df_preamp =  pd.read_hdf(file_name, key="ORMJDPreAmpDecoderForAdc")

    f1 = plt.figure(figsize=(14,8))

    for an_id in df_preamp.device_id.unique():
        for a_ch in df_preamp.channel.unique():
            ts = df_preamp.timestamp[df_preamp.device_id == an_id][df_preamp.channel == a_ch].tolist()      # timestamp
            ts_f = [ dt.datetime.fromtimestamp(t) for t in ts]                                              # formatted ts object
            v = df_preamp.adc[df_preamp.device_id == an_id][df_preamp.channel == a_ch].tolist()             # voltage reading

            if df_preamp.enabled[df_preamp.device_id == an_id][df_preamp.channel == a_ch].any():            # check that the channel is enabled before plotting
                detector_name = df_preamp.name[df_preamp.device_id == an_id][df_preamp.channel == a_ch].any()
                if len(detector_name)>0 and (detector_name[0] == 'B' or  detector_name[0] == 'P'):
                    plt.plot(ts_f,v, marker='+',ls=":", label="{} ".format(detector_name))
                else:   # don't add a legent key unless we're looking at a named detector channel
                    if(draw_non_detectors):
                        plt.plot(ts_f,v, marker='+',ls=":")

    ax=plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    fontP = FontProperties()
    fontP.set_size('x-small')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP, ncol=2)
    # plt.tight_layout()
    plt.ylabel("Baseline value (V)")
    plt.xlabel("Time")

def plot_waveforms(file_name, num_waveforms=5):
    df_gretina = pd.read_hdf(file_name, key="ORGretina4MWaveformDecoder")

    g4 = dl.Gretina4M(file_name)

    plt.figure()
    plt.xlabel("Time [ns]")
    plt.ylabel("ADC [arb]")

    for i, (index, row) in enumerate(df_gretina.iterrows()):
        wf = g4.parse_event_data(row)
        wf_sub = wf.data - np.mean(wf.data[:500])
        plt.plot(wf.time, wf_sub, ls="steps", c="b", alpha=0.1 )
        if i >=num_waveforms : break

def process(runNumber, n_max=5000):
    mjd_data_dir = os.path.join(os.getenv("DATADIR", "."), "mjd")
    raw_data_dir = os.path.join(mjd_data_dir,"raw")

    runList = [runNumber]

    from timeit import default_timer as timer
    start = timer()
    process_tier_0(raw_data_dir, runList, output_dir="", n_max=n_max)
    end = timer()
    print("Processing time: {} s".format(end - start))

if __name__=="__main__":
    main()
