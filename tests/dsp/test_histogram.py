import os
from pathlib import Path

import numpy as np

import pygama.lgdo.lh5_store as store
from pygama.dsp import build_dsp

config_dir = Path(__file__).parent / "configs"
dsp_file = "/tmp/LDQTA_r117_20200110T105115Z_cal_geds__numpy_test_dsp.lh5"


def test_histogram_fixed_width(lgnd_test_data):
    dsp_config = {
        "outputs": ["wf_hist_c" , "wf_borders_c"],
        "processors": {
            "wf_hist_c , wf_borders_c": {
                "function": "histogram",
                "module": "pygama.dsp.processors.histogram",
                "args": ["waveform","[100]","wf_hist_c(100)", "wf_borders_c(101)"],
                "unit": ["ADC", "ADC"]
            }
        }
    }
    build_dsp(
        f_raw=lgnd_test_data.get_path(
            "lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"
        ),
        f_dsp=dsp_file,
        dsp_config=dsp_config,
        write_mode="r",
    )
    assert os.path.exists(dsp_file)
    
    df = store.load_nda(dsp_file, ["wf_hist_c", "wf_borders_c"], "geds/dsp/")
    
    assert(len(df['wf_hist_c'][0])+1==len(df['wf_borders_c'][0]))
    for i in range(2,len(df['wf_borders_c'][0])):
        a=df['wf_borders_c'][0][i-1]-df['wf_borders_c'][0][i-2]
        b=df['wf_borders_c'][0][i]  -df['wf_borders_c'][0][i-1]
        assert(round(a,2)==round(b,2))

def test_histogram_variable_width(lgnd_test_data):
    dsp_config = {
        "outputs": ["wf_hist_c" , "wf_borders_c"],
        "processors": {
            "wf_hist_c , wf_borders_c": {
                "function": "histogram",
                "module": "pygama.dsp.processors.histogram",
                "args": ["waveform","[1,2,3,4,1,6,10,8,2,1]","wf_hist_c(10)", "wf_borders_c(11)"],
                "unit": ["ADC", "ADC"]
            }
        }
    }
    build_dsp(
        f_raw=lgnd_test_data.get_path(
            "lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"
        ),
        f_dsp=dsp_file,
        dsp_config=dsp_config,
        write_mode="r",
    )
    assert os.path.exists(dsp_file)
    
    df = store.load_nda(dsp_file, ["wf_hist_c", "wf_borders_c"], "geds/dsp/")
    bins = np.array([1,2,3,4,1,6,10,8,2,1])
    
    assert(len(df['wf_hist_c'][0])+1==len(df['wf_borders_c'][0]))
    for i in range(2,len(df['wf_borders_c'][0])):
        a= (df['wf_borders_c'][0][i-1]-df['wf_borders_c'][0][i-2])/bins[i-2]
        b= (df['wf_borders_c'][0][i]  -df['wf_borders_c'][0][i-1])/bins[i-1]
        assert(round(a,2)==round(b,2))