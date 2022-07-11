import json
from turtle import home
import pygama.lh5 as lh5
from pygama.dsp.WaveformBrowser import WaveformBrowser
import matplotlib.pyplot as plt
import numpy as np
print(np.__path__)
import pygama.analysis.histograms as ph
from scipy.optimize import curve_fit
from functools import reduce

home_dir = "/home/pkrause/software/pygama/experiments/lar_commissioning/"
f_raw = home_dir + "data/com/raw/sipmtest-202110/maj6-500usec-trace-lowthr-ch-0-11_raw.lh5"
f_config = home_dir + "software/meta/com/sipmtest-202110/r2d_config.json"
f_dsp = ""



def draw_WF(raw_file, dsp_file, channel,num_draw):
    ch = lh5.load_dfs(raw_file, ['channel'], '/spms/raw')['channel']
    ch_sel = ch == 1

    with open(dsp_file) as f:
        dsp_cfg = json.load(f)['dsp_config']

    wb = WaveformBrowser(raw_file, lh5_group='/spms/raw',
        x_lim=(0, 50000),
        y_lim=(-10, 100),
        dsp_config=dsp_cfg,
        waveforms=['wf_blsub'],
        selection=ch_sel,
        verbosity=0,
        n_drawn=num_draw)
    wb.draw_next()#; wb.ax.axhline(0)
    plt.show()

def pe_spectrum_area(raw_file, dsp_file, channel):
    area = lh5.load_nda(dsp_file, ['sum_wf'],'spms/dsp')['sum_wf']
    ch = lh5.load_nda(raw_file, ['channel'], 'spms/raw')['channel'][0:1000000]
    number_wfs_ch = len(np.where(ch == channel)[0])
    energy = area[ch == channel]
    bins = np.linspace(-5000, 50000, 500)
    plt.hist(energy, bins = bins, label = 'Area under waveform (0-30us)')
    plt.xlim(-5000,50000)
    #plt.ylim(0,1200)
    plt.xlabel('ADC')
    plt.ylabel('Number of events')
    plt.legend()
    plt.title('channel '+str(channel)+' ('+str(number_wfs_ch)+' waveforms)')
    plt.show()


pe_spectrum_area(f_raw,f_dsp,1)


