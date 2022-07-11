import json
import pygama.lh5 as lh5
from matplotlib.patches import Rectangle
from pygama.dsp.WaveformBrowser import WaveformBrowser
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from alive_progress import alive_bar
import math
import pandas as pd

home_dir = "/home/pkrause/software/pygama/experiments/lar_commissioning/"
f_raw = home_dir + "data/com/raw/2022-04-13-sipm-test/20220425-184658-source-8000-athr200,12,2-es4000_raw.lh5"
f_config = home_dir + "software/meta/com/sipmtest-202110/r2d_config_v2.json"
f_dsp = home_dir +"data/com/dsp/2022-04-13-sipm-test/20220425-184658-source-8000-athr200,12,2-es4000_dsp.lh5"

num=0
while True:
    print('events {} to {}'.format(num*10000,(num+1)*10000))
    try:
        df = lh5.load_nda(f_dsp, ['channel','wf_blsub','bl_slope','t0','timestamp'],'spms/dsp',idx_list=list(range(num*10000,(num+1)*10000)))
    except:
         print('Done')
         break
    t = np.arange(0, len(df['wf_blsub'][0])*0.016, 0.016)

    ch_sel = (df['channel'] == 12)
    bl_sel = abs(df['bl_slope'])<0.01
    evts=list(range(0,len(ch_sel)))
    accepted = np.compress(np.logical_and(ch_sel,bl_sel),evts)
    #fig = plt.figure(figsize=(12,8))

    wfsub= df['wf_blsub'][accepted]
    nums= 1
    integrals= [0]*len(wfsub)
    for i in range(len(wfsub)):
        y=df['wf_blsub'][i]
        dy= np.gradient(y, 0.016)
        peaks, properties = find_peaks(dy, prominence=1, height=1000)
        if(len(peaks)>0):
            #plt.plot(t,y)
            #plt.axvline(x=t[peaks[0]])
            #plt.gca().add_patch(Rectangle((t[peaks[0]-2],0),6,max(y),fill=True, color='g', alpha=0.5, zorder=100, figure=fig))
            integrals[i]=sum(y[peaks[0]-2:peaks[0]+373])
        else:
            integrals[i]=sum(y[0:375])
    
    
    with open('th4kbqspe{}.npy'.format(num), 'wb') as f:
        np.save(f,integrals)
    num = num+1
#binwidth=100
#plt.hist(integrals,bins=range(0, math.ceil(max(integrals)) + binwidth, binwidth))
#plt.savefig('th4kbqspepng'.format(4))  
#plt.show()

#plt.show()
#plt.clf()

