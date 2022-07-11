import json
import pygama.lh5 as lh5
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
f_raw = home_dir + "data/com/raw/2021-10-15-sipm-test/singles-amaj1-10usec-trace_raw.lh5"
f_config = home_dir + "software/meta/com/sipmtest-202110/r2d_config_v2.json"
f_dsp = home_dir +"data/com/dsp/2022-04-13-sipm-test/20220413-151859-m4-randomtrigger_all_dsp.lh5"

df = lh5.load_nda(f_dsp, ['channel','wf_blsub','bl_slope','t0','timestamp'],'spms/dsp')

t = np.arange(0, len(df['wf_blsub'][0])*0.016, 0.016)
trig = [0]*len(df['channel'])
id = [0]*len(df['channel'])
trigPos = [0]*len(df['channel'])
evts= len(df['channel'])
#evts=500
counter = 0
for i in range(evts):
        y=df['wf_blsub'][i]
        dy= np.gradient(y, 0.016)
        peaks, properties = find_peaks(dy[0:375], prominence=1, height=1000)
        if df['channel'][i] ==25 and len(peaks)>0 and False:
            plt.plot(t[0:375],y[0:375])
            plt.axvline(x=t[peaks[0]])
        if len(peaks) > 0:
            trig[i] = 1
            trigPos[i]=peaks[0]
        if i>0:
            if df['channel'][i] < df['channel'][i-1]:
                counter = counter +1
        id[i] = counter
        if i%1000==0:
            print("{} of {} events processed".format(i,evts))


#plt.show()
df['trig'] = trig
df['trigPos'] = trigPos
df['id'] = id
exch =[[6,2,3,25,27]]
pandf = pd.DataFrame(data={key: df[key] for key in ['id','channel','timestamp','trig','trigPos']})
pandf.to_hdf('pk_randomtrigger_all.h5', key='pk')  
# for ex in exch:
#     pandf = pandf.drop(pandf[(pandf.channel > 29) | pandf['channel'].isin(ex)].index)
#     #print(exch.index(ex))
#     #print(pandf[(pandf.trig ==1) & (pandf.id==0)]['channel'])

#     trich=[0]*pandf.at[pandf.index[-1],'id']
#     for i in range(pandf.at[pandf.index[-1],'id']):
#         trich[i]=pandf['trig'][(pandf.id == i)].sum()

#     binwidth=1
#     #print(len(trich))
#     n, bins, _= plt.hist(trich, bins=range(min(trich), max(trich) + binwidth, binwidth), density=True)
#     plt.xlabel("Number of triggered channels")
#     plt.ylabel("probability")
#     plt.title("Number of simultanious triggering channels in a 6us coincidence window")

#     bin_width = bins[1] - bins[0]
#     integral = bin_width * sum(n)
#     print(n[0])
#     plt.savefig('triggerchProb{}.pdf'.format(4))  
#     #plt.show()

#     plt.hist(pandf[pandf['trig'] == 1]['channel'],bins=range(0, 29 + binwidth, binwidth))
#     y_vals = plt.gca().get_yticks()
#     scale=1/(6e-3*pandf.at[pandf.index[-1],'id'])
#     plt.gca().set_yticklabels(['{:10.1f}'.format(x * scale) for x in y_vals])
#     plt.xlabel("channel number")
#     plt.ylabel("rate [kHz]")
#     plt.title("rates per channel")
#     plt.savefig('triggerchdist{}.pdf'.format(4)) 

#     plt.clf()
#plt.show()