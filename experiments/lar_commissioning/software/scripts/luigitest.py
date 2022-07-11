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

#from analyticalPulse import fitfunc

def func(t, a, a1, tau, tau1, t0): 
    return np.where(
        t<t0,
        0,
        a*np.exp(-(t-t0)/tau)+a1*np.exp(-(t-t0)/tau1))


def find_nearest_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

home_dir = "/home/pkrause/software/pygama/experiments/lar_commissioning/"
f_raw = home_dir + "data/com/raw/2021-10-15-sipm-test/singles-amaj1-10usec-trace_raw.lh5"
f_config = home_dir + "software/meta/com/sipmtest-202110/r2d_config_v2.json"
f_dsp = home_dir +"data/com/dsp/sipmtest-202110/singles-amaj1-10usec-trace_dsp.lh5"

df = lh5.load_nda(f_dsp, ['bl_mean','bl_std','bl_slope','channel','t0','wf_blsub'],'spms/dsp')
ch_sel = (df['channel'] == 1)
bl_sel = abs(df['bl_slope'])<0.01
tnull = df['t0']
evts=list(range(0,len(ch_sel)))
accepted = np.compress(np.logical_and(ch_sel,bl_sel),evts)



# wfs = lh5.load_nda(f_raw, ['values'], 'spms/raw/waveform', idx_list = accepted)['values'][:,0:1900]
# wfs = np.asarray(wfs, dtype = np.float32)

wfsub= df['wf_blsub'][accepted]
# bl_mean = df['bl_mean']
# print(bl_mean)
# for i in range(len(wfs)):
#      wfs[i] = wfs[i] - bl_mean[accepted][i]



t = np.arange(0, len(wfsub[0])*0.016, 0.016)

# shift wfs to the same t0 ~ 5us --> index 312
# t0_int = np.asarray(tnull, dtype = np.int16)[accepted]
# for i in range(len(t0_int)):
#     wfsub[i] = np.roll(wfsub[i], 312-t0_int[i], axis = 0)
#wfs = wfs[:,(int(r_t0_cut/0.016)-312):]   # delete values which were shifted to the start of the waveform

dwfs= [0]*len(wfsub)
dwfs2= [0]*len(wfsub)
for i in range(len(wfsub)):
    dwfs[i] = np.gradient(wfsub[i], 0.016)
    dwfs2[i] = np.gradient(dwfs[i], 0.016)
sumi= [0]*len(wfsub)
sumdiff= [0]*len(wfsub)
sumfit= [0]*len(wfsub)
evts = len(wfsub)
#evts= 20
singlePlot = True
with alive_bar(evts, ctrl_c=False, title=f'Progress: ') as bar:
    for i in range(evts):
        y = dwfs[i]
        sumi[i]=sum(wfsub[i][np.where((5.<= t) & (t<8.))])
        peaks, properties = find_peaks(y, prominence=1, height=800)
        sumdiff[i]=sum(wfsub[i][peaks+1])
        if(len(peaks))>0 and singlePlot:
            #plt.plot(t,wfsub[i])
            #plt.plot(t[peaks+1], wfsub[i][peaks+1], "x")
            for j in range(len(peaks)):
                if j <len(peaks)-1:
                    trange= t[peaks[j]:peaks[j+1]-10]
                    wrange= wfsub[i][peaks[j]:peaks[j+1]-10]

                else:
                    trange= t[peaks[j]:]
                    wrange= wfsub[i][peaks[j]:]
                
                popt, pcov = curve_fit(func, 
                                            trange,
                                            wrange, 
                                            p0=[wfsub[i][peaks[j]+1]/2,wfsub[i][peaks[j]+1],1.,0.1,t[peaks[j]]],
                                            bounds=([-np.inf,-np.inf,0.8,0.01,0],[np.inf,np.inf,50,0.5,np.inf]))
                #plt.plot(trange, func(trange, *popt), 'r-')
                #print(popt)
                sumfit[i] += integrate.quad(func,trange[0],trange[-1], args=(popt[0],popt[1],popt[2],popt[3],popt[4]))[0]
            #singlePlot=False
            #plt.show()
        bar()

plt.hist(sumi/max(sumi), bins='auto',label='sum')
plt.hist(sumdiff/max(sumdiff), bins='auto',label='peaks amplitude')
plt.hist([item / max(sumfit) for item in sumfit], bins='auto',label='sum fit')
plt.legend()
plt.show()

