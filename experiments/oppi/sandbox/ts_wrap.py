#!/usr/bin/env python3
import numpy as np
import pygama.lh5 as lh5
import matplotlib.pyplot as plt

# show how to correct for timestamp rollover with the struck 3302,
# and how to calculate the run duration using the dsp file (fastest).

f_dsp = '/Users/wisecg/Data/OPPI/dsp/oppi_run9_cyc2180_dsp.lh5'

sto = lh5.Store()
data = sto.read_object('ORSIS3302DecoderForEnergy/raw', f_dsp)

# correct for timestamp rollover
clock = 100e6 # 100 MHz
UINT_MAX = 4294967295 # (0xffffffff)
t_max = UINT_MAX / clock

# ts = data['timestamp'].nda.astype(np.int64) # has to be signed for np.diff
ts = data['timestamp'].nda / clock # converts to float

tdiff = np.diff(ts)
tdiff = np.insert(tdiff, 0 , 0)
iwrap = np.where(tdiff < 0)
iloop = np.append(iwrap[0], len(ts))

ts_new, t_roll = [], 0
for i, idx in enumerate(iloop):
    ilo = 0 if i==0 else iwrap[0][i-1]
    ihi = idx
    ts_block = ts[ilo:ihi]
    t_last = ts[ilo-1]
    t_diff = t_max - t_last
    ts_new.append(ts_block + t_roll)
    t_roll += t_last + t_diff 
ts_corr = np.concatenate(ts_new)

# calculate runtime
rt = ts_corr[-1] / 60 # minutes

# plt.plot(np.arange(len(ts)), ts, '.')
plt.plot(np.arange(len(ts_corr)), ts_corr, '.', label=f'runtime: {rt:.2f} min')
plt.legend()
plt.show()