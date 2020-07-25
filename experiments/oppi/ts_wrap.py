#!/usr/bin/env python3
import numpy as np
import pygama.io.lh5 as lh5
import matplotlib.pyplot as plt

# show how to correct for timestamp rollover with the struck 3302,
# and how to calculate the run duration using the dsp file (fastest).

f_dsp = '/Users/wisecg/Data/OPPI/dsp/oppi_run9_cyc2180_dsp.lh5'
# f_raw = '/data/eliza1/LEGEND/LH5/oppi/raw/oppi_run9_cyc2180_raw.lh5'
f_raw = 'test.lh5'

sto = lh5.Store()
# groups = sto.ls(f_dsp)
# data = sto.read_object('ORSIS3302DecoderForEnergy/raw', f_dsp)
# print(data.keys())

# groups = sto.ls(f_raw, 'ORSIS3302DecoderForEnergy/raw/')
data = sto.read_object('ORSIS3302DecoderForEnergy/raw/timestamp', f_raw)
ts = data.nda

# exit()
# wtfffff
# hitE = data['trapEmax'].nda
# ts = data['timestamp'].nda
# ts = np.sort(ts)
# print(hitE[:10])
for i in range(10000):
    print(i, ts[i])

# plt.semilogy(np.arange(len(ts)), ts, '.')
# plt.show()