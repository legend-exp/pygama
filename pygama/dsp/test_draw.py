from ProcessingChain import ProcessingChain
from transforms import *
import numpy as np
from units import *
from pygama.io import io_base as io
import pandas as pd
import matplotlib.pyplot as plt

verbose = 2 # 0=silent, 1=basic warnings, 2=basic output, 3=TMI
nReps = 8 # number of times to loop over file
wflen = 8192 # length of wf
nblock = 8 # number of wfs to process at once
bufferlen = 256 # number of wfs to read from disk at once

lh5 = io.LH5Store()
#data = lh5.read_object('ORSIS3302DecoderForEnergy', 't1_run1687.lh5', 0, bufferlen)
data = lh5.read_object('ORSIS3302DecoderForEnergy', 't1_run1687.lh5')

wf_in = data['waveform']['values'].nda
dt = data['waveform']['dt'].nda[0] * unit_parser.parse_unit(data['waveform']['dt'].attrs['units'])

# Set up processing chain
proc = ProcessingChain(block_width=nblock, clock_unit=dt, verbosity=verbose)
proc.add_input_buffer("wf", wf_in, dtype='float32')

proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
proc.add_processor(pole_zero, "wf_blsub", 75*us, "wf_pz")
proc.add_processor(trap_filter, "wf_pz", 10*us, 5*us, "wf_trap")
proc.add_processor(np.divide, "wf_trap", 10*us, "wf_trap_norm")
proc.add_processor(np.amax, "wf_trap_norm", 1, "trapE", signature='(n),()->()', types=['fi->f'])
proc.add_processor(avg_current, "wf_pz", 10, "curr")
proc.add_processor(np.amax, "curr", 1, "A_10", signature='(n),()->()', types=['fi->f'])
proc.add_processor(np.divide, "A_10", "trapE", "AoE")

wf     = proc.get_output_buffer("wf_blsub")
pz     = proc.get_output_buffer("wf_pz")
trap   = proc.get_output_buffer("wf_trap_norm")
curr   = proc.get_output_buffer("curr")
trapE  = proc.get_output_buffer("trapE")
aoe    = proc.get_output_buffer("AoE")

proc.execute()

plt.hist(trapE, bins=1000)
plt.xlabel("Energy (ADC)")
plt.show()

plt.scatter(trapE, aoe, s=1, marker='.')
plt.xlabel("Energy (ADC)")
plt.ylabel("A/E")
plt.ylim(0, 0.08)
plt.show()

xvals = np.arange(0, wf.shape[1]*10, 10)
plt.plot(xvals, wf[0,:], label="Raw WF")
plt.plot(xvals, pz[0,:], label="PZ WF")
plt.plot(xvals, trap[0,:], label="Trap WF")
plt.plot(xvals, curr[0,:], label="Current WF")
plt.xlabel("Time (ns)")
plt.ylabel("ADC")
plt.legend(loc='upper right')
plt.show()

