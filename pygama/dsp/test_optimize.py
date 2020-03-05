from ProcessingChain import ProcessingChain
from transforms import *
import numpy as np
from units import *
from pygama.io import io_base as io
import matplotlib.pyplot as plt

verbose = 1 # 0=silent, 1=basic warnings, 2=basic output, 3=TMI
nblock = 8 # number of wfs to process at once


lh5 = io.LH5Store()
data = lh5.read_object('ORSIS3302DecoderForEnergy', 't1_run1687.lh5')

wf_in = data['waveform']['values'].nda
dt = data['waveform']['dt'].nda[0] * unit_parser.parse_unit(data['waveform']['dt'].attrs['units'])

# Set up processing chain
proc = ProcessingChain(block_width=nblock, clock_unit=dt, verbosity=verbose)
proc.add_input_buffer("wf", wf_in, dtype='float32')

proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")

# Set up all of the different trap filters. Note that this is reusing the
# wf_pz and wf_trap variables!
for t in range(70, 80):
    proc.add_processor(pole_zero, "wf_blsub", t*us, "wf_pz")
    proc.add_processor(trap_filter, "wf_pz", 10*us, 5*us, "wf_trap")
    proc.add_processor(np.amax, "wf_trap", 1, "trapmax", signature='(n),()->()', types=['fi->f'])
    proc.add_processor(np.divide, "trapmax", 10*us, "trapE_"+str(t))

# Set up the LH5 output
lh5_out = io.LH5Table(size=proc.__buffer_len__)
for t in range(70, 80):
    lh5_out.add_field("trapE_"+str(t), io.LH5Array(proc.get_output_buffer("trapE_"+str(t)), attrs={"units":"ADC"}))

print(proc)

# Read from file and execute analysis
proc.execute()

lh5.write_object(lh5_out, "data", "optimizeE_run1687.lh5")

for t in range(70, 80):
    plt.hist(lh5_out["trapE_"+str(t)].nda, label=str(t)+" us", bins=50, lw=2, alpha = 0.5, range=(8100, 8600))
plt.xlabel("Energy (ADC)")
plt.legend(loc="upper left")
plt.show()
