from ProcessingChain import ProcessingChain
from transforms import mean_stdev, trap_filter
import numpy as np
from units import *
from pygama.io import io_base as io

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
proc = ProcessingChain(block_width=nblock, clock_unit=dt, verbosity=2)
proc.add_input_buffer("wf", wf_in, dtype='float32')
proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
bl=proc.get_output_buffer("bl")
bl_sig=proc.get_output_buffer("bl_sig")
proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
proc.add_processor(trap_filter, "wf_blsub", 10*us, 5*us, "wf_trap")
proc.add_processor(np.amax, "wf_trap", 1, "trapmax", signature='(n),()->()', types=['fi->f'])
proc.add_processor(np.divide, "trapmax", 10*us, "trapE")
Eout=proc.get_output_buffer("trapE")

# Set up the LH5 output
lh5_out = io.LH5Table(size=proc.__buffer_len__)
lh5_out.add_field("trapE", io.LH5Array(Eout, attrs={"units":"ADC"}))
lh5_out.add_field("bl", io.LH5Array(bl, attrs={"units":"ADC"}))
lh5_out.add_field("bl_sig", io.LH5Array(bl_sig, attrs={"units":"ADC"}))

print(proc)

# Read from file and execute analysis
for i in range(nReps):
    proc.execute()

lh5.write_object(lh5_out, "data", "t2_run1687.lh5")
