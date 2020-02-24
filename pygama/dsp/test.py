from ProcessingChain import ProcessingChain
from transforms import mean_rms, trap_filter
import numpy as np
from units import *

nReps = 8 # number of times to loop over file
wflen = 8192 # length of wf
nblock = 8 # number of wfs to process at once
bufferlen = 256 # number of wfs to read from disk at once

block_size = nblock*wflen
read_len = wflen*bufferlen

# Set up processing chain
proc = ProcessingChain(block_width=nblock, buffer_len=bufferlen, clock_unit=100*mhz, verbosity=3)
wfbuffer=np.zeros(read_len, np.uint16)
proc.add_input_buffer("wf(8192, float32)", wfbuffer)
proc.add_processor(mean_rms, "wf[0:1000]", "bl", "bl_sig")
proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
proc.add_processor(trap_filter, "wf_blsub", 10*us, 5*us, "wf_trap")
proc.add_processor(np.amax, "wf_trap", 1, "trapmax", signature='(n),()->()', types=['fi->f'])
proc.add_processor(np.divide, "trapmax", 1000, "trapE")
Eout=np.zeros(bufferlen, np.float32)
proc.add_output_buffer("trapE", Eout)

print(proc)

# Read from file and execute analysis
for i in range(nReps):
    with open('wfs.bin', 'r') as file:
        while True:
            # This is not ideal, since fromfile allocates memory and then it gets unnecessarily copied into wfbuffer. Unfortunately I can't find a numpy function to read it straight into the buffer... This will be fixed using the real pygama i/o
            pointlesslycopiedbuffer = np.fromfile(file, dtype=np.uint16, count=read_len)
            if(len(pointlesslycopiedbuffer)==0): break
            np.copyto(wfbuffer, pointlesslycopiedbuffer)
            
            proc.execute()
            #print(Eout)
