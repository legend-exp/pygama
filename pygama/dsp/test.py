import intercom as ic
import intercom_transforms as tr
import numpy as np

nReps = 8 # number of times to loop over file
wflen = 8192 # length of wf
nblock = 8 # number of wfs to process at once
bufferlen = 256 # number of wfs to read from disk at once

block_size = nblock*wflen
read_len = wflen*bufferlen

#Set up intercom
intercom = ic.Intercom(nblock, bufferlen)
wfbuffer=np.zeros(read_len, np.uint16)
intercom.add_input_buffer("wf(8192, float32)", wfbuffer)
intercom.add_processor(tr.mean_sigma, "wf[0:1000]", "bl", "bl_sig")
intercom.add_processor(np.subtract, "wf", "bl", "wf_blsub")
intercom.add_processor(tr.trapfilter, "wf_blsub", 1000, 500, "wf_trap")
intercom.add_processor(np.amax, "wf_trap", 1, "trapmax", signature='(n),()->()', types=['fi->f'])
intercom.add_processor(np.divide, "trapmax", 1000, "trapE")
Eout=np.zeros(bufferlen, np.float32)
intercom.add_output_buffer("trapE", Eout)

# Read from file and execute analysis
for i in range(nReps):
    with open('wfs.bin', 'r') as file:
        while True:
            # This is not ideal, since fromfile allocates memory and then it gets unnecessarily copied into wfbuffer. Unfortunately I can't find a numpy function to read it straight into the buffer... This will be fixed using the real pygama i/o
            pointlesslycopiedbuffer = np.fromfile(file, dtype=np.uint16, count=read_len)
            if(len(pointlesslycopiedbuffer)==0): break
            np.copyto(wfbuffer, pointlesslycopiedbuffer)
            
            intercom.execute()
            #print(Eout)
