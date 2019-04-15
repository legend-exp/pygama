#!/usr/bin/env python
import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward
import h5py
import psutil
from pprint import pprint
from pygama.utils import sizeof_fmt

def main():
    """
    uproot tutorial:
    https://hub.mybinder.org/user/scikit-hep-uproot-0ldc9pe0/lab
    """
    run = 25789
    gfile = "~/Data/mjddatadir/gatified/mjd_run{}.root".format(run)
    bfile = "~/Data/mjddatadir/built/OR_run{}.root".format(run)
    hfile = "~/Data/uproot/up_run{}.h5".format(run)
    hfile2 = "~/Data/uproot/root_run{}.h5".format(run)

    # explore_file(bfile)

    # -- read root file with uproot
    ihi = 100

    # uarrs = load_uproot(bfile, "MGTree", ihi=ihi)
    uarrs = load_uproot(bfile, "MGTree", ihi=ihi,
                        brlist=['fWaveforms', 'fAuxWaveforms'])

    # -- read/write uproot hdf5 files
    write_h5(hfile, uarrs)
    # h5arrs = load_h5(hfile)

    # -- read/write ROOT hdf5 files directly
    # uarrs = load_root(bfile, "MGTree", ihi=ihi)
    # write_h5(hfile2, uarrs)
    # h5arrs = load_h5(hfile2)

    # -- uproot reads in kDiffVarInt-compressed wfs,
    # so compare to our uncompressed output & make sure we can reproduce it.
    test_dvi(run)


def explore_file(file):
    """
    access a file, print branches, convert a small amount of data
    to numpy arrays with 'awkward' and check the resulting datatypes
    """
    ufile = uproot.open(file) # uproot.rootio.ROOTDirectory
    utree = ufile["MGTree"]   # uproot.rootio.TTree

    # pprint(ufile.allclasses()) # various uproot types
    # pprint(utree.allkeys())    # TBranchElement - don't use these directly

    nevt = 100
    uarrs = utree.arrays(entrystop = nevt) # dict.  can use this for chunking!
    # pprint(list(uarrs.keys())) # same as allkeys

    for k in sorted(uarrs.keys()):
        name = k.decode('utf-8')
        vals = uarrs[k]

        if isinstance(vals, np.ndarray):
            # print(name, type(vals), vals.shape)
            # print(vals) # works 100%
            continue

        elif isinstance(vals, awkward.ObjectArray):
            # print(name, type(vals), vals.shape)
            # if "Waveforms" in name:
            #     continue
            # else:
            #     print(vals) # fails on fWaveforms
            continue

        elif isinstance(vals, awkward.JaggedArray):
            # print(name, type(vals), vals.shape)
            # print(vals) # works 100%
            continue

        else:
            print("Couldn't parse type for:", name)


def load_uproot(file, ttname=None, ilo=None, ihi=None, brlist=None):
    """
    load uproot data into awkward arrays
    """
    ufile = uproot.open(file)
    utree = ufile[ttname]

    # call uproot's 'arrays' function
    if brlist is None:
        uarrs = utree.arrays(entrystart=ilo, entrystop=ihi)
    else:
        uarrs = utree.arrays(branches=brlist, entrystart=ilo, entrystop=ihi)
        for br in brlist:
            tmp = bytes(br, 'utf-8')
            if isinstance(uarrs[tmp], awkward.ObjectArray):
                # cast to awkward.JaggedArray, with a 1-d np shape: (nevt, )
                uarrs[tmp] = uarrs[tmp].content

    # change the keys to str()
    for bkey in list(uarrs.keys()):
        skey = bkey.decode('utf-8')
        uarrs[skey] = uarrs.pop(bkey)

    return uarrs


def write_h5(file, uarrs, mode="w"):
    """
    write an hdf5 file created by awkward, from awkward.JaggedArrays.
    - awkward/awkward/persist.py
    - awkward/tests/test_hdf5.py
    """
    file = os.path.expanduser(file)
    with h5py.File(file, mode) as hf:
        ah5 = awkward.hdf5(hf)
        for ds in uarrs:
            awk = awkward.JaggedArray.fromiter(uarrs[ds])
            ah5[ds] = awk


def load_h5(file):
    """
    read an hdf5 file created by awkward
    """
    h5arrs = {}
    file = os.path.expanduser(file)
    with h5py.File(file) as hf:
        ah5 = awkward.hdf5(hf)
        for ds in ah5:
            h5arrs[ds] = ah5[ds]
    return h5arrs


def load_root(file, ttname, ilo=None, ihi=None, brlist=None):
    """
    use pyroot to save decoded (uncompressed) MGTWaveforms,
    into awkward's hdf5 file object.
    this is to compare against uproot, which reads compressed wfs.
    """
    from ROOT import TFile, TTree, MGTWaveform, MJTMSWaveform

    tf = TFile(file)
    tt = tf.Get(ttname)
    nevt = tt.GetEntries()
    tt.GetEntry(0)
    is_ms = tt.run.GetUseMultisampling()

    # build w/ python primitive types and convert to JaggedArray after the loop.
    # JaggedArray requires one entry per event (have to handle multi-detector).
    br_list = ['fWaveforms', 'fAuxWaveforms', 'fMSWaveforms'] if is_ms else ['fWaveforms']
    pyarrs = {br:[] for br in br_list}
    delim = 0xDEADBEEF

    # loop over tree
    ilo = 0 if ilo == None else ilo
    ihi = nevt if ihi == None else ihi

    for i in range(ilo, ihi):
        tt.GetEntry(i)
        nwf = tt.channelData.GetEntries()

        # concat each hit into a single array
        ewf, ewfa, ewfms = [], [], []
        for j in range(nwf):
            if is_ms:
                wf = tt.event.GetWaveform(j)
                wfa = tt.event.GetAuxWaveform(j)
                wfms = MJTMSWaveform(wf, wfa)
                ewf.extend([wf[i] for i in range(wf.GetLength())])
                ewfa.extend([wfa[i] for i in range(wfa.GetLength())])
                ewfms.extend(wfms[i] for i in range(wfms.GetLength()))
                ewf.append(delim)
                ewfa.append(delim)
                ewfms.append(delim)
            else:
                wf = tt.event.GetWaveform(j)
                ewf.extend([wf[i] for i in range(wf.GetLength())])
                ewf.append(delim)

        if is_ms:
            pyarrs['fWaveforms'].append(ewf)
            pyarrs['fAuxWaveforms'].append(ewfa)
            pyarrs['fMSWaveforms'].append(ewfms)
        else:
            pyarrs['fWaveforms'].append(ewf)

    uarrs = {}
    for wf in pyarrs.keys():
        uarrs[wf] = awkward.fromiter(pyarrs[wf])

    return uarrs


def compress_dvi(wfarr):
    """
    run diff-var-int compression on a wf. corresponds to "writeVLSigned"
    - MGTWaveform: https://github.com/mppmu/MGDO/blob/master/Root/MGTWaveform.cc
    - zigzag encoding: https://gist.github.com/mfuerstenau/ba870a29e16536fdbaba
      [signed]=[encoded] : 0=0, -1=1, 1=2, -2=3, 2=4, -3=5, 3=6 ....
    """
    # print("\nrunning compressor...")
    comp_arr = []
    prev_samp = 0
    for i, samp in enumerate(wfarr):

        # take difference and zzencode it
        diff = int(samp - prev_samp)
        zdiff = ((diff >> 32) ^ (diff << 1)) # zigzag encode: assume 32-bit ints
        if zdiff < 0:
            print("error! exiting ...")
            exit()

        # write out the encoded value, 7 bits at a time
        while zdiff > 0:
            znext = zdiff >> 7
            if znext == 0:
                cint8 = zdiff & 0x7F
            else:
                cint8 = zdiff & 0x7F | 0x80
            comp_arr.append(cint8)
            zdiff = znext

        prev_samp = samp

    return np.array(comp_arr)


def decompress_dvi(cwf):
    """
    unpack a diff-var-int waveform. corresponds to "readVLSigned"
    - MGTWaveform: https://github.com/mppmu/MGDO/blob/master/Root/MGTWaveform.cc
    Basically we read in a series of 7-bit integers.
    """
    # print("\nrunning decompressor...")
    wf = []
    wf_val, prev7 = 0, 0

    for i, cint8 in enumerate(cwf):

        # get current value, OR it with any previous value
        if prev7 == 0:
            cint7 = cint8 & 0x7F
        else:
            cint7 = (cint8 << 7) | prev7

        # if bit 8 is 0, we've reached the end of this decoded value
        if cint8 & 0x80 == 0:
            diff = (cint7 >> 1) ^ -(cint7 & 1)
            wf_val += diff
            wf.append(float(wf_val))
            prev7 = 0 # reset

        # if not, save this cint7 to be OR'd with the next one
        else:
            prev7 = cint7

    return np.array(wf)


def test_dvi(run):
    """
    load hdf5 files and test the DiffVarInt compression
    """
    f1 = "~/Data/uproot/root_run{}.h5".format(run) # root (uncomp.) file
    f2 = "~/Data/uproot/up_run{}.h5".format(run) # uproot (dvi) file

    rarrs = load_h5(f1)
    uarrs = load_h5(f2)

    # check memory usage
    pid = os.getpid()
    mem = sizeof_fmt(psutil.Process(pid).memory_info().rss)
    print("PID: {}, current mem usage: {}".format(pid, mem))

    f, p = plt.subplots(2, 1, figsize=(10, 7))
    iwf = 19
    while True:
        if iwf != 19:
            inp = input()
            if inp == "q": exit()
            if inp == "p": iwf -= 2
            if inp.isdigit(): iwf = int(inp) - 1
        iwf += 1
        print("iwf:", iwf)

        # encode and decode the "true" root waveform - checks DVI algorithm
        # rwf = rarrs['fAuxWaveforms'][iwf]
        rwf = rarrs['fWaveforms'][iwf]
        # rwf = rarrs['fMSWaveforms'][iwf]
        rwf = rwf[rwf != 0xDEADBEEF] # can use this to event-build ...
        crwf = compress_dvi(rwf)
        drwf = decompress_dvi(crwf)

        # decode the uproot waveform -- can we reproduce the original?

        # idk why the uproot wfs have junk at the beginning
        noff = 143 # this gets the baseline and rising edge in the right place
        uwf = uarrs['fWaveforms'][iwf][noff:]
        duwf = decompress_dvi(uwf)

        # peel off

        # show waveforms
        p[0].cla()
        p[0].plot(np.arange(len(rwf)), rwf, '-b', lw=6, label="root, original")
        p[0].plot(np.arange(len(drwf)), drwf, '-m', lw=4, label="root, rconst")
        p[0].plot(np.arange(len(duwf)), duwf, '-c', lw=2, label="uproot, rconst")
        p[0].legend()

        # show compressed wfs
        p[1].cla()
        p[1].plot(np.arange(len(crwf)), crwf, '-b', label='root, compressed')
        p[1].plot(np.arange(len(uwf)), uwf, '-g', label='uproot, compressed')
        p[1].legend()

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)


if __name__=="__main__":
    main()