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
    # uarrs = load_uproot(bfile, "MGTree", ihi=ihi,
    #                     brlist=['fWaveforms', 'fAuxWaveforms'])

    # -- read/write uproot hdf5 files
    # write_h5(hfile, uarrs)
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
    """
    comp_arr = []
    last = 0
    for samp in wfarr:
        diff = int(samp - last)
        rest = ((diff >> 31) ^ (diff << 1)) # zzEnc
        print(diff, rest)
        if rest < 0:
            print("error! exiting...")
            return
        while rest > 0:
            new_rest = rest >> 7
            comp_val = (rest & 0x7F) if new_rest == 0 else ((rest & 0x7F) | 0x80)
            comp_arr.append(comp_val)
            rest = new_rest
        last = samp
    return np.array(comp_arr)


def unpack_dvi(cwf):
    """
    unpack a diff-var-int waveform. corresponds to "readVLSigned"
    - MGTWaveform: https://github.com/mppmu/MGDO/blob/master/Root/MGTWaveform.cc
    """
    new_wf = []
    max_pos = 8
    acc = 0
    for samp in cwf:
        diff, x, pos = 0, 0, 0
        while True:
            if pos > max_pos:
                print("error, gonna bail")
                return

            a = samp & 0x7F
            b = a << pos
            x = x | b

            # print("samp {}  a {}  b {}  x {}".format(samp, a, b, x))
            print("samp {}  samp {:>9}  a {:>9}  b {:>9}  x {:>9}"
                  .format(samp, bin(samp), bin(a), bin(b), bin(x)))

            if (x & 0x80) == 0:
                diff = ((x >> 1) ^ -(x & 1))
                break
            else:
                print("i got here")
                pos += 7

        acc += diff
        new_wf.append(float(acc))

    return np.array(new_wf)


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

    iwf = 19
    while True:
        if iwf != 19:
            inp = input()
            if inp == "q": exit()
            if inp == "p": iwf -= 2
            if inp.isdigit(): iwf = int(inp) - 1
        iwf += 1
        # print(iwf)

        rwf = rarrs['fAuxWaveforms'][iwf]
        rwf = rwf[rwf != 0xDEADBEEF] # can use this to event-build

        noff = 154 # idk why the uproot wfs have junk at the beginning
        uwf = uarrs['fAuxWaveforms'][iwf][noff:]

        # try to encode the rwf to match the uwf
        cwf = compress_dvi(rwf)
        exit()

        # now try to unpack the cwf to get the rwf back
        # cwf2 = unpack_dvi(cwf)
        # exit()

        plt.cla()

        plt.plot(np.arange(len(rwf)), rwf, '-b', label="aux wf")
        plt.plot(np.arange(len(cwf2)), cwf2, '-r', label="unpack_dvi")

        # plt.plot(np.arange(len(uwf)), uwf, '-r',
        #          label="uproot, len {}, noff {}".format(len(uwf), noff))
        # plt.plot(np.arange(len(cwf)), cwf, '-b',
        #          label="dvi, len {}".format(len(cwf)))

        plt.legend()
        plt.show(block=False)
        plt.pause(0.001)


if __name__=="__main__":
    main()