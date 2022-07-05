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
    # debug(bfile)
    quick_gat(gfile)
    exit()

    # -- read root file with uproot
    ihi = 100

    # uarrs = run_uproot(bfile, "MGTree", ihi=ihi)
    # uarrs = run_uproot(bfile, "MGTree", ihi=ihi,
                        # brlist=['fWaveforms', 'fAuxWaveforms'])

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


def debug(file):
    """
    fWaveforms is a custom class.  can we get uproot to read it,
    a little better than just reading out .content directly?
    - https://github.com/scikit-hep/uproot/issues/124
    """
    ufile = uproot.open(file)
    utree = ufile["MGTree"]

    # for c in ufile.allclasses():
        # print(c)
    # utree.show() # print class and streamer types

    # # -- some fancy pivarski array calls
    # from uproot import interpret, asjagged, astable, asdtype
    # arr1 = utree["class1"].array(interpret(utree["class1"], cntvers=True))
    # arr2 = utree["class2"].array(interpret(utree["class2"], tobject=False))
    # arr3 = utree["arr3"].array(asjagged(astable(asdtype([("id", "i4"), ("pmt", "u1"), ("tdc", "u4"), ("tot", "u1")])),skipbytes=10))
    # arr4 = utree["arr4"].array(asjagged(astable(asdtype([("id", "i4"), ("pmt", "u1"), ("tdc", "u4"), ("tot", "u1"),(" cnt", "u4"), (" vers", "u2"), ("trigger_mask", "u8")])), skipbytes=10))
    # print(arr1.columns)

    wfs = utree["fWaveforms"].array(entrystop=100)
    # wfs = utree.arrays([b'fWaveforms'], entrystop=100)
    # print(dir(wfs))
    wfc = wfs[b'fWaveforms'].content
    print(wfc[0])


def quick_gat(file):
    """
    uproot has a rad auto-dataframe method that should work for
    single-valued gatified files.
    """
    ufile = uproot.open(file)
    utree = ufile["mjdTree"]
    # df = ufile["mjdTree"].pandas.df() # this fails, probably on the MG objects
    df = ufile["mjdTree"].pandas.df(["trapENFCal","trapENAF"]) # this works, fast!

    # # try to filter out the bad objects
    # cols = []
    # arrtmp = utree.arrays(entrystop=10)
    # for key in arrtmp.keys():
    #     # print(key, type(arrtmp[key]))
    #     if isinstance(arrtmp[key], awkward.ObjectArray):
    #         continue
    #     cols.append(key)

    # can try manually creating the df
    # utree = ufile["threshTree"].arrays()
    # cols = []
    # for k in utree:
    #     if not isinstance(utree[k][0], np.ndarray):
    #         continue
    #     cols.append(pd.Series(utree[k][0], name=k.decode('utf-8')))
    # df = pd.concat(cols, axis=1)

    # df = ufile["mjdTree"].pandas.df(cols, flatten=True)
    print(df)


def run_uproot(file, ttname=None, ilo=None, ihi=None, brlist=None):
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

        # take difference
        diff = int(samp) - int(prev_samp)

        if diff == 0:
            comp_arr.append(0)
            continue

        # zzencode the diff, assuming 32-bit ints
        zdiff = ((diff >> 32-1) ^ (diff << 1))
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

    NOTE:
    these are written before the MGTWaveform compressor runs:
    - UInt_t R__c = R__b.WriteVersion(MGTWaveform::Class(), kTRUE);
    - R__b.WriteDouble(fSampFreq);
    - R__b.WriteDouble(fTOffset);
    - R__b.WriteInt(fWFType);
    - R__b.WriteULong64(fData.size());
    - R__b.WriteInt(fWFEncScheme);
    - R__b.WriteInt(fID);
    - R__b.SetByteCount(R__c, kTRUE);

    Need to implement:
    - ReadDouble, ReadInt, ReadULong64
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
        print("iwf:", iwf,"\n")

        # wf_type = 'fWaveforms'
        wf_type = 'fAuxWaveforms'

        # encode and decode the "true" root waveform - checks DVI algorithm
        rwf = rarrs[wf_type][iwf]
        rwf = rwf[rwf != 0xDEADBEEF] # can use this to event-build ...
        crwf = compress_dvi(rwf)
        drwf = decompress_dvi(crwf)

        # # compress/decomp a 2nd time, just for fun
        # cwf2 = compress_dvi(duwf)
        # dwf2 = decompress_dvi(cwf2)

        # -- decode the uproot waveform & metadata.
        # ilo and ihi are empirical for one waveform
        ilo, ihi = 143, -5
        uarr = uarrs[wf_type][iwf]
        header = uarr[:ilo]
        uwf = uarr[ilo:ihi]
        footer = uarr[ihi:]
        duwf = decompress_dvi(uwf)

        # -- decode the metadata --
        # ISSUE: Until we can decode the entire header, it's not safe
        # to run this routine on arbitrary built files -- the header length
        # might change.

        # # decode the header to unicode chars
        # print("Header is:")
        # print(header,"\n")
        # print([(chr(v).encode()).decode() for v in header])
        # print("\n", "".join([(chr(v).encode()).decode() for v in header]))
        # print("\nFooter is:", "".join([(chr(v).encode()).decode() for v in footer]))

        fSampFreq = header[-32:-24] # 64 bit double?
        fTOffset = header[-24:-16] # 64 bit double?
        fWFType = header[-16:-12] # 32 bit int
        fDataSize = header[-12:-4] # 64 bit int
        fWFEncScheme = header[-4:] # 32 bit int

        # ian says each value is an 8 bit int.
        fID = (footer[-3] << 8) + footer[-2]

        # print("remaining header:",header[:-32])
        # NOTE: to get the rest, ian suggests starting here:
        # https://github.com/root-project/root/blob/331efa4c00fefc38980eaa\
        # f7b41b8e95fcd1a23b/io/io/src/TBufferFile.cxx#L2975

        # converting to unicode chars i get:
        # @ ÿÿÿÿMGTClonesArray@	@ MGTWaveform;5@Î@@?¹@DWaveforms
        # MGTClonesArray --> MGTWaveform --> Waveforms (a TClonesArray thing)

        # exit()

        # show waveforms
        p[0].cla()
        p[0].plot(np.arange(len(drwf)), drwf, '.g', ms=12, label="root, rconst")
        p[0].plot(np.arange(len(rwf)), rwf, '.b', ms=8, label="root, original")
        # p[0].plot(np.arange(len(dwf2)), dwf2, '.r', label="root, doubled")
        p[0].plot(np.arange(len(duwf)), duwf, '.r', ms=4, label="uproot, rconst")
        # p[0].set_xlim(0, 40)
        # p[0].set_ylim(-900, -850)
        p[0].legend()

        # show compressed wfs
        p[1].cla()
        p[1].plot(np.arange(len(crwf)), crwf, '.b', ms=8, label='root, compressed')
        p[1].plot(np.arange(len(uwf)), uwf, '.r', ms=4, label='uproot, compressed')
        # p[1].set_xlim(0, 40)
        # p[1].set_ylim(-5,15)
        p[1].legend(loc=2)

        # # # show residuals
        # p[2].cla()
        # ts = np.arange(len(drwf))
        # p[2].plot(ts, rwf - drwf, ".k", label="orig - reconst")
        # # p[2].set_xlim(0, 40)
        # p[2].legend()

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)


if __name__=="__main__":
    main()