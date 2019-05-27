#!/usr/bin/env python3
import sys, os
import argparse
import time
import uproot
import awkward
import h5py
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from pygama.utils import update_progress

def main():
    """
    Convert Majorana skim, gatified, and built data files to HDF5, using uproot,
    awkward-array, pandas, and PyROOT.  Can also retrieve waveform data.
    ** Requires $MJDDATADIR contain three folders: `gatified`, `built`, `hdf5`
    Usage:
    $ python uproot_mjd.py -r 23578  (finds files by run number with GATDataSet)
    $ python uproot_mjd.py -f skimDS6_1.root (run directly on any ROOT file)
    """
    par = argparse.ArgumentParser(description="Majorana ROOT/HDF5 conversion")
    arg = par.add_argument
    arg("-r", "--run", nargs=1, help="access a run number (gatified/built)")
    arg("-f", "--file", nargs=1, help="access a filename (skim or gatified)")
    arg("-m", "--mode", nargs=1, help="HDF5: `pandas` (default), or `awkward`")
    arg("-n", "--nevt", nargs=1, help="limit number of events")
    arg("-w", "--wfs", action="store_true", help="save waveforms")
    arg("-o", "--wfopts", nargs='*', help="set waveform mode options")
    arg("-s", "--skip", action="store_true", help="skip to waveform saving")
    arg("-t", "--test", action="store_true", help="test mode")
    args = vars(par.parse_args())

    run = int(args["run"][0]) if args["run"] else None
    infile = args["file"][0] if args["file"] else None
    mode = args["mode"][0] if args["mode"] else "pandas"
    nevt = int(args["nevt"][0]) if args["nevt"] else None

    if args["test"]:
        # test_dvi(run)
        if mode=="pandas": read_pandas(run, infile)
        if mode=="awkward": read_awkward(run)
        exit()

    # -- default: save single and vector-valued data into hdf5 groups
    if not args["skip"]:
        write_HDF5(run, infile, mode, nevt)

    # -- optional: add a group to the hdf5 file for waveform data
    if args["wfs"]:
        wf_mode = args["wfopts"][0] if args["wfopts"] else "root"
        write_waveforms(run, infile, wf_mode)


def write_HDF5(run=None, infile=None, mode="pandas", nevt=None):
    """
    primary writer function.  contains several Majorana-specific choices.
    works on gatified or skim data.  (TODO: simulation data?)
    """
    if run is None and infile is None:
        print("You must specify either a run number or input filename.")
        exit()

    # declare inputs and outputs
    if run is not None:
        from ROOT import GATDataSet
        gds = GATDataSet()
        gfile = gds.GetPathToRun(run, GATDataSet.kGatified)
        infile, tname = gfile, "mjdTree"
        ufile = uproot.open(infile)

    if infile is not None:
        ufile = uproot.open(infile)
        # auto-detect and use the name of the first TTree we find
        for uc in ufile.allclasses():
            cname, ctype = uc[0], str(uc[1])
            if "TTree" in ctype:
                tname = cname.decode("utf-8").split(";")[0]
                print("Found TTree:", tname)
                break

    # strip the path and extension off the filename to create the hfile
    if run is None:
        label = infile.split("/")[-1].split(".root")[0]
        hfile = "{}/hdf5/{}.h5".format(os.environ["MJDDATADIR"], label)
    else:
        hfile = "{}/hdf5/mjd_run{}.h5".format(os.environ["MJDDATADIR"], run)

    # these MGDO object members don't have the same number of entries
    # as the rest of the vector-valued branches, so skip them for now
    skip_names = ["i","iH","iL","j","jH","jL","rawRun","c0Channels"]

    # get all relevant TTree branches & sort by data type
    event_names, hit_names = [], []

    utree = ufile[tname]
    uarrs = utree.arrays(entrystop=1)
    for k in sorted(uarrs.keys()):
        name = k.decode('utf-8')
        vals = uarrs[k]

        if isinstance(vals, np.ndarray):
            event_names.append(k)

        elif isinstance(vals, awkward.JaggedArray):
            if name in skip_names:
                continue
            hit_names.append(k)

        elif isinstance(vals, awkward.ObjectArray):
            # print("Skipping branch:", name)
            continue

    # write to pandas HDF5 (pytables)
    if mode=="pandas":
        print("writing pandas hdf5.\n  input:{}\n  output:{}".format(infile, hfile))

        df_events = ufile[tname].pandas.df(event_names, entrystop=nevt)
        df_hits = ufile[tname].pandas.df(hit_names, entrystop=nevt)

        if os.path.isfile(hfile):
            os.remove(hfile)

        opts = {
            "mode":"a", # 'r', 'r+', 'a' and 'w'
            "format":"table", # "fixed" can't be indexed w/ data_columns
            "complib":"blosc:snappy",
            "complevel":2,
            # "data_columns":["ievt"] # used for pytables' fast HDF5 dataset indexing
            }

        df_events.to_hdf(hfile, key="events", **opts)
        df_hits.to_hdf(hfile, key="hits", **opts)

    # -- write to awkward.hdf5 --
    elif mode=="awkward":
        print("Writing awkward hdf5.\n  input:{}\n  output:{}".format(infile, hfile))
        print("Warning: this mode is not well-developed and needs work")

        # FIXME: separate values, as above
        uarrs = utree.arrays(entrystop=nevt)

        # set awkward hdf5 options
        opts = {
            # "compression":2 # hmm, doesn't work?
        }
        with h5py.File(os.path.expanduser(hfile), "w") as hf:
            awk_h5 = awkward.hdf5(hf, **opts)
            for ds in uarrs:
                if isinstance(uarrs[ds], awkward.ObjectArray):
                    print("skipping dataset:", ds.decode('utf-8'))
                    continue
                awk_h5[ds.decode('utf-8')] = uarrs[ds]

        # ehhh, it's a work in progress.  probably won't need this until LEGEND

    # check the groups saved into the file
    with pd.HDFStore(hfile, 'r') as f:
        print("Keys:", f.keys())


def write_waveforms(run, infile, wf_mode="root", compression=None):
    """
    retrieve MGTWaveform objects & add them as a group to an HDF5 file.
    options:
    - wf_mode: root, uproot (modes for accessing a built file)
    - skim: T/F (access multiple built files & retrieve specific wfs)
    - compression: dvi ("diff-var-int")
    TODO:
    - cythonize compression functions (right now they're too slow to be useful)
    - apply nonlinearity correction (requires NLC input files)
    - try out david's compression algorithm
    """
    from ROOT import GATDataSet
    gds = GATDataSet()
    bfiles = {} # {run1:file1, ...}
    skim_mode = False

    # declare inputs and outputs
    if run is not None:
        bfile = gds.GetPathToRun(run, GATDataSet.kBuilt)
        infile, tname = bfile, "MGTree"
        bfiles[run] = bfile

    if infile is not None:
        if "skim" not in infile:
            bfiles[run] = infile
        else:
            skim_mode = True
            ufile = uproot.open(infile)
            utree = ufile["skimTree"]
            runs = utree["run"].array()
            run_set = sorted(list(set(runs)))
            for r in run_set:
                bfiles[r] = gds.GetPathToRun(int(r), GATDataSet.kBuilt)

    # create new output file (should match write_HDF5)
    if run is None:
        label = infile.split("/")[-1].split(".root")[0]
        hfile = "{}/hdf5/{}.h5".format(os.environ["MJDDATADIR"], label)
    else:
        hfile = "{}/hdf5/mjd_run{}.h5".format(os.environ["MJDDATADIR"], run)

    print("Saving waveforms.\n  input: {}\n  output: {}".format(infile, hfile))

    # -- create "skim --> built" lookup dataframe to grab waveforms --
    # (assumes pandas hdf5)
    df_evts = pd.read_hdf(hfile, key='events')
    df_hits = pd.read_hdf(hfile, key='hits')
    if skim_mode:
        df1 = df_evts[["run", "mH", "iEvent"]]
        df2 = df_hits[["iHit", "channel"]]
        tmp = df1.align(df2, axis=0)
        df_skim = pd.concat(tmp, axis=1)
    else:
        tmp = df_evts.align(df_hits, axis=0)
        df_skim = pd.concat(tmp, axis=1)

    # -- loop over entries and pull waveforms --

    wf_list = [] # fill this with ndarrays
    isave = 10000 # how often to write to the hdf5 output file
    nevt = df_skim.shape[0] # total entries
    prev_run = 0
    print("Scanning {} entries.  Skim mode? {}".format(nevt, skim_mode))

    for idx, ((entry, subentry), row) in enumerate(df_skim.iterrows()):

        if idx % 1000 == 0:
            update_progress(float(idx/nevt))
        if idx == nevt-1:
            update_progress(1)

        iE = int(row["iEvent"]) if skim_mode else entry
        iH = int(row["iHit"]) if skim_mode else subentry
        run = int(row["run"])

        # detect run boundaries and load built files
        if run != prev_run:
            if wf_mode == "root":
                from ROOT import TFile, TTree, MGTWaveform, MJTMSWaveform
                tf = TFile(bfiles[run])
                tt = tf.Get("MGTree")
                tt.GetEntry(0)
                is_ms = tt.run.GetUseMultisampling()
            elif wf_mode == "uproot":
                ufile = uproot.open(bfiles[run])
                utree = ufile["MGTree"]
            prev_run = run

        # get waveform with ROOT
        if wf_mode == "root":
            tt.GetEntry(iE)
            if is_ms:
                wfdown = tt.event.GetWaveform(iH) # downsampled
                wffull = tt.event.GetAuxWaveform(iH) # fully sampled
                wf = MJTMSWaveform(wfdown, wffull)
            else:
                wf = tt.event.GetWaveform(iH)

            wfarr = np.asarray(wf.GetVectorData()) # fastest dump to ndarray

            if compression == "dvi":
                # not quite ready.  need to cythonize this
                # will also need to do more work to shrink the #cols in wfdf
                wfarr = compress_dvi(wfarr)

        # get waveform with uproot
        elif wf_mode == "uproot":
            print("direct uproot wf retrieval isn't supported yet.  Exiting ...")
            exit()

        # add the wf to the list
        wf_list.append(wfarr)

        # periodically write to the hdf5 output file to relieve memory pressure.
        # this runs compression algorithms so we don't want to do it too often.
        if (idx % isave == 0 and idx!=0) or idx == nevt-1:

            print("Saving waveforms ...")
            wfdf = pd.DataFrame(wf_list) # empty vals are NaN when rows are jagged

            opts = {
                "mode":"a",
                "format":"table",
                "complib":"blosc:snappy",
                "complevel":1,
                }
            wfdf.to_hdf(hfile, key="waves", mode="a", append=True)

            # we might want this if we implement an awkward hdf5 mode
            # jag = awkward.JaggedArray.fromiter(wf_list)
            # print(jag.shape)

            # reset the wf list
            wf_list = []


def read_pandas(run, infile):
    """
    reads files created by write_HDF5.  also gives a couple examples
    of aligning ('friending') and skimming DataFrames of different shapes.
    I think all Majorana DataFrames should use "tables" mode because it
    allows appending to an active file, and HDF5 indexing for fast lookups
    """
    if infile is None:
        hfile = "{}/hdf5/mjd_run{}.h5".format(os.environ["MJDDATADIR"], run)
    else:
        hfile = infile

    with pd.HDFStore(hfile,'r') as f:
        keys = f.keys()

    print("Reading file: {}\nFound keys: {}".format(hfile, keys))

    df_evts = pd.read_hdf(hfile, key="events") # event-level data
    df_hits = pd.read_hdf(hfile, key="hits")   # hit-level data
    df_waves = pd.read_hdf(hfile, key="waves") # waveform data
    df_waves.reset_index(inplace=True) # required step -- fix hdf5 "append"

    print(df_evts)
    print(df_hits)
    print(df_waves)
    # print(df_evts.columns)

    # align skim file dataframes to create a "lookup list"
    if infile is not None and "skim" in infile:
        df1 = df_evts[["run", "mH", "iEvent"]]
        df2 = df_hits[["iHit", "channel"]]
        tmp = df1.align(df2, axis=0) # returns a tuple
        df_skim = pd.concat(tmp, axis=1)
        # df_skim = pd.merge(df_evts, df_hits) # doesn't work, no common columns
    else:
        # align gatified dataframes
        tmp = df_evts.align(df_hits, axis=0)
        df_skim = pd.concat(tmp, axis=1)


def read_awkward(run):
    """
    build the awkward hdf5 file into pd.DataFrames, matching read_pandas.
    NOTE:
    this route is necessary when we want load RAW data where we can't enforce
    that everything must use pandas.  This is how LEGEND tier0 data will be.
    For Majorana, nobody will care about this and we can just use pytables.
    """
    mjdir = os.environ["MJDDATADIR"]
    hfile = "{}/hdf5/mjd_run{}.h5".format(mjdir, run)

    # # method 1 -- read in with h5py & awkward (requires a loop to convert to DF)
    # h5arrs = {}
    # with h5py.File(os.path.expanduser(file)) as hf:
    #     ah5 = awkward.hdf5(hf)
    #     for ds in ah5:
    #         h5arrs[ds] = ah5[ds]

    # # method 2 -- try reading in with pd.HDFStore, instead of h5py+loop,
    # # since you want to convert to pandas anyway
    # with pd.HDFStore(hfile, 'r') as store:
    #     print("hi clint")


def compress_dvi(wfarr):
    """
    run diff-var-int compression on a wf. corresponds to "writeVLSigned"
    - MGTWaveform: https://github.com/mppmu/MGDO/blob/master/Root/MGTWaveform.cc
    - zigzag encoding: https://gist.github.com/mfuerstenau/ba870a29e16536fdbaba
      [signed]=[encoded] : 0=0, -1=1, 1=2, -2=3, 2=4, -3=5, 3=6 ....

    TODO:
    this requires a loop over the array, which is way too slow in python.
    move the compression to a library function and cythonize it.
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
    from pygama.utils import sizeof_fmt
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
