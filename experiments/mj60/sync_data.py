#!/usr/bin/env python3
import os
import glob
import json
from datetime import datetime
import subprocess as sp
from pprint import pprint
from pygama.utils import *

def main():
    """
    sync MJ60 data with cenpa-rocks.
    - rsync the entire Data/MJ60 directory using the $DATADIR variable
    - set flags to then remove raw/raw_to_dsp/tier2 files
    Hopefully we can reuse this script for C1.
    """
    global runDB
    with open("runDB.json") as f:
        runDB = json.load(f)

    # run_rsync()
    daq_cleanup()


def run_rsync(test=False):
    """
    run rsync on the entire $DATADIR/MJ60 folder (can take a while ...)
    """
    if "mjcenpa" not in os.environ["USER"]:
        print("Error, we're not on the MJ60 DAQ machine.  Exiting ...")
        exit()

    #raw_dir = runDB["loc_dir"] + "/"
    raw_dir = os.path.expandvars(runDB["loc_dir"] + "/")
    raw_rocks = "{}:{}/".format(runDB["rocks_login"], runDB["rocks_dir"])

    if test:
        cmd = "rsync -avh --dry-run {} {}".format(raw_dir, raw_rocks)
    else:
        cmd = "rsync -avh {} {}".format(raw_dir, raw_rocks)
    sh(cmd)


def daq_cleanup(keep_t1=False, keep_t2=False):
    """
    build a list of files on the DAQ and rocks, check integrity,
    and delete files on the DAQ only if we're sure the transfer was successful.
    MJ60 and C1 ORCA raw files have "BackgroundRun" in the filenames
    """
    if "mjcenpa" not in os.environ["USER"]:
        print("Error, we're not on the MJ60 DAQ machine.  Exiting ...")
        exit()

    # local (DAQ) list
    datadir_loc = os.path.expandvars(runDB["loc_dir"] + "/")
    filelist_loc = glob.glob(datadir_loc + "/**", recursive=True)
    # for f in filelist_loc:
        # print(f)

    # remote list
    args = ['ssh', runDB['rocks_login'], 'ls -R '+runDB["rocks_dir"]]
    ls = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = ls.communicate()
    out = out.decode('utf-8')
    filelist_rocks = out.split("\n")
    filelist_rocks = [f for f in filelist_rocks if ":" not in f and len(f)!=0]
    # for f in filelist_rocks:
        # print(f)

    # make sure all files have successfully transferred
    for f in filelist_loc:
        fname = f.split("/")[-1]
        if len(fname) == 0:
            continue
        if fname not in filelist_rocks:
            print("whoa, ", fname, "not found in remote list!")
            exit()

    print("All files in:\n    {}\nhave been backed up to cenpa-rocks."
          .format(datadir_loc))
    print("It should be OK to delete local files.")

    # don't delete these files, orca needs them
    ignore_list = [".Orca", "RunNumber"]

    # set these bools to not remove the pygama files
    if keep_t1:
        ignore_list.append("t1_run")
    if keep_t2:
        ignore_list.append("t2_run")

    # now delete old files, ask for Y/N confirmation
    print("OK to delete local files? [y/n]")
    if input() in ["y","Y"]:
        for f in filelist_loc:
            f.replace(" ", "\ ")
            if os.path.isfile(f):
                if any(ig in f for ig in ignore_list):
                    continue

                print("Deleting:", f)
                os.remove(f)

    now = datetime.now()
    print("Processing is up to date!", now.strftime("%Y-%m-%d %H:%M"))


def download_rocks():
    """
    fk , i also need to write a function to recall the raw
    files from cenpa-rocks for reprocessing (since for now
    all processing happens on the DAQ computer)
    """
    print("hi clint")


if __name__=="__main__":
    main()
