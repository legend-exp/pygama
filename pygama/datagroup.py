#!/usr/bin/env python3
import os, json
import pandas as pd
from parse import parse
from pprint import pprint

import pygama.utils as pu


class DataGroup:
    """
    Easily look up groups of files according to the LEGEND data convention,
    for processing and metadata organization.  
    
    Source: https://docs.legend-exp.org/index.php/apps/files/?dir=/LEGEND%20Documents/Technical%20Documents/Analysis&fileid=9140#pdfviewer
    """
    def __init__(self, run=None, rhi=None, runlist=None, config=None, mode=None,
                 tslo=None, tshi=None, nfiles=None):
        """
        """
        # DataFrames to store file key attributes
        self.daq_files = None # daq & raw 
        self.hit_files = None # raw, dsp, hit
        self.evt_files = None # evt
        
        # limit number of files (debug, testing)
        self.nfiles = nfiles

        # typical usage: include a JSON config file to set data paths, etc.
        if config is not None:
            self.set_config(os.path.expandvars(config))

        # define runs of interest (integers)
        self.runs = []
        if run!=None and rhi!=None:
            self.runs.extend([r for r in range(int(run), int(rhi+1))])
        elif run!=None:
            self.runs.append(int(run))
        elif runlist!=None:
            self.runs.extend(runlist)


    def set_config(self, config):
        """
        define paths to hit- and event-level data through self.config.
        """
        with open(config) as f:
            self.config = json.load(f)
            
        # this option is how we handle different filesystems, e.g. hades, cenpa
        self.experiment = self.config['experiment']

        # set the data tiers to consider (daq, raw, dsp, hit, evt, tmap ...)
        tiers = []
        if 'hit_dirs' in self.config:
            tiers.extend(self.config['hit_dirs'])
        if 'evt_dirs' in self.config:
            tiers.extend(self.config['evt_dirs'])
        
        # set file paths to each tier
        for tier in tiers:
            key = f'{tier}_dir'
            self.config[key] = os.path.expandvars(self.config[key])

        # load primary run lookup file (runDB)
        f_runDB = f"{self.config['meta_dir']}/{self.config['runDB']}"
        with open(os.path.expandvars(f_runDB)) as f:
            self.runDB = json.load(f)
        
            
    def find_daq_files(self, ft=None):
        """
        Do an os.walk through the daq directory and build a list of files to 
        consider, including any file that matches the file template.
        If self.nfiles is set, we stop after adding this many files to the list.
        
        Finally, we convert to a DataFrame to make sorting by other attributes 
        (e.g. timestamp) easier. 
        
        Add the corresponding 'raw` file name(s) for each daq file, using the template string from the config file. Note that the subsystem label
        "sysn" in the file will be left unfilled, and handled by daq_to_raw
        to write to multiple output files:
            `raw_file` : /base/LPGTA/raw/{sysn}/phy/[prefix]_{sysn}_raw.lh5
        """
        # required: file template string.  can be from argument or config dict
        if ft is None:
            ft = self.config['daq_to_raw']['daq_filename_template']
        else:
            ft = ft
            
        # make sure directory exists
        if not os.path.isdir(self.config['daq_dir']):
            print('DAQ directory not found: ', self.config['daq_dir'])
            exit()

        # list of dictionaries, converted to DataFrame
        daq_files = []
        
        if self.experiment == 'LPGTA':
            
            # look for folders matching these strings
            self.run_labels = [f'run{r:0>4d}' for r in self.runs]
            
            stop_walk = False
            for path, folders, files in os.walk(self.config['daq_dir']):
                
                if any(rl in path for rl in self.run_labels):
                    
                    # get the run number
                    run = [rl for rl in self.run_labels if rl in path][0]
                    run = int(run.split('run')[-1])
                    
                    for f in sorted(files):
                        
                        # only accept files matching our template
                        finfo = parse(ft, f)
                        if finfo is not None:
                            finfo = finfo.named # convert to dict
                            finfo['daq_file'] = path + '/' + f
                            finfo['run'] = run
                            
                            # convert to pd.datetime64 for easier sorting
                            ts = finfo['YYYYmmdd'] + finfo['hhmmss']
                            fmt = '%Y%m%d%H%M%S'
                            dt = pd.to_datetime(ts, format=fmt)
                            finfo['date'] = dt
                            
                            daq_files.append(finfo)
                            
                        if self.nfiles is not None and len(daq_files)==self.nfiles:
                            stop_walk = True
                        if stop_walk:
                            break
                if stop_walk:
                    break

        elif self.experiment == "ORCA":
            
            # here we need to search by the ORCA run number, which is treated
            # here as a "subrun" number
            print('lol tbd')
            

        # convert found files to DataFrame
        self.daq_files = pd.DataFrame(daq_files)

        def get_raw_file(row):
            """
            generate output filenames and paths needed by daq_to_raw.
            subsytem (sysn) is left unspecified
            """
            # map variables to the template
            raw_template = self.config['daq_to_raw']['raw_filename_template']
            sd = pu.SafeDict(row.to_dict())
            raw_file = raw_template.format_map(sd)
            
            # fill in the file path
            mods = self.config['daq_to_raw']['filename_info_mods']['rtp']
            rtp = mods[row['rtp']]
            raw_path = self.config['raw_dir'] + '/{sysn}/' + rtp + '/'
            
            row['raw_file'] = raw_path + raw_file
            
            return row

        self.daq_files = self.daq_files.apply(get_raw_file, axis=1)
        
        return self.daq_files
        

