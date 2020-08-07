#! /usr/bin/env python3

import os
import sys
import json
import h5py
import time
import numpy as np
from collections import OrderedDict
from pprint import pprint
import re
import importlib
import git
import argparse
from copy import deepcopy

import pygama
from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.units import *
from pygama.io import lh5


def raw_to_dsp(f_raw, f_dsp, dsp_config, lh5_tables=None, verbose=1,
               n_max=np.inf, overwrite=True, buffer_len=8):
    """
    Uses the ProcessingChain class.
    The list of processors is specifed via a JSON file.
    """
    t_start = time.time()

    if isinstance(dsp_config, str):
        with open(dsp_config, 'r') as config_file:
            dsp_config = json.load(config_file, object_pairs_hook=OrderedDict)
    
    if not isinstance(dsp_config, dict):
        raise Exception('Error, dsp_config must be an dict')

    raw_store = lh5.Store()
    lh5_file = raw_store.gimme_file(f_raw, 'r')

    # if no group is specified, assume we want to decode every table in the file
    if lh5_tables is None:
        lh5_tables = []
        lh5_tables_temp = raw_store.ls(f_raw)

        # sometimes 'raw' is nested, e.g g024/raw
        for tb in lh5_tables_temp:
            if "raw" not in tb:
                tbname = raw_store.ls(lh5_file[tb])[0]
                if "raw" in tbname:
                    tb = tb +'/'+ tbname # g024 + /raw
            lh5_tables.append(tb)
    
    # make sure every group points to waveforms, if not, remove the group
    for tb in lh5_tables:
        if 'raw' not in tb:
            lh5_tables.remove(tb)

    for tb in lh5_tables:
        print('Processing table: ', tb)

        # load primary table
        data_raw = raw_store.read_object(tb, f_raw, start_row=0, n_rows=n_max)
        pc, tb_out = build_processing_chain(data_raw, dsp_config, verbosity=verbose)
        
        print(f'Processing table: {tb} ...')
        pc.execute()
        
        print(f'Done.  Writing to file ...')
        raw_store.write_object(tb_out, tb.replace('/raw', '/dsp'), f_dsp)

    # write processing metadata
    dsp_info = lh5.Struct()
    dsp_info.add_field('timestamp', lh5.Scalar(np.uint64(time.time())))
    dsp_info.add_field('python_version', lh5.Scalar(sys.version))
    dsp_info.add_field('numpy_version', lh5.Scalar(np.version.version))
    dsp_info.add_field('h5py_version', lh5.Scalar(h5py.version.version))
    dsp_info.add_field('hdf5_version', lh5.Scalar(h5py.version.hdf5_version))
    dsp_info.add_field('pygama_version', lh5.Scalar(pygama.__version__))
    repo = git.Repo.init(pygama.__path__[0] + '/..')
    dsp_info.add_field('pygama_branch', lh5.Scalar(str(repo.active_branch)))
    dsp_info.add_field('pygama_commit', lh5.Scalar(repo.head.object.hexsha))
    dsp_info.add_field('dsp_config', lh5.Scalar(json.dumps(dsp_config, indent=2)))
    raw_store.write_object(dsp_info, 'dsp_info', f_dsp)

    t_elap = (time.time() - t_start) / 60
    print(f'Done processing.  Time elapsed: {t_elap:.2f} min.')



def build_processing_chain(lh5_in, dsp_config, outputs = None, verbosity=1,
                           block_width=8):
    """
    Produces a ProcessingChain object and an lh5 table for output parameters
    from an input lh5 table and a json recipe.
    
    Returns (proc_chain, lh5_out):
    - proc_chain: ProcessingChain object that is bound to lh5_in and lh5_out;
      all you need to do is handle file i/o for lh5_in/out and run execute
    - lh5_out: output LH5 table
    
    Required arguments:
    - lh5_in: input LH5 table
    - config: dict or name of json file containing a recipe for
      constructing the ProcessingChain object produced by this function.
      config is formated as a json dict with different processors. Config
      should have a dictionary called processors, containing dictionaries
      of the following format:
        Key: parameter name: name of parameter produced by the processor.
             can optionally provide multiple, separated by spaces
        Values:
          processor (req): name of gufunc
          module (req): name of module in which to find processor
          prereqs (req): name of parameters from other processors and from 
            input that are required to exist to run this
          args (req): list of arguments for processor, with variables passed
            by name or value. Names should either be inputs from lh5_in, or
            parameter names for other processors. Names of the format db.name
            will look up the parameter in the metadata. 
          kwargs (opt): kwargs used when adding processors to proc_chain
          init_args (opt): args used when initializing a processor that has
            static data (for factory functions)
          default (opt): default value for db parameters if not found
          unit (opt): unit to be used for attr in lh5 file.
      There may also be a list called 'outputs', containing a list of parameters
      to put into lh5_out.
    
    Optional keyword arguments:
    - outputs: list of parameters to put in the output lh5 table. If None,
      use the parameters in the 'outputs' list from config
    - verbosity: verbosity level:
            0: Print nothing (except errors...)
            1: Print basic warnings (default)
            2: Print basic debug info
            3: Print friggin' everything!    
    - block_width: number of entries to process at once.
    """
    
    if isinstance(dsp_config, str):
        with open(dsp_config) as f:
            dsp_config = json.load(f)
    else:
        # We don't want to modify the input!
        dsp_config = deepcopy(dsp_config)

    if outputs is None:
        outputs = dsp_config['outputs']

    processors = dsp_config['processors']
    
    # for processors with multiple outputs, add separate entries to the processor list
    for key in list(processors):
        keys = [k for k in re.split(",| ", key) if k!='']
        if len(keys)>1:
            for k in keys:
                processors[k] = key
    
    # Recursive function to crawl through the parameters/processors and get
    # a sequence of unique parameters such that parameters always appear after
    # their dependencies. For parameters that are not produced by the ProcChain
    # (i.e. input/db parameters), add them to the list of leafs
    # https://www.electricmonk.nl/docs/dependency_resolving_algorithm/dependency_resolving_algorithm.html
    def resolve_dependencies(par, resolved, leafs, unresolved=[]):
        if par in resolved:
            return
        elif par in unresolved:
            raise Exception('Circular references detected: %s -> %s' % (par, edge))

        # if we don't find a node, this is a leaf
        node = processors.get(par)
        if node is None:
            if par not in leafs:
                leafs.append(par)
            return

        # if it's a string, that means it is part of a processor that returns multiple outputs (see above); in that case, node is a str pointing to the actual node we want
        if isinstance(node, str):
            resolve_dependencies(node, resolved, leafs, unresolved)
            return
        
        edges = node['prereqs']
        unresolved.append(par)
        for edge in edges:
            resolve_dependencies(edge, resolved, leafs, unresolved)
        resolved.append(par)
        unresolved.remove(par)

    proc_par_list = [] # calculated from processors
    input_par_list = [] # input from file and used for processors
    copy_par_list = [] # copied from input to output
    out_par_list = []
    for out_par in outputs:
        if out_par not in processors:
            copy_par_list.append(out_par)
        else:
            resolve_dependencies(out_par, proc_par_list, input_par_list)
            out_par_list.append(out_par)
    proc_chain = ProcessingChain(block_width, verbosity = verbosity)
    
    # Now add all of the input buffers from lh5_in (and also the clk time)
    for input_par in input_par_list:
        buf_in = lh5_in.get(input_par)
        if buf_in is None:
            print("I don't know what to do with " + input_par + ". Building output without it!")
        elif isinstance(buf_in, lh5.Array):
            proc_chain.add_input_buffer(input_par, buf_in.nda)
        elif isinstance(buf_in, lh5.Table):
            # check if this is waveform
            if 't0' and 'dt' and 'values' in buf_in:
                proc_chain.add_input_buffer(input_par, buf_in['values'].nda, 'float32')
                clk = buf_in['dt'].nda[0] * unit_parser.parse_unit(lh5_in['waveform']['dt'].attrs['units'])
                if proc_chain._clk is not None and proc_chain._clk != clk:
                    print("Somehow you managed to set multiple clock frequencies...Using " + str(proc_chain._clk))
                else:
                    proc_chain._clk = clk

    # now add the processors
    for proc_par in proc_par_list:
        recipe = processors[proc_par]
        module = importlib.import_module(recipe['module'])
        func = getattr(module, recipe['function'])
        args = recipe['args']
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg[0:3]=='db.':
                #TODO: ADD METADATA LOOKUP!
                args[i] = recipe['defaults'][arg]
            
        kwargs = recipe.get('kwargs', {}) # might also need metadata lookup here
        # if init_args are defined, parse any strings and then call func
        # as a factory/constructor function
        try:
            init_args = recipe['init_args']
            for i, arg in enumerate(init_args):
                if isinstance(arg, str):
                    if arg[0:3]=='db.':
                        #TODO: ADD METADATA LOOKUP!
                        init_args[i] = recipe['defaults'][arg]
                    else:
                        # see if string can be parsed by proc_chain
                        try:
                            init_args[i] = proc_chain.get_variable(arg)
                        except:
                            pass
            if(verbosity>1):
                print("Building function", func.__name__, "from init_args", init_args)
            func = func(*init_args)
        except:
            pass
        proc_chain.add_processor(func, *args, **kwargs)

    
    # build the output buffers
    lh5_out = lh5.Table(size=proc_chain._buffer_len)
    
    # add inputs that are directly copied
    for copy_par in copy_par_list:
        buf_in = lh5_in.get(copy_par)
        if buf_in is None:
            print("I don't know what to do with " + input_par + ". Building output without it!")
        elif isinstance(buf_in, lh5.Array):
            print("Copying", copy_par, "to lh5_out")
            lh5_out.add_field(copy_par, buf_in)
    
    # finally, add the output buffers to lh5_out and the proc chain
    for out_par in out_par_list:
        recipe = processors[out_par]
        # special case for proc with multiple outputs
        if isinstance(recipe, str):
            i = [k for k in re.split(",| ", recipe) if k!=''].index(out_par)
            recipe = processors[recipe]
            unit = recipe['unit'][i]
        else:
            unit = recipe['unit']
        
        try:
            scale = convert(1, unit_parser.parse_unit(unit), clk)
        except InvalidConversion:
            scale = None
        
        buf_out = proc_chain.get_output_buffer(out_par, unit=scale)
        lh5_out.add_field(out_par, lh5.Array(buf_out, attrs={"units":unit}) )
    return (proc_chain, lh5_out)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
"""Process a single tier 1 LH5 file and produce a tier 2 LH5 file using a
json config file and raw_to_dsp.""")
    
    arg = parser.add_argument
    arg('file', help="Input (tier 1) LH5 file.")
    arg('-o', '--output',
        help="Name of output file. By default, output to ./t2_[input file name].")
    
    arg('-v', '--verbose', default=2, type=int,
        help="Verbosity level: 0=silent, 1=basic warnings, 2=verbose output, 3=debug. Default is 2.")
    
    arg('-b', '--block', default=8, type=int,
        help="Number of waveforms to process simultaneously. Default is 8")
    
    arg('-c', '--chunk', default=256, type=int,
        help="Number of waveforms to read from disk at a time. Default is 256. THIS IS NOT IMPLEMENTED YET!")
    arg('-n', '--nevents', default=None, type=int,
        help="Number of waveforms to process. By default do the whole file")
    arg('-g', '--group', default='',
        help="Name of group in LH5 file. By default process all base groups. Supports wildcards.")
    defaultconfig = os.path.dirname(os.path.realpath(__loader__.get_filename())) + '/dsp_config.json'
    arg('-j', '--jsonconfig', default=defaultconfig, type=str,
        help="Name of json file used by raw_to_dsp to construct the processing routines used. By default use dsp_config in pygama/apps.")
    arg('-p', '--outpar', default=None, action='append', type=str,
        help="Add outpar to list of parameters written to file. By default use the list defined in outputs list in config json file.")
    arg('-r', '--recreate', action='store_const', const=0, dest='writemode',
        help="Overwrite file if it already exists. Default option. Multually exclusive with --update and --append")
    arg('-u', '--update', action='store_const', const=1, dest='writemode',
        help="Update existing file with new values. Useful with the --outpar option. Mutually exclusive with --recreate and --append THIS IS NOT IMPLEMENTED YET!")
    arg('-a', '--append', action='store_const', const=1, dest='writemode',
        help="Append values to existing file. Mutually exclusive with --recreate and --update THIS IS NOT IMPLEMENTED YET!")
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = 't2_'+args.file[args.file.rfind('/')+1:].replace('t1_', '')
    
    raw_to_dsp(args.file, out, args.jsonconfig, lh5_tables=None, verbose=args.verbose, n_max=args.nevents, overwrite=args.writemode==0, buffer_len=args.block)
