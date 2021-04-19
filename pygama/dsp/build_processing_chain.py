import json
import re
import importlib
from copy import deepcopy

from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.units import *
from pygama import lh5


def build_processing_chain(lh5_in, dsp_config, db_dict = None,
                           outputs = None, verbosity=1, block_width=16):
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
    - db_dict: a nested dict pointing to values for db args.
      e.g. if a processor uses arg db.trap.risetime, it will look up
          db_dict['trap']['risetime']
      and use the found value. If no value is found, use the default defined
      in the config file.
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
    elif dsp_config is None:
        dsp_config = {'outputs':[], 'processors':{}}
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

    if verbosity>0:
        print('Processing parameters:', str(proc_par_list))
        print('Required input parameters:', str(input_par_list))
        print('Copied output parameters:', str(copy_par_list))
        print('Processed output parameters:', str(out_par_list))
        
    proc_chain = ProcessingChain(block_width, lh5_in.size, verbosity = verbosity)
    
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
                lookup_path = arg[3:].split('.')
                try:
                    node = db_dict
                    for key in lookup_path:
                        node = node[key]
                    args[i] = node
                    if(verbosity>0):
                        print("Database lookup: found", node, "for", arg)
                except:
                    try:
                        args[i] = recipe['defaults'][arg]
                        if(verbosity>0):
                            print("Database lookup: using default value of", args[i], "for", arg)
                    except:
                        raise Exception('Did not find', arg, 'in database, and could not find default value.')
            
        kwargs = recipe.get('kwargs', {}) # might also need db lookup here
        # if init_args are defined, parse any strings and then call func
        # as a factory/constructor function
        try:
            init_args = recipe['init_args']
            for i, arg in enumerate(init_args):
                if isinstance(arg, str) and arg[0:3]=='db.':
                    lookup_path = arg[3:].split('.')
                    try:
                        node = db_dict
                        for key in lookup_path:
                            node = node[key]
                        init_args[i] = node
                        if(verbosity>0):
                            print("Database lookup: found", node, "for", arg)
                    except:
                        try:
                            init_args[i] = recipe['defaults'][arg]
                            if(verbosity>0):
                                print("Database lookup: using default value of", init_args[i], "for", arg)
                        except:
                            raise Exception('Did not find', arg, 'in database, and could not find default value.')
                    arg = init_args[i]

                # see if string can be parsed by proc_chain
                if isinstance(arg, str):
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
        if isinstance(buf_in, lh5.Array):
            lh5_out.add_field(copy_par, buf_in)
        elif isinstance(buf_in, lh5.Table):
            # check if this is waveform
            if 't0' and 'dt' and 'values' in buf_in:
                lh5_out.add_field(copy_par, buf_in['values'])
                clk = buf_in['dt'].nda[0] * unit_parser.parse_unit(lh5_in['waveform']['dt'].attrs['units'])
                if proc_chain._clk is not None and proc_chain._clk != clk:
                    print("Somehow you managed to set multiple clock frequencies...Using " + str(proc_chain._clk))
                else:
                    proc_chain._clk = clk
        else:
            print("I don't know what to do with " + input_par + ". Building output without it!")
    
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
