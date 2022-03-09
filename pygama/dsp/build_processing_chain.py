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
    
    Parameters
    ----------
    lh5_in : lgdo.Table
        HDF5 table from which raw data is read. At least one row of entries
        should be read in prior to calling this!
    dsp_config: dict or str
        A dict or json filename containing the recipes for computing DSP
        parameter from raw parameters. The format is as follows:
        {
            "outputs" : [ "parnames", ... ] -> list of output parameters
                 to compute by default; see outputs parameter.
            "processors" : {
                 "name1, ..." : { -> names of parameters computed
                      "function" : str -> name of function to call. Function
                           should implement the gufunc interface, a factory
                           function returning a gufunc, or an arbitrary
                           function that can be mapped onto a gufunc
                      "module" : str -> name of module containing function
                      "args" : [ str or numeric, ... ] -> list of names of
                           computed and input parameters or constant values
                           used as inputs to function. Note that outputs
                           should be fed by reference as args! Arguments read
                           from the database are prepended with db.
                      "kwargs" : dict -> keyword arguments for
                           ProcesssingChain.add_processor.
                      "init_args" : [ str or numeric, ... ] -> list of names
                           of computed and input parameters or constant values
                           used to initialize a gufunc via a factory function
                      "unit" : str or [ strs, ... ] -> units for parameters
                      "defaults" : dict -> default value to be used for
                           arguments read from the database
                      "prereqs" : DEPRECATED [ strs, ...] -> list of parameters
                           that must be computed before these can
                 }
    outputs: [str, ...] (optional)
        List of parameters to put in the output lh5 table. If None,
        use the parameters in the 'outputs' list from config
    db_dict: dict (optional)
        A nested dict pointing to values for db args. e.g. if a processor
        uses arg db.trap.risetime, it will look up
          db_dict['trap']['risetime']
        and use the found value. If no value is found, use the default
        defined in the config file.
    verbosity : int (optional)
        0: Print nothing (except errors...)
        1: Print basic warnings (default)
        2: Print basic debug info
        3: Print friggin' everything!    
    block_width : int (optional)
        number of entries to process at once. To optimize performance,
        a multiple of 16 is preferred, but if performance is not an issue
        any value can be used.
    
    Returns
    -------
    (proc_chain, field_mask, lh5_out) : tuple
        proc_chain : ProcessingChain object that is executed
        field_mask : List of input fields that are used
        lh5_out : output lh5 table containing processed values
    """
    proc_chain = ProcessingChain(block_width, lh5_in.size, verbosity = verbosity)
    
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
    
    # prepare the processor list
    multi_out_procs = {}
    for key, node in processors.items():
        # if we have multiple outputs, add each to the processesors list
        keys = [k for k in re.split(",| ", key) if k!='']
        if len(keys)>1:
            for k in keys:
                multi_out_procs[k] = key

        # parse the arguments list for prereqs, if not included explicitly
        if not 'prereqs' in node:
            prereqs = []
            for arg in node['args']:
                if not isinstance(arg, str): continue
                for prereq in proc_chain.get_variable(arg, True):
                    if prereq not in prereqs and prereq not in keys and prereq != 'db':
                        prereqs.append(prereq)
            node['prereqs'] = prereqs

        if verbosity>=2:
            print("Prereqs for", key, "are", node['prereqs'])

    processors.update(multi_out_procs)
    
    # Recursive function to crawl through the parameters/processors and get
    # a sequence of unique parameters such that parameters always appear after
    # their dependencies. For parameters that are not produced by the ProcChain
    # (i.e. input/db parameters), add them to the list of leafs
    # https://www.electricmonk.nl/docs/dependency_resolving_algorithm/dependency_resolving_algorithm.html
    def resolve_dependencies(par, resolved, leafs, unresolved=[]):
        if par in resolved:
            return
        elif par in unresolved:
            raise ProcessingChainError('Circular references detected: %s -> %s' % (par, edge))

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
            if isinstance(arg, str) and 'db.' in arg:
                res = [i.start() for i in re.finditer('db.', arg)]
                for rs in res:
                    first = args[i].find('db.')
                    out = re.findall("[\dA-Za-z_.]*", args[i][first+3:])[0]
                    lookup_path = out.split(".")
                    database_str = f"db.{out}"
                    try:
                        node = db_dict
                        for key in lookup_path:
                            node = node[key]
                        if not isinstance(node, str):
                            node =str(node)
                        args[i] = args[i].replace(database_str, node)
                        if(verbosity>0):
                            print("Database lookup: found", node, "for", database_str)
                    except:
                        try:
                            default_val = recipe['defaults'][database_str]
                            if not isinstance(default_val, str):
                                default_val =str(default_val)
                            args[i] = args[i].replace(database_str, default_val)
                            if(verbosity>0):
                                print("Database lookup: using default value of", default_val, "for", database_str)
                        except:
                            raise Exception('Did not find', database_str, 'in database, and could not find default value.')            
        kwargs = recipe.get('kwargs', {}) # might also need db lookup here
        # if init_args are defined, parse any strings and then call func
        # as a factory/constructor function
        try:
            init_args = recipe['init_args']
            for i, arg in enumerate(init_args):
                if isinstance(arg, str) and 'db.' in arg:
                    res = [i.start() for i in re.finditer('db.', arg)]
                    for rs in res:
                        first = init_args[i].find('db.')
                        out = re.findall("[\dA-Za-z_.]*", init_args[i][first+3:])[0]
                        lookup_path = out.split(".")
                        database_str = f"db.{out}"
                        try:
                            node = db_dict
                            for key in lookup_path:
                                node = node[key]
                            if not isinstance(node, str):
                                node =str(node)
                            init_args[i] = init_args[i].replace(database_str, node)
                            if(verbosity>0):
                                print("Database lookup: found", node, "for", database_str)
                        except:
                            try:
                                default_val = recipe['defaults'][database_str]
                                if not isinstance(default_val, str):
                                    default_val =str(default_val)
                                init_args[i] = init_args[i].replace(database_str, default_val)
                                if(verbosity>0):
                                    print("Database lookup: using default value of", default_val, "for", database_str)
                            except:
                                raise Exception('Did not find', database_str, 'in database, and could not find default value.') 
                    arg = init_args[i]

                # see if string can be parsed by proc_chain
                if isinstance(arg, str):
                    init_args[i] = proc_chain.get_variable(arg)
                    
            if(verbosity>1):
                print("Building function", func.__name__, "from init_args", init_args)
            func = func(*init_args)
        except KeyError:
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

    field_mask = input_par_list + copy_par_list
    return (proc_chain, field_mask, lh5_out)
