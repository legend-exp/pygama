import os
import numpy as np
import pandas as pd
import h5py
import fnmatch
import glob

def get_lh5_datatype_name(obj):
    """Get the LH5 datatype name of an LH5 object"""
    if type(obj) == Table: return 'table'
    if type(obj) == Struct: return 'struct'
    if type(obj) == Scalar: return get_lh5_datatype_name(obj.value)
    if np.isscalar(obj): return get_lh5_element_type(obj)
    if type(obj) == Array: return 'array'
    if type(obj) == FixedSizeArray: return 'fixedsize_array'
    if type(obj) == ArrayOfEqualSizedArrays: return 'array_of_equalsized_arrays'
    if type(obj) == VectorOfVectors: return 'array'
    print('Cannot determine LH5 datatype name for object of type', type(obj).__name__)
    return None


def get_lh5_element_type(obj):
    """Get the LH5 element type of a scalar or array"""
    if isinstance(obj, str): return 'string'
    if hasattr(obj, 'dtype'):
        kind = obj.dtype.kind
        if kind == '?' or obj.dtype.name == 'bool': return 'bool'
        #FIXME: pygama will call all uint8's "blobs" by this logic...
        if kind in ['b', 'B', 'V']: return 'blob'
        if kind in ['i', 'u', 'f']: return 'real'
        if kind == 'c': return 'complex'
        if kind in ['S', 'a', 'U']: return 'string'
    print('Cannot determine LH5 element_type for object of type', type(obj).__name__)
    return None


def parse_datatype(datatype):
    """Parse datatype string and return type, shape, elements"""
    if '{' not in datatype: return 'scalar', (), datatype

    # for other datatypes, need to parse the datatype string
    from parse import parse
    datatype, element_description = parse('{}{{{}}}', datatype)
    if datatype.endswith('>'): 
        datatype, dims = parse('{}<{}>', datatype)
        dims = [int(i) for i in dims.split(',')]
        return datatype, tuple(dims), element_description
    else: return datatype, None, element_description.split(',')


def load_nda(f_list, par_list, tb_in):
    """
    given a list of files, a list of LH5 table parameters, and the HDF5 path,
    return a numpy array with all values for each parameter.
    """
    sto = Store()
    par_data = {par : [] for par in par_list}
    for f in f_list:
        for par in par_list:
            data, _ = sto.read_object(f'{tb_in}/{par}', f)
            if not data: continue
            par_data[par].append(data.nda)
    par_data = {par : np.concatenate(par_data[par]) for par in par_list}
    return par_data


def load_dfs(f_list, par_list, tb_in):
    """
    given a list of files (can use wildcards), a list of LH5 columns, and the
    HDF5 path, return a pandas DataFrame with all values for each parameter.
    """
    if isinstance(f_list, str): f_list = [f_list]
    # Expand wildcards
    f_list = [f for f_wc in f_list for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    return pd.DataFrame(load_nda(f_list, par_list, tb_in) )
