import os, glob
import numpy as np
import pandas as pd


def get_lh5_element_type(obj):
    """Get the lh5 element type of a scalar or array

    For use in the datatype attribute of lh5 objects

    Parameters
    ----------
    obj : str or any object with a numpy dtype

    Returns
    -------
    el_type : str
        A string stating the determined element type of the object.
    """
    if isinstance(obj, str): return 'string'
    if hasattr(obj, 'dtype'):
        kind = obj.dtype.kind
        if kind == '?' or obj.dtype.name == 'bool': return 'bool'
        #FIXME: pygama will call all uint8's "blobs" by this logic...
        if kind in ['b', 'B', 'V']: return 'blob'
        if kind in ['i', 'u', 'f']: return 'real'
        if kind == 'c': return 'complex'
        if kind in ['S', 'a', 'U']: return 'string'
    print('Cannot determine lh5 element_type for object of type', type(obj).__name__)
    return None


def parse_datatype(datatype):
    """Parse datatype string and return type, dims, elements

    Parameters
    ----------
    datatype : str
        a lh5-formatted datatype string

    Returns
    -------
    (type, dims, elements) : tuple
        type : str
            the datatype name
        dims : tuple(ints) or None
            if not None, a tuple of dimensions for the lh5 object. Note this is
            not the same as the np shape of the underlying data object. See the
            lh5 specification for more information. Also see
            ArrayOfEqualSizedArrays and Store.read_object() for example code
        elements: str or list of str's
            for numeric objects, the element type
            for struct-like  objects, the list of fields in the struct
    """
    if '{' not in datatype: return 'scalar', (), datatype

    # for other datatypes, need to parse the datatype string
    from parse import parse
    datatype, element_description = parse('{}{{{}}}', datatype)
    if datatype.endswith('>'): 
        datatype, dims = parse('{}<{}>', datatype)
        dims = [int(i) for i in dims.split(',')]
        return datatype, tuple(dims), element_description
    else: return datatype, None, element_description.split(',')


def load_nda(f_list, par_list, tb_in=''):
    """ Build a dictionary of ndarrays from lh5 data

    Given a list of files, a list of lh5 table parameters, and an optional group
    path, return a numpy array with all values for each parameter.

    Parameters
    ----------
    f_list : str or list of str's
        A list of files. Can contain wildcards
    par_list : list of str's
        A list of parameters to read from each file
    tb_in : str (optional)
        Optional group path within which to find the specified parameters

    Returns
    -------
    par_data : dict
        A dictionary of the parameter data keyed by the elements of par_list.
        Each entry contains the data for the specified parameter concatenated
        over all files in f_list
    """
    if isinstance(f_list, str): f_list = [f_list]
    # Expand wildcards
    f_list = [f for f_wc in f_list for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    sto = Store()
    par_data = {par : [] for par in par_list}
    for f in f_list:
        for par in par_list:
            data, _ = sto.read_object(f'{tb_in}/{par}', f)
            if not data: continue
            par_data[par].append(data.nda)
    par_data = {par : np.concatenate(par_data[par]) for par in par_list}
    return par_data


def load_dfs(f_list, par_list, tb_in=''):
    """ Build a pandas dataframe from lh5 data

    Given a list of files (can use wildcards), a list of lh5 columns, and
    optionally the group path, return a pandas DataFrame with all values for
    each parameter.

    Parameters
    ----------
    See load_nda for parameter specification

    Returns
    -------
    df : pandas.DataFrame
        Contains columns for each parameter in par_list, and rows containing all
        data for the associated parameters concatenated over all files in f_list
    """
    return pd.DataFrame( load_nda(f_list, par_list, tb_in) )
