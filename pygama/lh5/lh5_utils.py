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


