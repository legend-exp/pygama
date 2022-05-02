import numpy as np


def get_element_type(obj):
    """Get the lgdo element type of a scalar or array

    For use in lgdo datatype attributes

    Parameters
    ----------
    obj : any python object
        if a str, will automatically return "string"
        if the object has a dtype, that will be used for determining the element type
        otherwise will attempt to case the type of the object to a dtype

    Returns
    -------
    el_type : str
        A string stating the determined element type of the object.
    """

    # special handling for strings
    if isinstance(obj, str): return 'string'

    # the rest use dtypes
    dt = obj.dtype if hasattr(obj, 'dtype') else np.dtype(type(obj))
    kind = dt.kind

    if kind == 'b': return 'bool'
    if kind == 'V': return 'blob'
    if kind in ['i', 'u', 'f']: return 'real'
    if kind == 'c': return 'complex'
    if kind in ['S', 'U']: return 'string'

    # couldn't figure it out
    print('Cannot determine lgdo element_type for object of type', type(obj).__name__)
    return None


def parse_datatype(datatype):
    """Parse datatype string and return type, dims, elements

    Parameters
    ----------
    datatype : str
        a lgdo-formatted datatype string

    Returns
    -------
    (type, dims, elements) : tuple
        type : str
            the datatype name
        dims : tuple(ints) or None
            if not None, a tuple of dimensions for the lgdo. Note this is
            not the same as the np shape of the underlying data object. See the
            lgdo specification for more information. Also see
            ArrayOfEqualSizedArrays and LH5Store.read_object() for example code
        elements: str or list of str's
            for numeric objects, the element type
            for struct-like  objects, the list of fields in the struct
    """
    if '{' not in datatype: return 'scalar', None, datatype

    # for other datatypes, need to parse the datatype string
    from parse import parse
    datatype, element_description = parse('{}{{{}}}', datatype)
    if datatype.endswith('>'):
        datatype, dims = parse('{}<{}>', datatype)
        dims = [int(i) for i in dims.split(',')]
        return datatype, tuple(dims), element_description
    else: return datatype, None, element_description.split(',')
