import pandas as pd
from .struct import Struct


class Table(Struct):
    """
    A special struct of array or subtable "columns" of equal length.

    Holds onto an internal read/write location "loc" that is useful in managing
    table I/O using functions like push_row(), is_full(), and clear()
    """
    # TODO: overload getattr to allow access to fields as object attributes?


    def __init__(self, size=None, col_dict={}, attrs={}):
        """
        Parameters
        ----------
        size : int (optional)
            Sets the number of rows in the table. Arrays in col_dict will be
            resized to match size if both are not None.  If size is left as
            None, the number of table rows is determined from the length of the
            first array in col_dict. If neither is provided, a default length of
            1024 is used.
        col_dict : dict (optional)
            Instantiate this table using the supplied named lh5 array-like
            objects.  
            Note 1: no copy is performed, the objects are used directly.
            Note 2: if size is not None, all arrays will be resized to match it
            Note 3: if the arrays have different lengths, all will be resized to
            match the length of the first array
        attrs : dict (optional)
            A set of user attributes to be carried along with this lh5 object

        Initialization
        --------------
        self.loc is initialized to 0
        """
        super().__init__(obj_dict=col_dict, attrs=attrs)

        # if col_dict is not empty, set size according to it
        # if size is also supplied, resize all fields to match it
        # otherwise, warn if the supplied fields have varying size
        if len(col_dict) > 0: 
            do_warn = True if size is None else False
            self.resize(new_size=size, do_warn=do_warn)

        # if no col_dict, just set the size (default to 1024)
        else: self.size = size if size is not None else 1024

        # always start at loc=0
        self.loc = 0


    def datatype_name(self): 
        """The name for this object's lh5 datatype attribute"""
        return 'table'


    def __len__(self):
        """Provides __len__ for this array-like class"""
        return self.size


    def resize(self, new_size = None, do_warn = False):
        # if new_size = None, use the size from the first field
        for field, obj in self.items():
            if new_size is None: new_size = len(obj)
            elif len(obj) != new_size:
                if do_warn:
                    print('warning: resizing field', field, 
                          'with size', len(obj), '!=', new_size)
                obj.resize(new_size)
        self.size = new_size


    def push_row(self):
        self.loc += 1


    def is_full(self):
        return self.loc >= self.size


    def clear(self):
        self.loc = 0


    def add_field(self, name, obj, use_obj_size=False, do_warn=True):
        if not hasattr(obj, '__len__'):
            print('Table: Error: cannot add field of type', type(obj).__name__)
            return
        super().add_field(name, obj)

        # check / update sizes
        if self.size != len(obj):
            new_size = len(obj) if use_obj_size else self.size
            self.resize(new_size, do_warn)


    def get_dataframe(self, *cols, copy=False):
        """Get a pandase dataframe containing each of the columns given. If no
        columns are given, get include all fields as columns."""
        df = pd.DataFrame(copy=copy)
        if len(cols)==0:
            for col, dat in self.items():
                df[col] = dat.nda
        else:
            for col in cols:
                df[col] = self[col].nda
        return df

