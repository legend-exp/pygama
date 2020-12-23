import pandas as pd
from .struct import Struct
from .array import Array
from .fsarray import Array


class Table(Struct):
    """
    A special struct of array or subtable 'columns' of equal length.
    """
    
    def __init__(self, size=None, col_dict={}, attrs={}):
        # TODO: overload getattr to allow access to fields as object attributes?
        
        # if col_dict is not empty, its contents will be used directly in
        # the Table (not copied)
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


    def __len__(self):
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
        """Get a dataframe containing each of the columns given. If no columns
        are given, get include all fields as columns."""
        df = pd.DataFrame(copy=copy)
        if len(cols)==0:
            for col, dat in self.items():
                df[col] = dat.nda
        else:
            for col in cols:
                df[col] = self[col].nda
        return df

