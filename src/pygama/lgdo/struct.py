class Struct(dict):
    """
    A dictionary of lgdo's with an optional set of attributes.

    After instantiation, add fields using add_field() to keep datatype updated,
    or call update_datatype() after adding.
    """
    # TODO: overload setattr to require add_field for setting?


    def __init__(self, obj_dict={}, attrs={}):
        """
        Parameters
        ----------
        obj_dict : dict { str : lgdo } (optional)
            Instantiate this struct using the supplied named lgdo's.
            Note: no copy is performed, the objects are used directly.
        attrs : dict (optional)
            A set of user attributes to be carried along with this lgdo
        """
        self.update(obj_dict)
        self.attrs = dict(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print(type(self).__name__ + ': Warning: datatype does not match obj_dict!')
                print('datatype: ', self.attrs['datatype'])
                print('obj_dict.keys(): ', obj_dict.keys())
                print('form_datatype(): ', self.form_datatype())
                print('will be updated to the latter.')
        self.update_datatype()


    def datatype_name(self):
        """The name for this lgdo's datatype attribute"""
        return 'struct'


    def form_datatype(self):
        """Return this lgdo's datatype attribute string"""
        return self.datatype_name() + '{' + ','.join(self.keys()) + '}'


    def update_datatype(self):
        self.attrs['datatype'] = self.form_datatype()


    def add_field(self, name, obj):
        self[name] = obj
        self.update_datatype()


    def __len__(self):
        """Structs are considered length=1 """
        return 1

    def __str__(self):
        """Convert to string (e.g. for printing)"""
        tmp_attrs = self.attrs.copy()
        datatype = tmp_attrs.pop('datatype')
        string = datatype + " = " + super().__repr__() #__repr__ instead of __str__ to avoid infinite loop
        if len(tmp_attrs) > 0: string += '\nattrs = ' + str(tmp_attrs)
        return string

    def __repr__(self):
        return str(self)
