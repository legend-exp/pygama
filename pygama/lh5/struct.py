class Struct(dict):
    """
    A dictionary with an optional set of attributes.

    After instantiation, add fields using add_field() to keep datatype updated,
    or call update_datatype() after adding.
    """
    # TODO: overload setattr to require add_field for setting?


    def __init__(self, obj_dict={}, attrs={}):
        """
        Parameters
        ----------
        obj_dict : dict (optional)
            Instantiate this struct using the supplied named lh5 objects.
            Note: no copy is performed, the objects are used directly.
        attrs : dict (optional)
            A set of user attributes to be carried along with this lh5 object
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
        """The name for this object's lh5 datatype attribute"""
        return 'struct'


    def form_datatype(self):
        """Return this object's lh5 datatype attribute string"""
        return self.datatype_name() + '{' + ','.join(self.keys()) + '}'


    def update_datatype(self):
        self.attrs['datatype'] = self.form_datatype()


    def add_field(self, name, obj):
        self[name] = obj
        self.update_datatype()

