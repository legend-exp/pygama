from .array import Array


class FixedSizeArray(Array):
    """
    An array of fixed-size arrays

    Arrays with guaranteed shape along axes > 0: for example, an array of
    vectors will always length 3 on axis 1, and it will never change from
    application to application.  This data type is used for optimized memory
    handling on some platforms. We are not that sophisticated so we are just
    storing this identification for lgdo validity, i.e. for now this class is
    just an alias for Array, but keeps track of the datatype name.
    """


    def __init__(self, nda=None, shape=(), dtype=None, fill_val=None, attrs={}):
        """ See Array.__init__ for optional args """
        super().__init__(nda=nda, shape=shape, dtype=dtype, fill_val=fill_val, attrs=attrs)


    def datatype_name(self):
        """The name for this object's lh5 datatype attribute"""
        return 'fixedsize_array'
