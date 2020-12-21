from .array import Array

class FixedSizeArray(Array):
    """
    An array of fixed-size arrays

    Arrays with guaranteed shape along axes > 0: for example, an array of
    vectors will always length 3 on axis 1, and it will never change from
    application to application.  This data type is used for optimized memory
    handling on some platforms. We are not that sophisticated so we are just
    storing this identification for .lh5 validity, i.e. for now this class is
    just an alias.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
