"""
This is mymodule. It serves as an minimal working example
on how to use sphinx and sphinx gallery. 

Requirements: Sphinx, sphinx-gallery, matplotlib
"""

class MyClass():
    """
    This is MyClass. It has an `attribute`. 
    And getter and setter `MyClass.get_attribute` and 
    `MyClass.set_attribute`.
    """   
    def __init__(self, **kwargs):
        """
        The init function docstring.
        """
        self.attribute = "init"
        
    def set_attribute(self, arg):
        """
        Sets the attribute.
        """
        self.attribute = arg
        
    def get_attribute(self):
        """
        Returns the attribute
        """
        return self.attribute
        
        
def myfunction(arg):
    """
    Creates MyClass instance and sets its attribute.
    Then prints it.
    """
    m = MyClass()
    m.set_attribute(arg)
    print(m.get_attribute())

    