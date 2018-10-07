"""
Example 1
=========

This is example 1.
"""

import sys
sys.path.append('../mymodule')
import module

import matplotlib.pyplot as plt

m = module.MyClass()
plt.title(m.get_attribute())
plt.show()
