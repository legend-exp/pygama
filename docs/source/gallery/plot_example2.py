"""
Example 2
=========

This is example 2.
For more information, also see
:ref:`sphx_glr_gallery_plot_example1.py`.
"""

import sys
sys.path.append('../mymodule')
import module

import matplotlib.pyplot as plt

m = module.MyClass()
m.set_attribute("yumulu")
plt.xlabel(m.get_attribute())
module.myfunction("yada da")
plt.show()
