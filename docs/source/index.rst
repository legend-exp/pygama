Welcome to pygama's documentation!
==================================

`pygama <https://github.com/legend-exp/pygama>`_ is a Python package for:

* converting physics data acquisition system output to LEGEND LH5-format HDF5 files
* performing bulk digital signal processing on time-series data
* optimizing those digital signal processing (DSP) routines and tuning associated analysis parameters
* generating and selecting high-level event data for further analysis

Getting started
---------------

Install on local systems with `pip <https://pip.pypa.io/en/stable/getting-started>`_:

.. code-block:: console

    $ git clone https://github.com/legend-exp/pygama
    $ pip install pygama

.. note::

    If installing in a user directory (typically when invoking pip as a normal
    user), make sure ``~/.local/bin`` is appended to ``PATH``.

If you plan to develop *pygama*, have a look at our :doc:`developer's guide
<developer>`.

Next steps
----------

.. toctree::
   :maxdepth: 1

   manuals/index
   tutorials
   Package API reference <generated/modules>
   developer
