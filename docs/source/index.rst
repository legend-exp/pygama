Welcome to pygama's documentation!
==================================

*pygama* is a Python package for:

* converting physics data acquisition system output to LEGEND LH5-format HDF5 files
* performing bulk digital signal processing on time-series data
* optimizing those digital signal processing (DSP) routines and tuning associated analysis parameters
* generating and selecting high-level event data for further analysis

Getting started
---------------

*pygama* is published on the `Python Package Index
<https://pypi.org/project/pygama>`_. Install on local systems with `pip
<https://pip.pypa.io/en/stable/getting-started>`_:

.. tab:: Stable release

    .. code-block:: console

        $ pip install pygama

.. tab:: Unstable (``main`` branch)

    .. code-block:: console

        $ pip install pygama@git+https://github.com/legend-exp/pygama@main

.. tab:: Linux Containers

    Get a LEGEND container with *pygama* pre-installed on `Docker hub
    <https://hub.docker.com/r/legendexp/legend-software>`_ or follow
    instructions on the `LEGEND wiki
    <https://legend-exp.atlassian.net/l/cp/nF1ww5KH>`_.

If you plan to develop *pygama*, refer to the :doc:`developer's guide
<developer>`.

.. attention::

    If installing in a user directory (typically when invoking pip as a normal
    user), make sure ``~/.local/bin`` is appended to ``PATH``. The ``pygama``
    executable is installed there.

Next steps
----------

.. toctree::
   :maxdepth: 1

   Package API reference <api/modules>

.. toctree::
   :maxdepth: 1
   :caption: Related projects

   LEGEND Data Objects <https://legend-pydataobj.readthedocs.io>
   Decoding Digitizer Data <https://legend-daq2lh5.readthedocs.io>
   Digital Signal Processing <https://dspeed.readthedocs.io>

.. toctree::
   :maxdepth: 1
   :caption: Development

   Source Code <https://github.com/legend-exp/pygama>
   License <https://github.com/legend-exp/pygama/blob/main/LICENSE>
   Changelog <https://github.com/legend-exp/pygama/releases>
   developer
