<img src=".github/logo.png" alt="pygama logo" align="left" height="150">

# pygama

[![PyPI](https://img.shields.io/pypi/v/pygama?logo=pypi)](https://pypi.org/project/pygama/)
[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/legend-exp/pygama?logo=git)](https://github.com/legend-exp/pygama/tags)
[![GitHub Workflow Status](https://img.shields.io/github/checks-status/legend-exp/pygama/main?label=main%20branch&logo=github)](https://github.com/legend-exp/pygama/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://img.shields.io/codecov/c/github/legend-exp/pygama?logo=codecov)](https://app.codecov.io/gh/legend-exp/pygama)
[![GitHub issues](https://img.shields.io/github/issues/legend-exp/pygama?logo=github)](https://github.com/legend-exp/pygama/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/legend-exp/pygama?logo=github)](https://github.com/legend-exp/pygama/pulls)
[![License](https://img.shields.io/github/license/legend-exp/pygama)](https://github.com/legend-exp/pygama/blob/main/LICENSE)
[![Read the Docs](https://img.shields.io/readthedocs/pygama?logo=readthedocs)](https://pygama.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10614246.svg)](https://zenodo.org/doi/10.5281/zenodo.10614246)

*pygama* is a Python package for:

- converting physics data acquisition system output to
  [LH5-format](https://legend-exp.github.io/legend-data-format-specs) HDF5
  files (functionality provided by the
  [legend-pydataobj](https://legend-pydataobj.readthedocs.io) and
  [legend-daq2lh5](https://legend-daq2lh5.readthedocs.io) packages)
- performing bulk digital signal processing (DSP) on time-series data
  (functionality provided by the [dspeed](https://dspeed.readthedocs.io)
  package)
- optimizing DSP routines and tuning associated analysis parameters
- generating and selecting high-level event data for further analysis

Check out the [online documentation](https://pygama.readthedocs.io).

If you are using this software, consider
[citing](https://zenodo.org/doi/10.5281/zenodo.10614246)!

## Related repositories

- [legend-exp/legend-pydataobj](https://github.com/legend-exp/legend-pydataobj)
  → LEGEND Python Data Objects
- [legend-exp/legend-daq2lh5](https://github.com/legend-exp/legend-daq2lh5)
  → Convert digitizer data to LEGEND HDF5
- [legend-exp/dspeed](https://github.com/legend-exp/dspeed)
  → Fast Digital Signal Processing for particle detector signals in Python
