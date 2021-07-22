# pygama
Python based package for data processing and analysis

## installation
Install on local systems with:
```bash
$ git clone [url]
$ pip install -e pygama
```
Installation at NERSC:
```
pip install -e pygama --user
```
Uninstall: `pip uninstall pygama`

To run pygama at NERSC (and set up JupyterHub), we have additional instructions [at this link](https://github.com/legend-exp/legend/wiki/Computing-Resources-at-NERSC#configuring-jupyter--nersc).  

## overview

`pygama` is a python package for:
* converting physics data acquisition system output to "lh5"-format hdf5 files
* performing bulk digital signal processing on time-series data
* optimizing those DSP routines and tuning associated analysis parameters
* generating and selecting high-level event data for further analysis

The basic steps for a typical analysis are:

1. Convert DAQ output to "raw" lh5 format using [`build_raw`](pygama/raw)
2. Browse the [LEGEND data objects](pygama/lgdo) in the lh5 files to verify its integrity
3. Optimize the DSP parameters using `dsp_optimize.py` in [pargen](pygama/pargen)
4. Run [`build_dsp`](pygama/dsp) on the raw files to generate "dsp" lh5 output
5. Optimize the hit parameters using routines in [pargen](pygama/pargen)
4. Run `build_hit` (or create your own version) to generate hit files from the dsp data
5. Run `hit_to_evt` to generate files with event structures
6. Concatenate / join / filter evt and raw-dsp-hit data to extract the fields you need for higher-level analysis

## testing

pygama testing uses the [pytest](https://pytest.org) framework following the [numpy testing guidelines](https://numpy.org/doc/stable/reference/testing.html#testing-guidelines). To run tests, just enter `pytest` at the command line.
