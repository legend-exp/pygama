# pygama
Python based package for data processing and analysis

## installation
Install with
```bash
$ git clone [url]
$ pip install -e pygama
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

1. Convert DAQ output to "raw" lh5 format using `daq_to_raw`
2. Browse data in the lh5 files to verify its integrity
3. Run `raw_to_dsp` on the raw files to generate "dsp" lh5 output
  a. Optimize the DSP parameters
  b. Build an analysis parameter database storing the optimized parameters
  c. Re-run with the optimized DSP routines
4. Run `dsp_to_hit` (or create your own version) to generate hit files from the dsp data
5. Run `hit_to_evt` to generate files with event structures
6. Concatenate / join / filter evt and raw-dsp-hit data to extract the fields you need for higher-level analysis

These steps are detailed below.

## daq_to_raw

The primary function for DAQ output conversion into raw lh5 files is [`daq_to_raw` in `pygama.io`](pygama/io/daq_to_raw.py#L16). This is a one-to many function: one input DAQ file can generate one or more output raw files. Control of which data ends up in which files, and in which hdf5 groups inside of each file, is controlled via [channel groups](pygama/io/ch_group.py). If no `ch_group` is specified, all decoded data should be written to a single output file, with all fields from each hardware decoder in their own output table.

Currently we support only the following hardware:
* FlashCams
* SIS3302 read out with [ORCA](https://github.com/unc-enap/Orca)
* GRETINA digitizer (MJD firmware) read out with ORCA

Partial support is in place but requires updating for
* SIS3316 read out with ORCA
* SIS3316 read out with llamaDAQ
* CAEN DT57XX digitizers read out with CoMPASS

(link to daq_to_raw tutorial)

#### daq_to_raw to-do's
* Time coincidence map generation
* Waveform object implementation
* fully implement remaining DAQ loops / hardware
* add consistency / data integrity checks
* documentation
* Unit tests
   * Generate raw file from a "standard" daq file with expected output to screen
   * Check that all expected columns exist, have the right number of rows, and check md5sums of their data
   * Add a few additional lh5 fields for lh5 tests
* Optimization

## lh5 files

Pygama works with "LEGEND HDF5" (lh5) format data. [The lh5 specification can be found here](https://github.com/legend-exp/legend-data-format-specs). Our python implementation is in [pygama.lh5](pygama/lh5).

lh5 files can be browsed easily in python like any [hdf5](https://www.hdfgroup.org/) file using [h5py](https://www.h5py.org/). (link to lh5 browsing tutorial in lh5 readme... LEGEND collaborators can [view J. Detwiler's presentation here](https://indico.legend-exp.org/event/371/contributions/1915/attachments/1167/1696/20200730_PGTProcessing.pdf))

In addition to the standard h5py interface, pygama provides a [WaveformBrowser](pygama/dsp/WaveformBrowser.py). (link to waveform browser tutorial... LEGEND collaborators can [view I. Guinn's presentation here](https://indico.legend-exp.org/event/455/contributions/2368/attachments/1439/2173/201217_WaveformBrowser.pdf))

#### lh5 to-do's
* table joins
* waveform object definition and compression
* fix overwrite
* Flecher32 md5sums
* unit tests


## raw_to_dsp

DSP is performed by extracting a table of raw data including waveforms and passing it to the [ProcessingChain](pygama/dsp/ProcessingChain.py). The primary function for DSP is [`raw_to_dsp`](../master/pygama/io/raw_to_dsp.py).

DSP is controlled via a [json](https://www.json.org)-formatted file that sets up which routines can be run, and which parameters are selected for output. See an example [dsp.json file here](experiments/lpgta/LPGTA_dsp.json). The DSP can refer to a dictionary of "database" values (see analysis parameters database below).

Available processors include all numpy ufuncs as well as [this list of custom processors](pygama/dsp/_processors).

(link to tutorial on dsp)

#### raw_to_dsp to-do's

* SiPM DSP
* Resting baseline and PZ correction in trap filters
* Improved vectorization
* Additional filters
* Unit tests
  * process a standard input file and check the output
  * one unit test for each processor
* Optimization

## analysis parameters database

The DSP and other routines can make use of an analysis parameters database, which is a [json](https://www.json.org)-formatted file read in as a python dictionary. It can be sent to the DSP routines to load optimal parameters for a given channel.

(link to example, tutorial, etc... LEGEND collaborators can [view J. Detwiler's example here](https://indico.legend-exp.org/event/470/contributions/2407/attachments/1456/2193/20210114_PygamaUpdate.pdf))

## dsp optimization

DSP optimzation 

(LEGEND collaborators can [view J. Detwiler's example here](https://indico.legend-exp.org/event/470/contributions/2407/attachments/1456/2193/20210114_PygamaUpdate.pdf))

## parameter tuning

TBD

## dsp_to_hit

TBD

## hit_to_evt

TBD

## automation

TBD
