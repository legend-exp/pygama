# pygama.raw

## build_raw

The primary function for data conversion into raw lh5 files is
[`build_raw`](build_raw.py). This is a one-to many function: one input DAQ file
can generate one or more output raw files. Control of which data ends up in
which files, and in which hdf5 groups inside of each file, is controlled via
[channel groups](ch_group.py) (see below). If no `ch_group` is specified, all
decoded data should be written to a single output file, with all fields from
each hardware decoder in their own output table.

Currently we support only the following hardware:
* FlashCams (requires [pyfcutils](https://github.com/legend-exp/pyfcutils))
* SIS3302 read out with [ORCA](https://github.com/unc-enap/Orca)
* GRETINA digitizer (MJD firmware) read out with ORCA

Partial support is in place but requires updating for
* SIS3316 read out with ORCA
* SIS3316 read out with llamaDAQ
* CAEN DT57XX digitizers read out with CoMPASS


#### raw to-do's
* read_chunk()
* build_raw() multiple (sequential) input file conversion with wildcard support
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
