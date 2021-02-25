## pygama.dsp

Digital signal processing routines for waveform blocks.  Generally we have `Calculators`, which take a block and return a column (single-valued), and `Transforms`, which take a block and return another block.

Pygama to-do list:
https://docs.google.com/document/d/1ecOSJbIfC8p4OtYX3IngcsMnGxi7kOgaPVxwUFVHxGE/edit?usp=sharing

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

