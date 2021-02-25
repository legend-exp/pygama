# pygama.lgdo

Pygama works with "LEGEND Data Objects" (lgdo) defined in the [LEGEND data
format specification](https://github.com/legend-exp/legend-data-format-specs).
This submodule serves as the python implementation of that specification. The
general strategy for the implementation is to dress standard python and numpy
objects with an "attr" dictionary holding lgdo metadata, plus some convenience
functions. The basic data object classes are:

* Scalar: typed python scalar. Access data via the "value" attribute
* Array: basic numpy ndarray. Access data via the "nda" attribute. 
* FixedSizeArray: basic numpy ndarray. Access data via the "nda" attribute.
* ArrayOfEqualSizedArrays: multi-dimensional numpy ndarray. Access data via the nda attribute.
* VectorOfVectors: a variable length array of variable length arrays.
Implemented as a pair of Arrays: "flattened_data" holding the raw data, and
"cumulative_length" whose ith element is the sum of the lengths of the vectors
with index <= i
* Struct: a python dictionary containing lgdos. Derives from dict
* Table: a Struct whose elements ("columns") are all array types with the same
length (number of rows)

Currently the primary on-disk format for lgdo's is LEGEND HDF5 (lh5) files. IO
is done via the class LH5Store.
lh5 files can also be browsed easily in python like any
[hdf5](https://www.hdfgroup.org/) file using [h5py](https://www.h5py.org/).

Tutorial coming soon. LEGEND collaborators can view a presentation on J.
Detwiler showing basic data browsing methods
[here](https://indico.legend-exp.org/event/371/contributions/1915/attachments/1167/1696/20200730_PGTProcessing.pdf).

#### lh5 to-do's
* waveform object definition and compression
* fix overwrite
* Flecher32 md5sums
* unit tests
* tutorial
* "provenance" into lgdo attrs: spec version, creator, creation time
