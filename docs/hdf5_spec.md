# LEGEND Data Format Specs [DRAFT]

## General considerations

In the interest of long-term data accessibility and to ensure compliance with FAIR data principles,

* The number of different file formats should be kept to a reasonable minimum.

* Only mature, well documented and and widely supported data formats with mature implementations/bindings for multiple programming languages are used.

* Custom file formats are, if at all, only used for raw data produced by DAQ systems. As raw data tends to be archived long-term, any custom raw data formats must fulfil the following requirements:

    * A complete formal description of the format exists and is made publicly available under a license that allows for independent third-party implementations.

    * At least verified implementations is made publicly available under an open-source license.


## Choice of file formats (preliminary)

Depending on the kind of data, the following formats are preferred:

* Binary data: HDF5

* Metadata: JSON


## Abstract data model

LEGEND data should, whereever possible, be representable by a simple data model consisting of:

* Scalar values

* Vectors or higher-dimensional arrays. Arrays may be flat and contain scalar numerical values or nested and contain arrays, but must not contain structs or tables.

* Structs (resp. "Dicts" or named tuples) of named fields. Fields may contain scalar values, arrays or structs. In-memory representations of structs may be objects, named t

* Tables (a.k.a. "DataFrames"), represented by structs of column-vectors of equal length.

Numerical values may be accompanied by physical units.

A generic mapping of this data model must be defined for each file format used. The mapping must be self-documenting.

# Low-level DAQ Data Structure

## General DAQ structure

DAQ data is represented by a table, each row represents a DAQ event on a single (physical or logical) input channel. Event building will happen at a higher data level.

The detailed structure of DAQ data will depend on the DAQ system and experimental setup. However, fields like these may become mandatory for all DAQs:

* `ch`: `array<1>{real}`
* `evttype`: `array<1>{enum{evt_undef=0,evt_real=1,evt_pulser=2,evt_mc=3,evt_baseline=4}}`
* `daqevtno`: `array<1>{real}`

A DAQ system with waveform digitization will provide columns like

* `waveform_lf`: array<1>{waveform}, see below
* `waveform_hf`: array<1>{waveform}, see below

If the DAQ performs an internal pulse-shape analysis (digital or analog), energy reconstruction and other columns may be available, e.g.:

* `psa_energy`: `array<1>{real}`
* `psa_trise`: `array<1>{real}`

Other DAQ and setup-specific columns will often be present, e.g.

* `muveto`: `array<1>{real}`

The collaboration will decide on a list of recommended columns names, to ensure columns with same semantics will have the same name, independent of DAQ/setup.

Legacy data that does not separate low-level DAQ data and event building will also include a column

* `evtno`: `array<1>{real}`


## Waveform vectors

Waveform data as be stored either directly in compressed form. Uncompressed waveform data is stored as a `table{t0,dt,values}`:

* `t0`: `array<1>{real}`
* `dt`: `array<1>{real}`
* Either `values`: `array<1>{array<1>{real}}` or `array_of_equalsized_arrays<1,1>{real}`
* or `encvalues`: `table{bytes,... codec information ...}`

* `encvalues`: `table{bytes,... codec information ...}`
*
* `bytes`: `array<1>{array<1>{real}}`
* `some_codec_information`: ...

Compressed waveform data is stored as a `table{t0,dt,encvalues}`:

* `t0`: `array<1>{real}`
* `dt`: `array<1>{real}`
* `encvalues`: `table{bytes,... codec information ...}`

The column `encvalues` has the structure

* `bytes`: `array<1>{array<1>{real}}`
* `some_codec_information`: ...

# HDF5 File Format [Incomplete]

HDF5 is used as the primary binary data format in LEGEND.

The following describes a mapping between the abstract data model and HDF5. This specifies the structure of the HDF5 implicitly, but precisely, for any data that conforms to the data model. The mapping purposefully uses only common and basic HDF5 features, to ensure it can be easily and reliably implemented in multiple programming languages.


## HDF5 datasets, groups and attributes

Different data types may be stored as an HDF5 dataset of the same type (e.g. a 2-dimensional dataset may represent a matrix or a vector of same-sized vectors). To make the HDF5 files self-documenting, the HDF5 attribute "datatype" is used to indicate the type semantics of datasets and groups.


## Abstract data model representation

The abstract data model is mapped as follows:


## Scalar values

Single scalar values are stored as 0-dimensional datasets

Attribute "datatype": "real", "string", "symbol", ...


## Arrays

### Flat arrays

Flat n-dimension arrays are stored as n-dimensional datasets

Attribute "datatype": "array<2>{ELEMENT_TYPE}"


### Fixed-sized arrays

...

Attribute "datatype": fixedsize_array<1>{ELEMENT_TYPE}"


### Arrays of arrays of same size

Nested arrays of dimensionality n, m, ... are stored as flat n+m+n dimensional datasets.

... attribute to denote dimensionality split ...

Attribute "datatype": "array_of_equalsized_arrays<N,M>{ELEMENT_TYPE}"


### Vectors of vectors of different size

Data of the inner arrays is flattened into a single 1-dimensional dataset. An auxiliary dataset stores the cumulative sum of the size of the inner arrays.

...

HDF Attributes:

* "datatype": "array<1>{array<1>{ELEMENT_TYPE}}"
* "cumsum_length_ds": Name (as string) of the dataset that stores the cumulative sum of the size of the inner arrays.

Note: Instead of referring to the auxiliary dataset by name, a HDF5 dataset reference may be used in the future (still to be evaluated).


## Structs

Structs are stored as HDF5 groups. Fields that are structs themselves are stored as sub-groups, scalars and arrays as datasets. Groups and datasets in the group are named after the fields of the struct.

HDF Attributes:

* "datatype": "struct{FIELDNAME_1,FIELDNAME_2,...}"


## Tables

A Table are stored are group of datasets, each representing a column of the table.

HDF Attributes:

* "datatype": "table{COLNAME_1,COLNAME_2,...}"


## Enums

Enum values are stores as integer values, but with the "datatype" attribute: "enum{NAME=INT_VALUE,...}". So a vector of enum values will have a "datatype" attribute like "array<N>{enum{NAME=INT_VALUE,...}}""


# Values with physical units

For values with physical units, the dataset only contains the numerical values. The attribute "units" stores the unit information. The attribute value is the string representation of the common scientific notation for the unit. Unicode must not be used.

HDF Attributes:

* "units": e.g. "mm", "um", "keV"


# Data Compression

In addition to compression features provided by standard data formats (HDF5, etc.), LEGEND data uses some custom data compression.

In the interest of long-term data accessibility and to ensure compliance with FAIR data principles, use of custom data compression methods has to be limited to a minimum number of methods and use cases. Long-term use is only acceptable if:

*  The custom compression significantly outperforms standard compression methods in compression ratio and/or (de-)compression speed for important use cases.

* A complete formal description of the algorithms exists and is made publicly available under a license that allows for independent third-party implementations.

* Verified implementations exist in a least two different programming languages, at least one of which has been implemented independently from the formal description of the algorithm and at least one of which is made publicly available under an open-source license.


## Lossless compression of integer-valued waveform vectors

As detector waveforms have specific shapes, custom compression algorithms optimized for this use case can show a much higher speed/throughput than generic compression algorithms, at similar compression ratios.

Currently, we use the following custom integer-waveform compression algorithms:

* radware-sigcompress v1.0

Other compression algorithms are being developed, tested and evaluated.

Note: The algorithm(s) in use are still subject to change, long-term data compatibility is not guaranteed at this point.


### radware-sigcompress

There is no formal description of the radware-sigcompress algorithm yet, so the C-code of the original implementation ("sigcompress.c") will serve as the reference for now:

```C
// radware-sigcompress, v1.0
//
// This code is licensed under the MIT License (MIT).
// Copyright (c) 2018, David C. Radford <radforddc@ornl.gov>

int compress_signal(short *sig_in, unsigned short *sig_out, int sig_len_in) {

  int   i, j, max1, max2, min1, min2, ds, nb1, nb2;
  int   iso, nw, bp, dd1, dd2;
  unsigned short db[2];
  unsigned int   *dd = (unsigned int *) db;
  static unsigned short mask[17] = {0, 1,3,7,15, 31,63,127,255,
                                    511,1023,2047,4095, 8191,16383,32767,65535};

  //static int len[17] = {4096, 2048,512,256,128, 128,128,128,128,
  //                      128,128,128,128, 48,48,48,48};
  /* ------------ do compression of signal ------------ */
  j = iso = bp = 0;

  sig_out[iso++] = sig_len_in;     // signal length
  while (j < sig_len_in) {         // j = starting index of section of signal
    // find optimal method and length for compression of next section of signal
    max1 = min1 = sig_in[j];
    max2 = -16000;
    min2 = 16000;
    nb1 = nb2 = 2;
    nw = 1;
    for (i=j+1; i < sig_len_in && i < j+48; i++) { // FIXME; # 48 could be tuned better?
      if (max1 < sig_in[i]) max1 = sig_in[i];
      if (min1 > sig_in[i]) min1 = sig_in[i];
      ds = sig_in[i] - sig_in[i-1];
      if (max2 < ds) max2 = ds;
      if (min2 > ds) min2 = ds;
        nw++;
    }
    if (max1-min1 <= max2-min2) { // use absolute values
      nb2 = 99;
      while (max1 - min1 > mask[nb1]) nb1++;
      //for (; i < sig_len_in && i < j+len[nb1]; i++) {
      for (; i < sig_len_in && i < j+128; i++) { // FIXME; # 128 could be tuned better?
        if (max1 < sig_in[i]) max1 = sig_in[i];
        dd1 = max1 - min1;
        if (min1 > sig_in[i]) dd1 = max1 - sig_in[i];
        if (dd1 > mask[nb1]) break;
        if (min1 > sig_in[i]) min1 = sig_in[i];
        nw++;
      }
    } else {                      // use difference values
      nb1 = 99;
      while (max2 - min2 > mask[nb2]) nb2++;
      //for (; i < sig_len_in && i < j+len[nb1]; i++) {
      for (; i < sig_len_in && i < j+128; i++) { // FIXME; # 128 could be tuned better?
        ds = sig_in[i] - sig_in[i-1];
        if (max2 < ds) max2 = ds;
        dd2 = max2 - min2;
        if (min2 > ds) dd2 = max2 - ds;
        if (dd2 > mask[nb2]) break;
        if (min2 > ds) min2 = ds;
        nw++;
      }
    }

    if (bp > 0) iso++;
    /*  -----  do actual compression  -----  */
    sig_out[iso++] = nw;  // compressed signal data, first byte = # samples
    bp = 0;               // bit pointer
    if (nb1 <= nb2) {
      /*  -----  encode absolute values  -----  */
      sig_out[iso++] = nb1;                    // # bits used for encoding
      sig_out[iso++] = (unsigned short) min1;  // min value used for encoding
      for (i = iso; i <= iso + nw*nb1/16; i++) sig_out[i] = 0;
      for (i = j; i < j + nw; i++) {
        dd[0] = sig_in[i] - min1;              // value to encode
        dd[0] = dd[0] << (32 - bp - nb1);
        sig_out[iso] |= db[1];
        bp += nb1;
        if (bp > 15) {
          sig_out[++iso] = db[0];
          bp -= 16;
        }
      }

    } else {
      /*  -----  encode derivative / difference values  -----  */
      sig_out[iso++] = nb2 + 32;  // # bits used for encoding, plus flag
      sig_out[iso++] = (unsigned short) sig_in[j];  // starting signal value
      sig_out[iso++] = (unsigned short) min2;       // min value used for encoding
      for (i = iso; i <= iso + nw*nb2/16; i++) sig_out[i] = 0;
      for (i = j+1; i < j + nw; i++) {
        dd[0] = sig_in[i] - sig_in[i-1] - min2;     // value to encode
        dd[0]= dd[0] << (32 - bp - nb2);
        sig_out[iso] |= db[1];
        bp += nb2;
        if (bp > 15) {
          sig_out[++iso] = db[0];
          bp -= 16;
        }
      }
    }
    j += nw;
  }

  if (bp > 0) iso++;
  if (iso%2) iso++;     // make sure iso is even for 4-byte padding
  return iso;           // number of shorts in compressed signal data

} /* compress_signal */


int decompress_signal(unsigned short *sig_in, short *sig_out, int sig_len_in) {

  int   i, j, min, nb, isi, iso, nw, bp, siglen;
  unsigned short db[2];
  unsigned int   *dd = (unsigned int *) db;
  static unsigned short mask[17] = {0, 1,3,7,15, 31,63,127,255,
                                    511,1023,2047,4095, 8191,16383,32767,65535};

  /* ------------ do decompression of signal ------------ */
  j = isi = iso = bp = 0;
  siglen = (short) sig_in[isi++];  // signal length
  //printf("<<< siglen = %d\n", siglen);
  for (i=0; i<2048; i++) sig_out[i] = 0;
  while (isi < sig_len_in && iso < siglen) {
    if (bp > 0) isi++;
    bp = 0;              // bit pointer
    nw = sig_in[isi++];  // number of samples encoded in this chunk
    nb = sig_in[isi++];  // number of bits used in compression

    if (nb < 32) {
      /*  -----  decode absolute values  -----  */
      min = (short) sig_in[isi++];  // min value used for encoding
      db[0] = sig_in[isi];
      for (i = 0; i < nw && iso < siglen; i++) {
        if (bp+nb > 15) {
          bp -= 16;
          db[1] = sig_in[isi++];
          db[0] = sig_in[isi];
          dd[0] = dd[0] << (bp+nb);
        } else {
          dd[0] = dd[0] << nb;
        }
        sig_out[iso++] = (db[1] & mask[nb]) + min;
        bp += nb;
      }

    } else {
      nb -= 32;
      /*  -----  decode derivative / difference values  -----  */
      sig_out[iso++] = (short) sig_in[isi++];  // starting signal value
      min = (short) sig_in[isi++];             // min value used for encoding
      db[0] = sig_in[isi];
      for (i = 1; i < nw && iso < siglen; i++) {
        if (bp+nb > 15) {
          bp -= 16;
          db[1] = sig_in[isi++];
          db[0] = sig_in[isi];
          dd[0] = dd[0] << (bp+nb);
        } else {
          dd[0] = dd[0] << nb;
        }
        sig_out[iso] = (db[1] & mask[nb]) + min + sig_out[iso-1]; iso++;
        bp += nb;
      }
    }
    j += nw;
  }

  if (siglen != iso) {
    printf("ERROR in decompress_signal: iso (%d ) != siglen (%d)!\n",
           iso, siglen);
  }
  return siglen;       // number of shorts in decompressed signal data

} /* decompress_signal */
```