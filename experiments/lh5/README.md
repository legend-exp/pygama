LEGEND HDF5 DAQ Data Hierarchy (O. Schulz, Draft!)
==================================================

Notation:

* `name [Group]`: HDF5 group (essentially a "subdirectory" within the HDF5 file)
* `name [Vector{T}(n)]`: 1-dim HDF5 dataset of type T with length n
* `name [Vector{T}(m, n)]`: 2-dim HDF5 dataset of type T with size m x n


Conventions:

* Group names are plural ("traces"), dataset names are singular ("offset").

* All group and dataset names are lower-case and do not contain underscores
  (to allow an implicitly defined bi-directional mapping between
  `groupa/groupb/datasetc` in HDF5 and `groupa_groupb_datasetc` in other
  systems/contexts).

* Broadcasting: Vectors that correspond to other vectors in a one-to-one
  fashion, but have length one, are treated as having the same length as
  the corresponding vector and containing the same value in each element.
  This is well supported by Python and Julia. Example: Same encoding scheme
  or same maximum number of samples for all waveforms.


Structure:

* raw [Group]: Raw DAQ data, preserve as much information as possible
  * ged [Group]: HPGe subsystem/data-stream
      * timestamp [Vector{UInt64}(nevents)]: Raw DAQ event timestamp
      * daqno [Vector{Int32}(nevents)]: ADC/DAQ-system number
      * chno [Vector{Int32}(nevents)]: Channel number
      * bufferno [Vector{UInt32}(nevents)]: see below
      * chunkno [Vector{UInt32}(nevents)]: see below
      * daqflags [Vector{UInt32}(nevents)]: e.g. Bits for overflow,
        underflow, pre-/post-pileup, veto inputs, etc.
      * psa [Group]: If the DAQ supports online PSA (energy reco, sums, ...)
          * e.g. energy [Vector{Float32}(nevents)]
          * e.g. trigmax [Vector{Int}(nevents)]: e.g. for Struck SIS3315
          * e.g. sum [Group]: Vector-of-vectors (see below), e.g. for
            Struck SIS3315
      * traces (or waveforms) [Group]
          * main [Group]: e.g. long LF trace
              * samplerate [Vector{Int32|Int64}(nevents)]: sampling frequency
                (in Hz)
              * dynrange [Group]:
                  * lo [Vector{Int32}(nevents)]: min ADC sample value
                  * hi [Vector{Int32}(nevents)]: max ADC sample value
              * maxnsamples [Vector{Int32}(nevents)]: maximum number of samples
              * offset [Vector{Int64}(nevents)]: typically the number of
                pre-trigger samples
              * encoding [Vector{Int}(nevents)]: sample data encoding scheme
              * samples [Group]:
                  * Either: [Vector-of-vectors{T}, see below]: T = UInt8 for
                    encoded (i.e. compressed) data and T = UInt16|Int32, etc.
                    for raw sample values (not compressed).
                  * Or: [Matrix{T}(nevents, nsamples)
          * aux [Group]: e.g. short HF trace, same structure as main

  * pmt [Group]: PMT subsystem/data-stream, structure similar to "ged"

  * spm [Group]: SiPM subsystem/data-stream, structure similar to "ged"

  * ...


Notes:

* It's likely that a file will only contain a subset of the all channels of a
  single data stream. Still, the data structure should allow to store all
  data of all channels of all subsystems at all levels of analysis in one
  file. This is useful for development and testing with small amounts of data,
  allows arbitrary merging of data, and makes it easy to auto-detect what
  information a file contains (independently of the file name).

* Waveform samples may either be stored directly as signed or unsigned
  integer values, or stored compressed as a byte-stream (separate for each
  waveform).

* Depending on the DAQ-System, some waveforms of the same channel may be
  shorter than others (typically when an event occurs shortly after a previous
  event, resulting in fewer pre-trigger samples). So for raw data, even
  uncompressed samples will typically be stored as a vector-of-vectors.
  However, for DAQ systems that always produce a the same fixed number of
  samples for all channels, uncompressed samples can be stored as an
  nevents x nsamples matrix.

* ADCs/DAQ-Systems: A DAQ-system may consist of a single ADC, or multiple
  ADC cards. We'll call it a single ADC/DAQ-System if all of it's channels
  run on the same sampling clock and (ideally) switch buffers at the same time
  (see below). GERDA has a single synchronized DAQ-System, but LEGEND may well
  have several ADCs/DAQ-Systems that run independently and are synchronized
  offline (by timestamp) during analysis.

* Buffers: Typically, ADCs have buffers that can store up to a certain
  number of events and there is usually a short dead time when switching to
  a new buffer. So switching buffers should be recorded by storing a buffer
  number per event. It is desirable that all channels of a single
  ADC/DAQ-System switch buffers at the same time.

* Events will typically *not* be stored in strict chronological order: A
  number of consecutive events will be stored for one channel of an
  ADC/DAQ-System (e.g. all events in an buffer fill), then some events for the
  next channel of the same ADC/DAQ-System, then for the first channel of
  another ADC, and so on. This is also desirable for analysis, as events of
  the same channel of the same ADC will be processed with the same parameters.
  However, chronological order must be guaranteed between all events of the
  same channel of the same ADC/DAQ-System.

* GERDA stores two waveforms for each channel for each event: A long trace
  with reduced time resolution, and a short trace with full time resolution.
  LEGEND will not necessarily do the same, but the data structure should
  allow for multiple traces per channel and event.

* The raw-data structure above is designed to be reusable in later stages of
  pulse-shape analysis (e.g. to store pre-processed samples for fitting,
  machine learning, etc.).


Special data structures:

* Vector-of-vectors (of different length): Stored as a group with two
  datasets:

    * values: Concatenated values of all vectors
    * elemptr: Valid indices of vector `values`

  The outer vector contains `length(elemptr) - 1` vectors. The values of
  vector `i` are stored in `values[elemptr[i]]` up to (not including)
  `values[elemptr[i+1]]`.
