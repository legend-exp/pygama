Decoding digitizer data
=======================

The primary function for data conversion into raw-tier LH5 files is
:func:`.raw.build_raw.build_raw`. This is a one-to many function: one input DAQ
file can generate one or more output raw files. Control of which data ends up
in which files, and in which HDF5 groups inside of each file, is controlled via
:mod:`.raw.raw_buffer`.

Currently we support the following DAQ data formats:

* `FlashCam <https://www.mizzi-computer.de/home>`_
* `CoMPASS <https://www.caen.it/products/compass>`_
* `ORCA <https://github.com/unc-enap/Orca>`_, reading out:

  - FlashCam
  - `Struck SIS3302 <https://www.struck.de/sis3302.htm>`_
  - `Struck SIS3316 <https://www.struck.de/sis3316.html>`_

The examples in the following are based on the FlashCam (via ORCA) decoders.

Configuration
-------------

Basic usage of |build_raw| requires zero configuration: ::

    from pygama.raw import build_raw
    build_raw("daq-data.ext", out_spec="raw-data.lh5")

pygama will autodetect the DAQ format (if not, the *in_stream_type* is your
friend), decode all the data it can and save it to an LH5 file named
``raw-data.lh5``. The data in the output file is organized by record types (e.g.
event stream, DAQ hardware status, configuration, etc.)

.. tip::
   Check the |build_raw| documentation for a full list of useful options.

When the *out_spec* argument is a dictionary or a string ending with ``.json``,
it is interpreted as a configuration dictionary or a JSON file containing it,
respectively. Technically, this dictionary configures a
:class:`~.raw.raw_buffer.RawBufferLibrary`.

.. tip::
   The full configuration format specification is documented in depth in
   :meth:`.raw.raw_buffer.RawBufferLibrary.set_from_json_dict`.

Let's use the following configuration file as an example:

.. code-block::
   :caption: ``raw-out-spec.json``
   :linenos:

    {
      "ORFlashCamWaveformDecoder" : {
        "group1-{key:07d}/raw" : {
          "key_list" : [[1, 3], 9],
          "out_stream" : "{filename}"
        },
        "group2-{key:07d}/raw" : {
          "key_list" : [[11, 13]],
          "out_stream" : "{filename}"
        }
      },
      "OrcaHeaderDecoder" : {
        "header-data" : {
          "key_list" : ["*"],
          "out_stream" : "{filename}"
        }
      },
      "*" : {
        "extra/{name}" : {
          "key_list" : ["*"],
          "out_stream" : "extra.lh5"
        }
      }
    }

The first-level keys specify the names of the
:class:`~.raw.data_decoder.DataDecoder`-derived classes to be used in the
decoding. In the example above,
:class:`~.raw.orca.orca_flashcam.ORFlashCamWaveformDecoder` and
:class:`~.raw.orca.orca_flashcam.OrcaHeaderDecoder`. The user can also use just
``*``, which matches any other decoder known to pygama.

The second-level dictionary keys are the names used to label the decoded
objects (:class:`~.raw.raw_buffer.RawBuffer`\ s) in the output file. These
string can include `format specifiers
<https://docs.python.org/3/library/string.html#format-string-syntax>`_ for
variable expansion (see next section). The first key in
``ORFlashCamWaveformDecoder``, for example, will result in data being written
to ``group1-0000001/raw``, ``group1-0000002/raw`` etc., depending on the value
of ``key``. The computed label is stored in a variable called ``name``, which
can be expanded in other configuration fields.

.. note::
   If the first-level key is ``*``, ``name`` is expanded to the data decoder
   name instead of the raw buffer name. The last configuration block from the
   example will result in data from the e.g. ``AuxDecoder1`` decoder being
   written as ``extra/AuxDecoder1`` in the output file ``extra.lh5``.

The first fundamental configuration inside this block is ``key_list``. In this
context, "keys" refer to the labels used by the specific data decoder for
DAQ "streams" or "channels". The ``key_list`` list can be effectively use to
select channels to be decoded. Examples of possible values:

- ``[1, 3, 5]``: channels 1, 3 and 5
- ``[[1, 7]]``: all channels from 1 to 7
- ``["*"]``: all available channels

During decoding, the value of the current key is stored in the variable
``key``, which can be expanded in other configuration fields. This feature
allows, as seen above, to label channel data individually and programmatically.
The second configuration block, for example, in ``ORFlashCamWaveformDecoder``
will result in data from channels 11, 12, and 13 to be written as
``group2-0000011/raw``, ``group2-0000012/raw`` and ``group2-0000013/raw``.

The second configuration field is ``out_stream``, i.e. the output stream to
which the data should be written. A colon (``:``) can be used to separate the
stream name or address from an in-stream path or port. Examples:

- LH5 file and group: ``/path/filename.lh5:/group``
- Socket and port: ``198.0.0.100:8000``
- Variable to be expanded: ``{filename}``

Variable expansion
^^^^^^^^^^^^^^^^^^

As mentioned, the |build_raw| configuration supports variable expansion through
the `format string syntax
<https://docs.python.org/3/library/string.html#format-string-syntax>`_. The two
predefined variables are ``key`` and ``name``, but any other variable can be
expanded by passing its value to |build_raw| as keyword argument. For example,
for the the configuration shown above, ``filename`` must be defined like this: ::

    build_raw("daq-data.orca", out_spec="raw-out-spec.json", filename="raw-data.lh5")

.. note::
   ``key`` and ``name`` can be overloaded by keyword arguments in |build_raw|.

Output
^^^^^^

Running |build_raw| with the examined configuration on an example ORCA DAQ file
results in the following two LH5 files being produced:

.. code-block:: none

   raw-data.lh5
   ├── group1-000001
   │   └── raw
   ├── group1-000002
   │   └── raw
   ├── group1-000003
   │   └── raw
   ├── group1-000009
   │   └── raw
   ├── group2-000011
   │   └── raw
   ├── group2-000013
   │   └── raw
   └── header-data

   extra.lH5
   └── extra
       ├── FCConfig
       └── ORRunDecoderForRun

Data post-processing
^^^^^^^^^^^^^^^^^^^^

.. warning::
   to be written

Command line interface
----------------------

A command line interface to |build_raw| is available through the ``pygama``
executable via the ``build-raw`` sub-command. This can be used to quickly
convert digitizer data without custom scripting. Here are some examples of what
can be achieved:

.. code-block:: console

    $ pygama build-raw --help  # display usage and exit

Convert files and save them in the original directory with the same filenames
(but new extension ``.lh5``):

.. code-block:: console

    $ pygama [-v] build-raw data/*.orca  # increase verbosity with -v
    $ pygama build-raw --overwrite data/*.orca  # overwrite output files
    $ # set maximum number of rows to be considered from each file
    $ pygama build-raw --max-rows 100 data/*.orca

Customize the group layout of the LH5 files in a JSON configuration file (see
above section):

.. code-block:: json

  {
    "FCEventDecoder": {
      "ch{key:0>3d}/raw": {
        "key_list": [[0, 58]],
          "out_stream": "{orig_basename}.lh5"
        }
      }
    }
  }

and pass it to the command line:

.. code-block:: console

    $ pygama build-raw --out-spec fcio-config.json data/*.fcio

.. note::
   A special keyword ``orig_basename`` is automatically replaced in the JSON
   configuration by the original DAQ file name without extension. Such a
   feature is useful to users that want to customize the HDF5 group layout
   without having to worry about file naming. This keyword is only available
   through the command line.

.. seealso::
   See |build_raw| and ``pygama build-raw --help`` for a full list of
   conversion options.

.. |build_raw| replace:: :func:`~.raw.build_raw.build_raw`
