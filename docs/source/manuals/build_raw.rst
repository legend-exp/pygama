Decoding digitizer data
=======================

*Under construction...*

Command line interface
----------------------

A command line interface to :func:`~.raw.build_raw.build_raw` is available
through the ``pygama`` executable via the ``build-raw`` sub-command. This can
be used to quickly convert digitizer data without custom scripting. Here are
some examples of what can be achieved:

.. code-block:: console

    $ pygama build-raw --help  # display usage and exit

Convert files and save them in the original directory with the same filenames
(but new extension ``.lh5``):

.. code-block:: console

    $ pygama [-v] build-raw data/*.orca  # increase verbosity with -v
    $ pygama build-raw --overwrite data/*.orca  # overwrite output files
    $ # set maximum number of rows to be considered from each file
    $ pygama build-raw --max-rows 100 data/*.orca

Customize the group layout of the LH5 files in a JSON configuration file (refer
to the :func:`~.raw.build_raw.build_raw` documentation for details):

.. code-block:: json

  {
    "FCEventDecoder": {
      "g{key:0>3d}": {
        "key_list": [[0, 58]],
          "out_stream": "{orig_basename}.lh5:/{name}/raw"
        },
        "s{key:0>3d}": {
          "key_list": [[59, 119]],
          "out_stream": "{orig_basename}.lh5:/{name}/raw"
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
   See :func:`~.raw.build_raw.build_raw` and ``pygama build-raw --help`` for a
   full list of conversion options.
