"""This subpackage provides some custom processors to process hit-structured
data into event-structured data.

Custom processors must adhere to the following signature: ::

    def my_evt_processor(
        datainfo,
        tcm,
        table_names,
        *,  # all following arguments are keyword-only
        arg1,
        arg2,
        ...
    ) -> LGDO:
        # ...

The first three arguments are automatically supplied by :func:`.build_evt`,
when the function is called from the :func:`.build_evt` configuration.

- `datainfo`: a :obj:`.DataInfo` object that specifies tier names, file names,
  HDF5 groups in which data is found and pattern used by hit table names to
  encode the channel identifier (e.g. ``ch{}``).
- `tcm`: :obj:`.TCMData` object that holds the TCM data, to be used for event
  reconstruction.
- `table_names`: a list of hit table names to read the data from.

The remaining arguments are characteristic to the processor and can be supplied
in the function call from the :func:`.build_evt` configuration.

The function must return an :class:`~lgdo.types.lgdo.LGDO` object suitable for
insertion in the final table with event data.

For examples, have a look at the existing processors provided by this subpackage.
"""
