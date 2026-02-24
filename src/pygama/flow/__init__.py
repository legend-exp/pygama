"""
Routines for performing structured queries on LEGEND data.

Queries selectively access LEGEND data using simple boolean selection
expressions to access data from multiple datasources, including metadata
and parameter databases, and many tiers of data production. The queries return
the data in Tabular formats (``ak``, ``pd``, ``np``) that can be used to
perform further analysis.

They also are designed to access data efficiently and at scale, by
(as much as possible) accessing only what is necessary to complete the query,
and by breaking the query into chunks that can fit in memory and be operated
on in parallel.

Several query commands exist, to access data at different levels:

- :meth:`query_runs` accesses all runs in a data production, using information
  from the cycle names
- :meth:`query_meta` accesses channel and run data, grabbing information from
  the metadata repository, the parameters databases, and the run information from
  cycle names
- :meth:`query_data` accesses event data for select channels and runs, grabbing
  information from all data production tiers, parameter databases, and run information
  from cycle names
- :meth:`query_hist` accesses event data, as above, and returns a histogram
- :meth:`query_evt` accesses `evt` tier data for a selection of runs; note that this
  cannot make use of other data tiers or of metadata (yet...)
- :meth:`build_iterator` creates a :class:`LH5Iterator` object to load a selection of
  data fields and metadata for a selection of channels and runs. This can be used to
  perform more advanced queries if needed.

The information about the data production required to perform these queries can be
accessed from its `dataflow-config.yaml` file. The information needed comes from
the paths used for dataflow, and a set of query parameters (which are arguments
of the above set of functions). Note that the `dataflow-config.yaml` need not come
from an existing production; one can be modified to point towards additional paths
as needed!

A template for a minimal `dataflow-config.yaml` file::
    paths:
        metadata: $_/inputs # path to metadata root dir; usually $REFPROD/inputs

        tier_raw: $_/generated/tier/raw # path to raw tier root dir
        tier_[tla]: $_/generated/tier/... # path to root dir for tier named TLA
        ...

        par_[tla]: $_/generated/par/... # path to root dir for pargen database
    ...

    query:
        cycle_def: experiment-period-run-datatype-starttime # REQUIRED: hyphen-separated-list-of-fields-in-cycle-name; these will be columns of run db
        metadata: LegendMetadata # REQUIRED: name of metadata class (e.g. LegendMetadata)
        rundb_tier: # tier to use to populate run DB; default to "raw"
        ignored_cycles: dataprod/config/ignored_cycles # path in metadata to list of cycles to skip; by default do not ignore any
        tiers: ["raw", "dsp", "hit", "evt"] # list of tiers TLAs to use from paths for parameter and data queries; default use all
        chan_db: # path in metadata to list of channels for a given run. Use format string syntax, which may refer to any values in the run DB (i.e. cycle_def fields, cycle name, and relative path). If no value was provided, call "metadata.channelmap(on = starttime)", where "starttime" is drawn from the run db
        par_db: #optional info for navigating par dbs. If missing use "par_db.on(starttime)[@chan.name]"
            cycle_entry: # sub-path to entry for cycle, using format string syntax, which may include values from run db; if missing or falsey, call .on
            chan_entry: @chan.name # sub-sub-path to entry for channel, using format string syntax, which may include values from run or chan dbs; if missing or falsey, assume same values for all channels in a cycle (useful if only one channel per cycle...)
        meta_dbs: # optional list of metadata DBs
            name: # @name of db for access in query_meta
                path: # metadata path to database
                cycle_entry: # sub-path to entry for cycle, using format string syntax, which may include values from run db; if missing or falsey, call .on
                chan_entry: # sub-sub-path to entry for channel, using format string syntax, which may include values from run or chan dbs; if missing or falsey, use same value for all chans
                # based on these, we will search "metadata["[path][.on(starttime)|/cycle_entry][/chan_entry]
            ...
        tables:
            raw: ch{@chan.daq.rawid:07d}/raw
            evt: evt
            [tla]: # path to table for channel in tier_[tla] lh5 files. Use format string syntax, which can refer to values from the run_db and chan_db; e.g. "ch{@chan.daq.rawid}/raw"
            ...

These ``dataflow-config.yaml`` files should not often need to be created, and will
be provided with most dataprods! If you set the environment variable $REFPROD then
the file will automatically be accessed from the referenced directory!
"""

from legendmeta.query import query_meta, query_runs

from .build_iterator import build_iterator
from .query_data import query_data
from .query_evt import query_evt
from .query_hist import query_hist

__all__ = [
    "query_runs",
    "query_meta",
    "query_data",
    "query_hist",
    "query_evt",
    "build_iterator",
]
