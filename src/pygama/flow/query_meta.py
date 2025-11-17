from lgdo import lh5, Table, Array, VectorOfVectors
from legendmeta import LegendMetadata
from pathlib import Path
from dbetto import Props, TextDB
import awkward as ak
import pandas as pd
from collections.abc import Collection
from pygama.flow.utils import to_datetime
import os
import re
from lgdo import Table

def query_metadata(
    runs: str,
    channels: str,
    fields: Collection[str]=[],
    prod_config: Path|str = "$REFPROD/dataflow-config.yaml",
    par_tiers: Collection[str] = None,
    by_run: bool = False,
    return_query_vals: bool = False,
    library: str = 'ak'
):
    """Query the metadata and pars data for a reference production. Return
    a table containing one entry for each run/channel with the requested
    data fields. Can also provide selections based on period, run, datatype
    and time information, as well as information found in the metadata, for
    that run/channel including detector database parameters.

    Parameters
    ----------
    runs
        expression used to select list of runs. Expression can contain
        the following variables:
        - period: name of period (e.g. ``"p01"``)
        - run: name of run (e.g. ``"r000"``)
        - datatype: three character ID of datatype (e.g. ``"cal"``)
        - starttime: start key for a run (e.g. ``"20230101T123456Z"``). This is
        -   converted to a _np.datetime64:https://numpy.org/doc/stable/reference/arrays.datetime.html#arrays-datetime (e.g. ``np.datetime64(53, "Y")``)

        Examples:
        - ``"p>=6 and p<=8 and datatype=='cal'"`` selects calibration data from periods 6, 7 and 8.
        - ``"period in ['p03', 'p04', 'p06'] and run == 'r000' and t=='phy'"`` selects the first run of physics data from periods 3, 4 and 6

    channels
        expression used to select channels for each run. Expression can
        contain variables found in detectors found in channel map. It
        can also use values from other metadata sources that are referenced
        using the references described under the `fields` arg.

        Examples:
        - ``"@det.system=='geds' and @det.type=='icpc' and @det.analysis.usability=='on'"``
          selects all ICPC detectors for each run that are marked as usable
        - ``"@det.name=='S010' and @det.analysis.processible"`` selects SiPM channel 10 and
          will only include runs where it is can be processed

        Note: if a parameter does not exist for a channel, it will evaluate to ``None``.
        If this causes an error to be thrown, this expression will evaluate to ``False``.

    fields
        list of fields to include in the table. Fields are drawn from multiple tree-like
        metadata sources, whose roots are referenced using shorthands prepended with ``@``.
        These references will navigate to locations corresponding to the run and channel
        for each entry. Parameters will be aliased to legal python variable names; this
        can be done manually be preceding the ``@`` with a preferred name (``[alias]@[path]);
        the variable will be passed to a column with this name, and this name can be used
        in queries. By default, replace periods in path with underscores.

        Data sources:
        - @det: value from channel map
        - @run: value from run info
        - @par: value from analysis parameters database in refprod

        Example:
        - ``["@det.daq.rawid", "@run.livetime", "aoe_low_cut@pars.pars.operations.AoE_Low_Cut.parameters.a"]``

    ref_prod
        base directory or config file of reference production. If a dir is provided, look for
        ``dataflow_config.yaml`` inside it; config file should be json or yaml.

    par_tiers
        search only provided tiers for pars. If ``None`` search all found tiers. Can
        speed up search.

    by_run
        if ``True``, return nested array grouped by run, with inner variable length arrays of
        channel data

    return_query_vals
        if ``True``, return values found in query as columns; else only return those in ``fields``
    
    library
        format of returned table. Can be ''ak'', ''lgdo'' or ''pd''
    """
    # return ak.array of periods, runs, channels, and fields from pars db
    ref_config = Props.read_from(lh5.utils.expand_path(prod_config), subst_pathvar=True)
    ref_paths = ref_config['paths']
    meta = LegendMetadata(ref_paths["metadata"])
    runinfo = meta.datasets.runinfo

    # Loop through run list and select runs matching query
    run_data = []
    for p, runlist in runinfo.items():
        for r, datalist in runlist.items():
            for dtype, info in datalist.items():
                run_record = {
                    "period": p,
                    "run": r,
                    "datatype": dtype,
                    "starttime": info.start_key,
                }
                if eval(runs, {}, run_record | {"starttime":to_datetime(info.start_key).replace(tzinfo=None)}):
                    run_data.append(run_record)

    # get the paths and groups corresponding to our runs+channels queries
    ch_data = []
    if par_tiers is None:
        par_dbs = [TextDB(path, lazy=True) for key, path in ref_paths.items() if key[:4]=="par_" and os.path.exists(path)]
    else:
        par_dbs = [TextDB(ref_paths[f"par_{tier}"], lazy=True) for tier in par_tiers]

    # get list of fields needed and build mapping to column names
    col_name_map = {}
    col_list = set()
    expr_vars = re.findall("[\\w\\.\\:@]+\\(?", channels)

    # capture alias@path.to.val into two variables
    parser = re.compile("(?:(\\w+))?(@\\w+(?:\\.\\w+)*)?")
    for field in fields + expr_vars:
        match = parser.fullmatch(field)
        if match is None or match == (None, None):
            raise ValueError()
        alias, path = match.groups()

        # map from path to alias
        if path is not None and col_name_map.get(path, None) is None:
            col_name_map[path] = alias
            # alias must be unique
            if alias is not None and any(path!=p and alias==a for p, a in col_name_map.items()):
                raise ValueError(f"alias {alias} already assigned")
        
        # path can only be aliased to a single name
        elif path in col_name_map and alias != col_name_map[path]:
            raise ValueError()
        
        # If this is in the field list, add to col_list
        if field in fields:
            if alias is not None:
                col_list.add(alias)
            else:
                col_list.add(path)
        
        elif field in expr_vars:
            if alias is not None:
                channels = channels.replace(field, alias)
            else:
                channels = channels.replace(field, path)

    # Find all the un-aliased paths and asign them an alias
    for path, alias in col_name_map.items():
        if alias is None:
            alias = path.replace(".", "_").replace("@", "")
            col_name_map[path] = alias
            channels = channels.replace(path, alias)
            if path in col_list:
                col_list.remove(path)
                col_list.add(alias)

    # Now loop through runs and find detectors for each run matching channel query
    records = []
    eval_success = False # track if the eval ever succeeds

    for run in run_data:
        detlist = meta.channelmap(run["starttime"], run["datatype"])
        r_info = runinfo[run["period"]][run["run"]][run["datatype"]]
        if by_run:
            if return_query_vals:
                ch_records = { k:[] for k in col_name_map.values() }
            else:
                ch_records = { k:[] for k in col_list }

            records.append(run | ch_records)

        for det in detlist.values():
            ch_record = dict()
            for path, col_name in col_name_map.items():
                p = path.split(".")
                
                param = None
                if p[0] == "@det":
                    param = det
                    for key in p[1:]:
                        try:
                            param = param[key]
                        except:
                            param = None
                            break
                elif p[0] == "@pars":
                    # search for the param in any of the tiers
                    for par_db in par_dbs:
                        try:
                            param = par_db.cal[run["period"]][run["run"]]
                            param.scan()
                            param = param[next(iter(param.keys()))][f"ch{det.daq.rawid}"]
                            for key in p[1:]:
                                param = param[key]
                        except:
                            param = None
                            continue
                        break
                elif p[0] == "@run":
                    # search for the parameter in runinfo
                    param = r_info
                    for key in p[1:]:
                        try:
                            param = param[key]
                        except:
                            param = None
                            break
                else:
                    raise ValueError(f"could not find metadata location {f[0]}. Options are '@pars', '@det'")

                ch_record[col_name] = param

            # Evaluate the channel expression on the found values
            try:
                keep_record = bool(eval(channels, {}, ch_record | run))
            except:
                continue
            eval_success = True

            if keep_record:
                if not return_query_vals:
                    ch_record = { k:v for k, v in ch_record.items() if k in col_list }
                if by_run:
                    for k, v in ch_record.items():
                        ch_records[k].append(v)
                else:
                    records.append(run | ch_record)

    # if evaluating channels query was never successful...
    if not eval_success:
        raise ValueError("Could not interpret channel query for any channels")

    result = ak.Array(records)
    if library == 'ak':
        return ak.Array(result)
    elif library == 'pd':
        return ak.to_dataframe(ak.Array(result))
    else:
        return Table({f: Array(a) if a.ndim==1 else VectorOfVectors(a) for f, a in result.items()})
