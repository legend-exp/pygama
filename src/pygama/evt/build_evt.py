"""
This module implements routines to build the `evt` tier.
"""

from __future__ import annotations

import importlib
import itertools
import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import awkward as ak
import numpy as np
from lgdo import Array, ArrayOfEqualSizedArrays, Table, VectorOfVectors, lh5
from lgdo.lh5 import LH5Store

from . import aggregators, utils

log = logging.getLogger(__name__)


def build_evt(
    datainfo: utils.DataInfo | Mapping[str, Sequence[str, ...]],
    config: str | Mapping[str, ...],
    wo_mode: str = "write_safe",
) -> None | Table:
    r"""Transform data from hit-structured tiers to event-structured data.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found,
        (see :obj:`.utils.DataInfo`). Example: ::

            # syntax: {"tier-name": ("file-name", "hdf5-group"[, "table-format"])}
            {
              "tcm": ("data-tier_tcm.lh5", "hardware_tcm_1"),
              "dsp": ("data-tier_dsp.lh5", "dsp", "ch{}"),
              "hit": ("data-tier_hit.lh5", "hit", "ch{}"),
              "evt": ("data-tier_evt.lh5", "evt")
            }

    config
        name of configuration file or dictionary defining event fields. Channel
        lists can be defined by importing a metadata module.

        - ``operations`` defines the fields ``name=key``, where ``channels``
          specifies the channels used to for this field (either a string or a
          list of strings),
        - ``aggregation_mode`` defines how the channels should be combined (see
          :func:`evaluate_expression`).
        - ``expression`` defnies the mathematical/special function to apply
          (see :func:`evaluate_expression`),
        - ``query`` defines an expression to mask the aggregation.
        - ``parameters`` defines any other parameter used in expression.

        For example:

        .. code-block:: json

            {
              "channels": {
                "geds_on": ["ch1084803", "ch1084804", "ch1121600"],
                "spms_on": ["ch1057600", "ch1059201", "ch1062405"],
                "muon": "ch1027202",
              },
              "operations": {
                "energy_id":{
                  "channels": "geds_on",
                  "aggregation_mode": "gather",
                  "query": "hit.cuspEmax_ctc_cal > 25",
                  "expression": "tcm.array_id",
                  "sort": "ascend_by:dsp.tp_0_est"
                },
                "energy":{
                  "aggregation_mode": "keep_at_ch:evt.energy_id",
                  "expression": "hit.cuspEmax_ctc_cal > 25"
                }
                "is_muon_rejected":{
                  "channels": "muon",
                  "aggregation_mode": "any",
                  "expression": "dsp.wf_max > a",
                  "parameters": {"a": 15100},
                  "initial": false
                },
                "multiplicity":{
                  "channels":  ["geds_on", "geds_no_psd", "geds_ac"],
                  "aggregation_mode": "sum",
                  "expression": "hit.cuspEmax_ctc_cal > a",
                  "parameters": {"a": 25},
                  "initial": 0
                },
                "t0":{
                  "aggregation_mode": "keep_at_ch:evt.energy_id",
                  "expression": "dsp.tp_0_est",
                  "initial": "np.nan"
                },
                "lar_energy":{
                  "channels": "spms_on",
                  "aggregation_mode": "function",
                  "expression": "pygama.evt.modules.spms.gather_pulse_data(<...>, observable='hit.energy_in_pe')"
                },
              }
            }

    wo_mode
        writing mode, see :func:`lgdo.lh5.core.write`.
    """
    if not isinstance(config, dict):
        with open(config) as f:
            config = json.load(f)

    if "channels" not in config.keys():
        raise ValueError("channel field needs to be specified in the config")
    if "operations" not in config.keys():
        raise ValueError("operations field needs to be specified in the config")

    # convert into a nice named tuple
    f = utils.make_files_config(datainfo)

    # check chname_fmt validity
    chname_fmt = f.hit.table_fmt
    pattern_check = re.findall(r"{([^}]*?)}", chname_fmt)
    if len(pattern_check) != 1:
        raise ValueError("chname_fmt must have exactly one placeholder {}")
    elif "{" in pattern_check[0] or "}" in pattern_check[0]:
        raise ValueError(f"{chname_fmt=} has an invalid placeholder.")

    if (
        utils.get_table_name_by_pattern(
            chname_fmt,
            utils.get_tcm_id_by_pattern(chname_fmt, lh5.ls(f.hit.file)[0]),
        )
        != lh5.ls(f.hit.file)[0]
    ):
        raise ValueError(f"chname_fmt {chname_fmt} does not match keys in data!")

    # create channel list according to config
    # This can be either read from the meta data
    # or a list of channel names
    log.debug("creating channel dictionary")

    channels = {}

    for k, v in config["channels"].items():
        if isinstance(v, dict):
            # it is a meta module. module_name must exist
            if "module" not in v.keys():
                raise ValueError(
                    "Need module_name to load channel via a meta data module"
                )

            attr = {}
            # the time_key argument is mandatory
            if "time_key" not in v.keys():
                raise RuntimeError("the 'time_key' configuration field is mandatory")

            # if "None" do None
            elif "None" == v["time_key"]:
                attr["time_key"] = None

            # load module
            p, m = v["module"].rsplit(".", 1)
            met = getattr(importlib.import_module(p, package=__package__), m)
            channels[k] = met(v | attr)

        elif isinstance(v, str):
            channels[k] = [v]

        elif isinstance(v, list):
            channels[k] = [e for e in v]

    # load tcm data from disk
    tcm = utils.TCMData(
        id=lh5.read_as(f"/{f.tcm.group}/array_id", f.tcm.file, library="np"),
        idx=lh5.read_as(f"/{f.tcm.group}/array_idx", f.tcm.file, library="np"),
        cumulative_length=lh5.read_as(
            f"/{f.tcm.group}/cumulative_length", f.tcm.file, library="np"
        ),
    )

    # get number of events in file (ask the TCM)
    n_rows = len(tcm.cumulative_length)
    table = Table(size=n_rows)

    # now loop over operations (columns in evt table)
    for k, v in config["operations"].items():
        log.debug(f"processing evt field {k}")

        # if mode not defined in operation, it can only be an operation on the
        # evt level
        if "aggregation_mode" not in v.keys():
            var = {}
            if "parameters" in v.keys():
                var = var | v["parameters"]

            # compute and eventually get rid of evt. suffix
            obj = table.eval(v["expression"].replace(f"{f.evt.group}.", ""), var)

            # add attributes if present
            if "lgdo_attrs" in v.keys():
                obj.attrs |= v["lgdo_attrs"]

        # else we build the event entry
        else:
            if "channels" not in v.keys():
                channels_e = []
            elif isinstance(v["channels"], str):
                channels_e = channels[v["channels"]]
            elif isinstance(v["channels"], list):
                channels_e = list(
                    itertools.chain.from_iterable([channels[e] for e in v["channels"]])
                )
            channels_skip = []
            if "exclude_channels" in v.keys():
                if isinstance(v["exclude_channels"], str):
                    channels_skip = channels[v["exclude_channels"]]
                elif isinstance(v["exclude_channels"], list):
                    channels_skip = list(
                        itertools.chain.from_iterable(
                            [channels[e] for e in v["exclude_channels"]]
                        )
                    )

            pars, query, defaultv, srter = None, None, np.nan, None
            if "parameters" in v.keys():
                pars = v["parameters"]
            if "query" in v.keys():
                query = v["query"]
            if "initial" in v.keys():
                defaultv = v["initial"]
                if isinstance(defaultv, str) and (
                    defaultv in ["np.nan", "np.inf", "-np.inf"]
                ):
                    defaultv = eval(defaultv)

            if "sort" in v.keys():
                srter = v["sort"]

            obj = evaluate_expression(
                datainfo,
                tcm,
                channels=channels_e,
                channels_skip=channels_skip,
                mode=v["aggregation_mode"],
                expr=v["expression"],
                n_rows=n_rows,
                table=table,
                parameters=pars,
                query=query,
                default_value=defaultv,
                sorter=srter,
            )

            # add attribute if present
            if "lgdo_attrs" in v.keys():
                obj.attrs |= v["lgdo_attrs"]

        log.debug(f"new column {k!s} = {obj!r}")
        table.add_field(k, obj)

    store = LH5Store()

    # write output fields into outfile
    if "outputs" in config.keys():
        if len(config["outputs"]) < 1:
            log.warning("No output fields specified, no file will be written.")
            return table
        else:
            clms_to_remove = [e for e in table.keys() if e not in config["outputs"]]
            for fld in clms_to_remove:
                table.remove_field(fld, True)

            if f.evt.file is not None:
                store.write(
                    obj=table,
                    name=f.evt.group,
                    lh5_file=f.evt.file,
                    wo_mode=wo_mode,
                )
            else:
                return table
    else:
        log.warning("no output fields specified, no file will be written.")


def evaluate_expression(
    datainfo: utils.DataInfo | Mapping[str, Sequence[str, ...]],
    tcm: utils.TCMData,
    channels: Sequence[str],
    channels_skip: Sequence[list],
    mode: str,
    expr: str,
    n_rows: int,
    table: Table = None,
    parameters: Mapping[str, Any] = None,
    query: str = None,
    default_value: bool | int | float = np.nan,
    sorter: str = None,
) -> Array | ArrayOfEqualSizedArrays | VectorOfVectors:
    """Evaluates the expression defined by the user across all channels
    according to the mode.

    Parameters
    ----------
    datainfo
        input and output LH5 files with HDF5 groups where tables are found.
        (see :obj:`.utils.DataInfo`)
    tcm
        tcm data structure (see :obj:`.utils.TCMData`)
    channels
       list of channel names across which expression gets evaluated
    channels_skip
       list of channels which get set to default value during evaluation. In
       function mode they are removed entirely
    mode
       The mode determines how the event entry is calculated across channels.
       Options are:

       - ``first_at:sorter``: aggregates across channels by returning the
         expression of the channel with smallest value of sorter.
       - ``last_at``: aggregates across channels by returning the expression of
         the channel with largest value of sorter.
       - ``sum``: aggregates by summation.
       - ``any``: aggregates by logical or.
       - ``all``: aggregates by logical and.
       - ``keep_at_ch:ch_field``: aggregates according to passed ch_field.
       - ``keep_at_idx:tcm_idx_field``: aggregates according to passed tcm
         index field.
       - ``gather``: channels are not combined, but result saved as
         :class:`.VectorOfVectors`.
       - ``function``: the function call specified in `expr` is evaluated, and
         the resulting column is inserted into the output table.

    query
       a query that can mask the aggregation.
    expr
       the expression. That can be any mathematical equation/comparison. If
       `mode` is ``function``, the expression needs to be a special processing
       function defined in modules (e.g. :func:`.modules.spm.get_energy`). In
       the expression parameters from either hit, dsp, evt tier (from
       operations performed before this one! Dictionary operations order
       matters), or from the ``parameters`` field can be used.
    n_rows
       number of rows to be processed.
    table
       table of `evt` tier data.
    parameters
       dictionary of parameters defined in the ``parameters`` field in the
       configuration dictionary.
    default_value
       default value of evaluation.
    sorter
       can be used to sort vector outputs according to sorter expression (see
       :func:`.evaluate_to_vector`).

    Note
    ----
    The specification of custom functions that can be used as expression is
    documented in :mod:`.modules`.
    """
    f = utils.make_files_config(datainfo)

    # build dictionary of parameter names and their values
    # a parameter can be a column in the existing table...
    pars_dict = {}

    if table is not None:
        ok_types = (Array, ArrayOfEqualSizedArrays, VectorOfVectors)
        pars_dict = {k: v for k, v in table.items() if isinstance(v, ok_types)}

    # ...or defined through the configuration
    if parameters:
        pars_dict = pars_dict | parameters

    if mode == "function":
        # syntax:
        #
        #     pygama.evt.modules.spms.my_func([...], arg1=val, arg2=val)

        # get arguments list passed to the function (outermost parentheses)
        args_str = re.search(r"\((.*)\)$", expr.strip()).group(1)

        # handle tier scoping: evt.<>
        if f.evt.group is not None:
            args_str = args_str.replace(f.evt.group + ".", "")

        good_chns = [x for x in channels if x not in channels_skip]

        # replace stuff before first comma with list of mandatory args
        full_args_str = "datainfo, tcm, table_names," + ",".join(
            args_str.split(",")[1:]
        )

        # get module and function names
        func_call = expr.strip().split("(")[0]
        subpackage, func = func_call.rsplit(".", 1)
        package = subpackage.split(".")[0]

        # import function into current namespace
        log.debug(f"importing module {subpackage}")
        importlib.import_module(subpackage, package=__package__)

        # declare imported package as globals (see eval() call later)
        globs = {
            package: importlib.import_module(package),
        }

        # lookup dictionary for variables used in function arguments (see eval() call later)
        locs = {"datainfo": f, "tcm": tcm, "table_names": good_chns} | pars_dict

        # evil eval() to avoid annoying args casting logic
        call_str = f"{func_call}({full_args_str})"
        log.debug(f"evaluating {call_str}")
        log.debug(f"...globals={globs} and locals={locs}")
        log.debug(f"...locals={locs}")

        return eval(call_str, globs, locs)

    else:
        # find parameters in evt file or in parameters
        field_list = re.findall(
            rf"({f.evt.group}|{f.hit.group}|{f.dsp.group}).([a-zA-Z_$][\w$]*)", expr
        )

        # check if query is either on channel basis or evt basis (and not a mix)
        query_mask = query
        if query is not None:
            if f"{f.evt.group}." in query and (
                f"{f.hit.group}." in query or f"{f.dsp.group}." in query
            ):
                raise ValueError(
                    f"Query can't be a mix of {f.evt.group} tier and lower tiers."
                )

            # if it is an evt query we can evaluate it directly here
            if table and f"{f.evt.group}." in query:
                query_mask = eval(query.replace(f"{f.evt.group}.", ""), table)

        # switch through modes
        if table and (
            mode.startswith("keep_at_ch:") or mode.startswith("keep_at_idx:")
        ):
            if mode.startswith("keep_at_ch:"):
                ch_comp = table[mode[11:].replace(f"{f.evt.group}.", "")]
            else:
                ch_comp = table[mode[12:].replace(f"{f.evt.group}.", "")]
                if isinstance(ch_comp, Array):
                    ch_comp = Array(tcm.id[ch_comp.view_as("np")])
                elif isinstance(ch_comp, VectorOfVectors):
                    ch_comp = ch_comp.view_as("ak")
                    ch_comp = VectorOfVectors(
                        ak.unflatten(
                            tcm.id[ak.flatten(ch_comp)], ak.count(ch_comp, axis=-1)
                        )
                    )
                else:
                    raise NotImplementedError(
                        type(ch_comp)
                        + " not supported (only Array and VectorOfVectors are supported)"
                    )

            if isinstance(ch_comp, Array):
                return aggregators.evaluate_at_channel(
                    datainfo=datainfo,
                    tcm=tcm,
                    channels_skip=channels_skip,
                    expr=expr,
                    field_list=field_list,
                    ch_comp=ch_comp,
                    pars_dict=pars_dict,
                    default_value=default_value,
                )

            if isinstance(ch_comp, VectorOfVectors):
                return aggregators.evaluate_at_channel_vov(
                    datainfo=datainfo,
                    tcm=tcm,
                    expr=expr,
                    field_list=field_list,
                    ch_comp=ch_comp,
                    channels_skip=channels_skip,
                    pars_dict=pars_dict,
                    default_value=default_value,
                )

            raise NotImplementedError(
                "{type(ch_comp).__name__} not supported "
                "(only Array and VectorOfVectors are supported)"
            )

        if "first_at:" in mode or "last_at:" in mode:
            sorter = tuple(
                re.findall(
                    rf"({f.evt.group}|{f.hit.group}|{f.dsp.group}).([a-zA-Z_$][\w$]*)",
                    mode.split("first_at:")[-1],
                )[0]
            )
            return aggregators.evaluate_to_first_or_last(
                datainfo=datainfo,
                tcm=tcm,
                channels=channels,
                channels_skip=channels_skip,
                expr=expr,
                field_list=field_list,
                query=query_mask,
                n_rows=n_rows,
                sorter=sorter,
                pars_dict=pars_dict,
                default_value=default_value,
                is_first=True if "first_at:" in mode else False,
            )

        if mode in ["sum", "any", "all"]:
            return aggregators.evaluate_to_scalar(
                datainfo=datainfo,
                tcm=tcm,
                mode=mode,
                channels=channels,
                channels_skip=channels_skip,
                expr=expr,
                field_list=field_list,
                query=query_mask,
                n_rows=n_rows,
                pars_dict=pars_dict,
                default_value=default_value,
            )
        if "gather" == mode:
            return aggregators.evaluate_to_vector(
                datainfo=datainfo,
                tcm=tcm,
                channels=channels,
                channels_skip=channels_skip,
                expr=expr,
                field_list=field_list,
                query=query_mask,
                n_rows=n_rows,
                pars_dict=pars_dict,
                default_value=default_value,
                sorter=sorter,
            )

        raise ValueError(f"'{mode}' is not a valid mode")
