"""
This module implements routines to build the `evt` tier.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from importlib import import_module

import awkward as ak
import numpy as np
from lgdo import Array, ArrayOfEqualSizedArrays, Table, VectorOfVectors, lh5
from lgdo.lh5 import LH5Store

from . import aggregators, utils

log = logging.getLogger(__name__)


def build_evt(
    f_tcm: str,
    f_dsp: str,
    f_hit: str,
    evt_config: str | dict,
    f_evt: str | None = None,
    wo_mode: str = "write_safe",
    evt_group: str = "evt",
    tcm_group: str = "hardware_tcm_1",
    dsp_group: str = "dsp",
    hit_group: str = "hit",
    tcm_id_table_pattern: str = "ch{}",
) -> None | Table:
    """Transform data from the `hit` and `dsp` levels which a channel sorted to a
    event sorted data format.

    Parameters
    ----------
    f_tcm
        input LH5 file of the `tcm` level.
    f_dsp
        input LH5 file of the `dsp` level.
    f_hit
        input LH5 file of the `hit` level.
    evt_config
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
                  "expression": "dsp.wf_max>a",
                  "parameters": {"a":15100},
                  "initial": false
                },
                "multiplicity":{
                  "channels":  ["geds_on", "geds_no_psd", "geds_ac"],
                  "aggregation_mode": "sum",
                  "expression": "hit.cuspEmax_ctc_cal > a",
                  "parameters": {"a":25},
                  "initial": 0
                },
                "t0":{
                  "aggregation_mode": "keep_at_ch:evt.energy_id",
                  "expression": "dsp.tp_0_est"
                },
                "lar_energy":{
                  "channels": "spms_on",
                  "aggregation_mode": "function",
                  "expression": ".modules.spm.get_energy(0.5, evt.t0, 48000, 1000, 5000)"
                },
              }
            }

    f_evt
        name of the output file. If ``None``, return the output :class:`.Table`
        instead of writing to disk.
    wo_mode
        writing mode.
    evt group
        LH5 root group name of `evt` tier.
    tcm_group
        LH5 root group in `tcm` file.
    dsp_group
        LH5 root group in `dsp` file.
    hit_group
        LH5 root group in `hit` file.
    tcm_id_table_pattern
        pattern to format `tcm` id values to table name in higher tiers. Must
        have one placeholder which is the `tcm` id.
    """

    store = LH5Store()
    tbl_cfg = evt_config
    if not isinstance(tbl_cfg, (str, dict)):
        raise TypeError()
    if isinstance(tbl_cfg, str):
        with open(tbl_cfg) as f:
            tbl_cfg = json.load(f)

    if "channels" not in tbl_cfg.keys():
        raise ValueError("channel field needs to be specified in the config")
    if "operations" not in tbl_cfg.keys():
        raise ValueError("operations field needs to be specified in the config")

    # check tcm_id_table_pattern validity
    pattern_check = re.findall(r"{([^}]*?)}", tcm_id_table_pattern)
    if len(pattern_check) != 1:
        raise ValueError(
            f"tcm_id_table_pattern must have exactly one placeholder. {tcm_id_table_pattern} is invalid."
        )
    elif "{" in pattern_check[0] or "}" in pattern_check[0]:
        raise ValueError(
            f"tcm_id_table_pattern {tcm_id_table_pattern} has an invalid placeholder."
        )

    if (
        utils.get_table_name_by_pattern(
            tcm_id_table_pattern,
            utils.get_tcm_id_by_pattern(tcm_id_table_pattern, lh5.ls(f_hit)[0]),
        )
        != lh5.ls(f_hit)[0]
    ):
        raise ValueError(
            f"tcm_id_table_pattern {tcm_id_table_pattern} does not match keys in data!"
        )

    # create channel list according to config
    # This can be either read from the meta data
    # or a list of channel names
    log.debug("Creating channel dictionary")

    chns = {}

    for k, v in tbl_cfg["channels"].items():
        if isinstance(v, dict):
            # it is a meta module. module_name must exist
            if "module" not in v.keys():
                raise ValueError(
                    "Need module_name to load channel via a meta data module"
                )

            attr = {}
            # the time_key argument is set to the time key of the DSP file
            # in case it is not provided by the config
            if "time_key" not in v.keys():
                attr["time_key"] = re.search(r"\d{8}T\d{6}Z", f_dsp).group(0)

            # if "None" do None
            elif "None" == v["time_key"]:
                attr["time_key"] = None

            # load module
            p, m = v["module"].rsplit(".", 1)
            met = getattr(import_module(p, package=__package__), m)
            chns[k] = met(v | attr)

        elif isinstance(v, str):
            chns[k] = [v]

        elif isinstance(v, list):
            chns[k] = [e for e in v]

    nrows = store.read_n_rows(f"/{tcm_group}/cumulative_length", f_tcm)

    table = Table(size=nrows)

    for k, v in tbl_cfg["operations"].items():
        log.debug("Processing field " + k)

        # if mode not defined in operation, it can only be an operation on the evt level.
        if "aggregation_mode" not in v.keys():
            var = {}
            if "parameters" in v.keys():
                var = var | v["parameters"]
            res = table.eval(v["expression"].replace(f"{evt_group}.", ""), var)

            # add attribute if present
            if "lgdo_attrs" in v.keys():
                res.attrs |= v["lgdo_attrs"]

            table.add_field(k, res)

        # Else we build the event entry
        else:
            if "channels" not in v.keys():
                chns_e = []
            elif isinstance(v["channels"], str):
                chns_e = chns[v["channels"]]
            elif isinstance(v["channels"], list):
                chns_e = list(
                    itertools.chain.from_iterable([chns[e] for e in v["channels"]])
                )
            chns_rm = []
            if "exclude_channels" in v.keys():
                if isinstance(v["exclude_channels"], str):
                    chns_rm = chns[v["exclude_channels"]]
                elif isinstance(v["exclude_channels"], list):
                    chns_rm = list(
                        itertools.chain.from_iterable(
                            [chns[e] for e in v["exclude_channels"]]
                        )
                    )

            pars, qry, defaultv, srter = None, None, np.nan, None
            if "parameters" in v.keys():
                pars = v["parameters"]
            if "query" in v.keys():
                qry = v["query"]
            if "initial" in v.keys():
                defaultv = v["initial"]
                if isinstance(defaultv, str) and (
                    defaultv in ["np.nan", "np.inf", "-np.inf"]
                ):
                    defaultv = eval(defaultv)
            if "sort" in v.keys():
                srter = v["sort"]

            obj = evaluate_expression(
                f_tcm=f_tcm,
                f_hit=f_hit,
                f_dsp=f_dsp,
                chns=chns_e,
                chns_rm=chns_rm,
                mode=v["aggregation_mode"],
                expr=v["expression"],
                nrows=nrows,
                table=table,
                para=pars,
                qry=qry,
                defv=defaultv,
                sorter=srter,
                tcm_id_table_pattern=tcm_id_table_pattern,
                evt_group=evt_group,
                hit_group=hit_group,
                dsp_group=dsp_group,
                tcm_group=tcm_group,
            )

            # add attribute if present
            if "lgdo_attrs" in v.keys():
                obj.attrs |= v["lgdo_attrs"]

            table.add_field(k, obj)

    # write output fields into f_evt
    if "outputs" in tbl_cfg.keys():
        if len(tbl_cfg["outputs"]) < 1:
            log.warning("No output fields specified, no file will be written.")
            return table
        else:
            clms_to_remove = [e for e in table.keys() if e not in tbl_cfg["outputs"]]
            for fld in clms_to_remove:
                table.remove_field(fld, True)

            if f_evt:
                store.write(
                    obj=table, name=f"/{evt_group}/", lh5_file=f_evt, wo_mode=wo_mode
                )
            else:
                return table
    else:
        log.warning("No output fields specified, no file will be written.")

    key = re.search(r"\d{8}T\d{6}Z", f_hit).group(0)
    log.info(
        f"Applied {len(tbl_cfg['operations'])} operations to key {key} and saved "
        f"{len(tbl_cfg['outputs'])} evt fields across {len(chns)} channel groups"
    )


def evaluate_expression(
    f_tcm: str,
    f_hit: str,
    f_dsp: str,
    chns: list,
    chns_rm: list,
    mode: str,
    expr: str,
    nrows: int,
    table: Table = None,
    para: dict = None,
    qry: str = None,
    defv: bool | int | float = np.nan,
    sorter: str = None,
    tcm_id_table_pattern: str = "ch{}",
    evt_group: str = "evt",
    hit_group: str = "hit",
    dsp_group: str = "dsp",
    tcm_group: str = "tcm",
) -> Array | ArrayOfEqualSizedArrays | VectorOfVectors:
    """Evaluates the expression defined by the user across all channels
    according to the mode.

    Parameters
    ----------
    f_tcm
       path to `tcm` tier file.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    chns
       list of channel names across which expression gets evaluated (form:
       ``ch<rawid>``).
    chns_rm
       list of channels which get set to default value during evaluation. In
       function mode they are removed entirely (form: ``ch<rawid>``)
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
       - ``gather``: Channels are not combined, but result saved as
         :class:`.VectorOfVectors`.

    qry
       a query that can mask the aggregation.
    expr
       the expression. That can be any mathematical equation/comparison. If
       `mode` is ``function``, the expression needs to be a special processing
       function defined in modules (e.g. :func:`.modules.spm.get_energy`). In
       the expression parameters from either hit, dsp, evt tier (from
       operations performed before this one! Dictionary operations order
       matters), or from the ``parameters`` field can be used.
    nrows
       number of rows to be processed.
    table
       table of `evt` tier data.
    para
       dictionary of parameters defined in the ``parameters`` field in the
       configuration dictionary.
    defv
       default value of evaluation.
    sorter
       can be used to sort vector outputs according to sorter expression (see
       :func:`evaluate_to_vector`).
    tcm_id_table_pattern
        pattern to format tcm id values to table name in higher tiers. Must have one
        placeholder which is the `tcm` id.
    evt group
        LH5 root group name of `evt` tier.
    tcm_group
        LH5 root group in `tcm` file.
    dsp_group
        LH5 root group in `dsp` file.
    hit_group
        LH5 root group in `hit` file.
    """

    store = LH5Store()

    # find parameters in evt file or in parameters
    exprl = re.findall(
        rf"({evt_group}|{hit_group}|{dsp_group}).([a-zA-Z_$][\w$]*)", expr
    )
    var_ph = {}
    if table:
        var_ph = var_ph | {
            e: table[e].view_as("ak")
            for e in table.keys()
            if isinstance(table[e], (Array, ArrayOfEqualSizedArrays, VectorOfVectors))
        }
    if para:
        var_ph = var_ph | para

    if mode == "function":
        # evaluate expression
        func, params = expr.split("(")
        params = (
            params.replace(f"{dsp_group}.", f"{dsp_group}_")
            .replace(f"{hit_group}.", f"{hit_group}_")
            .replace(f"{evt_group}.", "")
        )
        params = [
            f_hit,
            f_dsp,
            f_tcm,
            hit_group,
            dsp_group,
            tcm_group,
            tcm_id_table_pattern,
            [x for x in chns if x not in chns_rm],
        ] + [utils.num_and_pars(e, var_ph) for e in params[:-1].split(",")]

        # load function dynamically
        p, m = func.rsplit(".", 1)
        met = getattr(import_module(p, package=__package__), m)
        return met(*params)

    else:
        # check if query is either on channel basis or evt basis (and not a mix)
        qry_mask = qry
        if qry is not None:
            if f"{evt_group}." in qry and (
                f"{hit_group}." in qry or f"{dsp_group}." in qry
            ):
                raise ValueError(
                    f"Query can't be a mix of {evt_group} tier and lower tiers."
                )

            # if it is an evt query we can evaluate it directly here
            if table and f"{evt_group}." in qry:
                qry_mask = eval(qry.replace(f"{evt_group}.", ""), table)

        # load TCM data to define an event
        ids = store.read(f"/{tcm_group}/array_id", f_tcm)[0].view_as("np")
        idx = store.read(f"/{tcm_group}/array_idx", f_tcm)[0].view_as("np")
        cumulength = store.read(f"/{tcm_group}/cumulative_length", f_tcm)[0].view_as(
            "np"
        )

        # switch through modes
        if table and (("keep_at_ch:" == mode[:11]) or ("keep_at_idx:" == mode[:12])):
            if "keep_at_ch:" == mode[:11]:
                ch_comp = table[mode[11:].replace(f"{evt_group}.", "")]
            else:
                ch_comp = table[mode[12:].replace(f"{evt_group}.", "")]
                if isinstance(ch_comp, Array):
                    ch_comp = Array(nda=ids[ch_comp.view_as("np")])
                elif isinstance(ch_comp, VectorOfVectors):
                    ch_comp = ch_comp.view_as("ak")
                    ch_comp = VectorOfVectors(
                        array=ak.unflatten(
                            ids[ak.flatten(ch_comp)], ak.count(ch_comp, axis=-1)
                        )
                    )
                else:
                    raise NotImplementedError(
                        type(ch_comp)
                        + " not supported (only Array and VectorOfVectors are supported)"
                    )

            if isinstance(ch_comp, Array):
                return aggregators.evaluate_at_channel(
                    cumulength=cumulength,
                    idx=idx,
                    ids=ids,
                    f_hit=f_hit,
                    f_dsp=f_dsp,
                    chns_rm=chns_rm,
                    expr=expr,
                    exprl=exprl,
                    ch_comp=ch_comp,
                    var_ph=var_ph,
                    defv=defv,
                    tcm_id_table_pattern=tcm_id_table_pattern,
                    evt_group=evt_group,
                    hit_group=hit_group,
                    dsp_group=dsp_group,
                )
            elif isinstance(ch_comp, VectorOfVectors):
                return aggregators.evaluate_at_channel_vov(
                    cumulength=cumulength,
                    idx=idx,
                    ids=ids,
                    f_hit=f_hit,
                    f_dsp=f_dsp,
                    expr=expr,
                    exprl=exprl,
                    ch_comp=ch_comp,
                    chns_rm=chns_rm,
                    var_ph=var_ph,
                    defv=defv,
                    tcm_id_table_pattern=tcm_id_table_pattern,
                    evt_group=evt_group,
                    hit_group=hit_group,
                    dsp_group=dsp_group,
                )
            else:
                raise NotImplementedError(
                    type(ch_comp)
                    + " not supported (only Array and VectorOfVectors are supported)"
                )
        elif "first_at:" in mode or "last_at:" in mode:
            sorter = tuple(
                re.findall(
                    rf"({evt_group}|{hit_group}|{dsp_group}).([a-zA-Z_$][\w$]*)",
                    mode.split("first_at:")[-1],
                )[0]
            )
            return aggregators.evaluate_to_first_or_last(
                cumulength=cumulength,
                idx=idx,
                ids=ids,
                f_hit=f_hit,
                f_dsp=f_dsp,
                chns=chns,
                chns_rm=chns_rm,
                expr=expr,
                exprl=exprl,
                qry=qry_mask,
                nrows=nrows,
                sorter=sorter,
                var_ph=var_ph,
                defv=defv,
                is_first=True if "first_at:" in mode else False,
                tcm_id_table_pattern=tcm_id_table_pattern,
                evt_group=evt_group,
                hit_group=hit_group,
                dsp_group=dsp_group,
            )
        elif mode in ["sum", "any", "all"]:
            return aggregators.evaluate_to_scalar(
                mode=mode,
                cumulength=cumulength,
                idx=idx,
                ids=ids,
                f_hit=f_hit,
                f_dsp=f_dsp,
                chns=chns,
                chns_rm=chns_rm,
                expr=expr,
                exprl=exprl,
                qry=qry_mask,
                nrows=nrows,
                var_ph=var_ph,
                defv=defv,
                tcm_id_table_pattern=tcm_id_table_pattern,
                evt_group=evt_group,
                hit_group=hit_group,
                dsp_group=dsp_group,
            )
        elif "gather" == mode:
            return aggregators.evaluate_to_vector(
                cumulength=cumulength,
                idx=idx,
                ids=ids,
                f_hit=f_hit,
                f_dsp=f_dsp,
                chns=chns,
                chns_rm=chns_rm,
                expr=expr,
                exprl=exprl,
                qry=qry_mask,
                nrows=nrows,
                var_ph=var_ph,
                defv=defv,
                sorter=sorter,
                tcm_id_table_pattern=tcm_id_table_pattern,
                evt_group=evt_group,
                hit_group=hit_group,
                dsp_group=dsp_group,
            )
        else:
            raise ValueError(mode + " not a valid mode")
