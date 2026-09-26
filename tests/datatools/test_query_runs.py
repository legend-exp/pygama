from __future__ import annotations

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from pygama.datatools import CycleRecord, list_run_fields, query_runs
from pygama.datatools.utils import _read_dataflow_config

CYCLES_ALL_TIERS = {
    "l200-p03-r001-cal-20230318T012144Z",
    "l200-p03-r001-cal-20230318T012228Z",
    "l200-p03-r001-phy-20230322T160139Z",
    "l200-p03-r001-phy-20230322T170202Z",
}
CYCLES_DSP_ONLY = {
    "l200-p03-r000-cal-20230311T235840Z",
}
CYCLES_RAW_ONLY = {
    "l200-p18-r000-cal-20251107T191821Z",
    "l200-p18-r000-cal-20251107T192416Z",
    "l200-p18-r003-phy-20251126T011947Z",
}
CYCLES_IGNORED = {
    "l200-p13-r007-aph-20250101T003931Z",
    "l200-p14-r004-cal-20250606T010224Z",
}


def test_query_runs(test_refprod):  # noqa: ARG001
    # check calling with different datatypes; check that
    # entries evaluate as CycleRecords
    run_list = query_runs()
    assert isinstance(run_list, ak.Array)
    assert len(run_list) == len(CYCLES_ALL_TIERS)
    assert isinstance(run_list[0], CycleRecord)

    run_list = query_runs(library="pd")
    assert isinstance(run_list, pd.DataFrame)
    assert len(run_list) == len(CYCLES_ALL_TIERS)
    assert isinstance(run_list.loc[0], CycleRecord)

    run_list = query_runs(library="np")
    assert isinstance(run_list, np.ndarray)
    assert len(run_list) == len(CYCLES_ALL_TIERS)
    assert isinstance(run_list[0], CycleRecord)

    with pytest.raises(ValueError):
        query_runs(library="invalid")


def test_query_runs_filtering(test_refprod):  # noqa: ARG001
    # Basic filter
    res = query_runs("period == 'p03'")
    assert len(res) == len({c for c in CYCLES_ALL_TIERS if "p03" in c})
    assert ak.all(res.period == "p03")

    # Complex logic
    res = query_runs("period == 'p03' and datatype == 'cal'")
    assert ak.all(res.period == "p03")
    assert ak.all(res.datatype == "cal")

    # Membership filter
    res = query_runs("starttime in ['20230318T012228Z', '20230322T160139Z']")
    assert len(res) == 2


def test_query_runs_join(test_refprod):
    res = query_runs(join="inner")
    assert set(res.cycle) == CYCLES_ALL_TIERS

    res = query_runs(join="outer")
    assert set(res.cycle) == CYCLES_ALL_TIERS | CYCLES_DSP_ONLY | CYCLES_RAW_ONLY
    assert all(
        (rec["tier_dsp"] is None) == (rec.cycle in CYCLES_RAW_ONLY) for rec in res
    )
    assert all(
        (rec["tier_raw"] is None) == (rec.cycle in CYCLES_DSP_ONLY) for rec in res
    )
    assert all(
        (rec["tier_hit"] is None) == (rec.cycle in (CYCLES_DSP_ONLY | CYCLES_RAW_ONLY))
        for rec in res
    )

    res = query_runs(join="dsp")
    assert set(res.cycle) == CYCLES_ALL_TIERS | CYCLES_DSP_ONLY
    assert all(rec["tier_dsp"] is not None for rec in res)
    assert all(
        (rec["tier_raw"] is None) == (rec.cycle in CYCLES_DSP_ONLY) for rec in res
    )
    assert all(
        (rec["tier_hit"] is None) == (rec.cycle in CYCLES_DSP_ONLY) for rec in res
    )

    res = query_runs(join="raw")
    assert set(res.cycle) == CYCLES_ALL_TIERS | CYCLES_RAW_ONLY
    assert all(
        (rec["tier_dsp"] is None) == (rec.cycle in CYCLES_RAW_ONLY) for rec in res
    )
    assert all(rec["tier_raw"] is not None for rec in res)
    assert all(
        (rec["tier_hit"] is None) == (rec.cycle in CYCLES_RAW_ONLY) for rec in res
    )

    with pytest.raises(ValueError):
        query_runs(join="invalid")

    # same tests, with a tier that has no files
    df_config, _, _ = _read_dataflow_config()
    df_config["paths"]["tier_new"] = f"{test_refprod}/generated/tier/new"

    res = query_runs(join="inner", tiers=["raw", "new"], dataflow_config=df_config)
    assert len(res) == 0

    res = query_runs(join="outer", tiers=["raw", "new"], dataflow_config=df_config)
    assert set(res.cycle) == CYCLES_ALL_TIERS | CYCLES_RAW_ONLY
    assert all(rec["tier_new"] is None for rec in res)

    res = query_runs(join="outer", tiers=["new", "raw"], dataflow_config=df_config)
    assert set(res.cycle) == CYCLES_ALL_TIERS | CYCLES_RAW_ONLY
    assert all(rec["tier_new"] is None for rec in res)


def test_query_runs_sorting(test_refprod):  # noqa: ARG001
    # Single field sort
    res = query_runs(join="outer", sort_by="datatype")
    assert all(res.datatype[i] <= res.datatype[i + 1] for i in range(len(res) - 1))


def test_query_runs_grouping(test_refprod):  # noqa: ARG001
    # Group by datatype
    res = query_runs(group_by="datatype")
    assert isinstance(res, ak.Array)
    assert len(res) == 2
    assert list(res.datatype) == ["cal", "phy"]
    assert list(ak.num(res.cycle, axis=-1)) == [2, 2]

    # multi-group
    res = query_runs(group_by=("period", "run", "datatype"))
    assert isinstance(res, ak.Array)
    assert len(res) == 2
    assert list(res.datatype) == ["cal", "phy"]
    assert list(res.period) == ["p03", "p03"]
    assert list(res.run) == ["r001", "r001"]
    assert list(ak.num(res.cycle, axis=-1)) == [2, 2]


def test_query_runs_ignored_cycles(test_refprod):  # noqa: ARG001
    res = query_runs(join="outer")
    assert set(res.cycle) == CYCLES_ALL_TIERS | CYCLES_DSP_ONLY | CYCLES_RAW_ONLY

    res = query_runs(join="outer", ignored_cycles=[])
    assert (
        set(res.cycle)
        == CYCLES_ALL_TIERS | CYCLES_DSP_ONLY | CYCLES_RAW_ONLY | CYCLES_IGNORED
    )


def test_multiprocessing(test_refprod):  # noqa: ARG001
    res1 = query_runs()
    res2 = query_runs(processes=2)
    assert all(ak.all(res1[f] == res2[f]) for f in res1.fields)


def test_list_run_fields():
    dataflow_config = {
        "paths": {
            "tier_raw": "/tmp/raw",
            "tier_dsp": "/tmp/dsp",
        },
        "query": {
            "cycle_def": "experiment-period-run-datatype",
            "tiers": ["raw", "dsp"],
        },
    }

    fields = list_run_fields(dataflow_config=dataflow_config)
    assert fields == {
        "relpath",
        "cycle",
        "experiment",
        "period",
        "run",
        "datatype",
        "tier_raw",
        "tier_dsp",
    }

    fields = list_run_fields(
        dataflow_config=dataflow_config, tiers=["raw"], cycle_def="experiment-run"
    )
    assert fields == {
        "relpath",
        "cycle",
        "experiment",
        "run",
        "tier_raw",
    }
