from __future__ import annotations

from pygama.datatools import list_run_fields


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
