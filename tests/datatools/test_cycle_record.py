from __future__ import annotations

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from pygama.datatools import CycleRecord


def test_cycle_record():
    # Test CycleRecord __init__ and isinstance
    record = CycleRecord({"relpath": "abc/def", "cycle": "abd-def-ghi"})

    # dict should evaluate as a CycleRecord and convert to above
    record_dict = dict(record)
    assert isinstance(record_dict, CycleRecord)
    assert CycleRecord(record_dict) == record

    # ak.Record should evaluate as a CycleRecord and convert to above
    record_ak = ak.Record(record_dict)
    assert isinstance(record_ak, CycleRecord)
    assert CycleRecord(record_ak) == record

    # np.ndarray should evaluate as a CycleRecord and convert to above
    record_np = np.array(
        tuple(record_dict.values()),
        dtype={"names": list(record_dict), "formats": ["<U7", "<U11"]},
    )
    assert isinstance(record_np, CycleRecord)
    assert CycleRecord(record_np) == record

    # pd.Series should evaluate as a CycleRecord and convert to above
    record_pd = pd.Series(record_dict)
    assert isinstance(record_pd, CycleRecord)
    assert CycleRecord(record_pd) == record

    # test that isinstance returns false for non CycleRecords
    assert not isinstance(list(record_dict.items()), CycleRecord)
    bad_record = dict(record)
    del bad_record["cycle"]
    assert not isinstance(bad_record, CycleRecord)
    with pytest.raises(ValueError):
        CycleRecord(bad_record)


def test_get_cycle_record(test_refprod):
    record = CycleRecord.get_cycle_record(
        f"{test_refprod}/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_raw.lh5"
    )

    assert isinstance(record, CycleRecord)
    assert record["experiment"] == "l200"
    assert record["period"] == "p03"
    assert record["run"] == "r001"
    assert record["datatype"] == "cal"
    assert record["starttime"] == "20230318T012228Z"
    assert record["relpath"] == "cal/p03/r001"
    assert record["cycle"] == "l200-p03-r001-cal-20230318T012228Z"
    assert (
        record["tier_raw"]
        == f"{test_refprod}/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_raw.lh5"
    )
    assert (
        record["tier_dsp"]
        == f"{test_refprod}/generated/tier/dsp/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_dsp.lh5"
    )
    assert (
        record["tier_hit"]
        == f"{test_refprod}/generated/tier/hit/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_hit.lh5"
    )

    # needs to be a lh5 file
    with pytest.raises(ValueError):
        CycleRecord.get_cycle_record(
            f"{test_refprod}/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_raw.root"
        )

    # bad cycle def
    with pytest.raises(ValueError):
        CycleRecord.get_cycle_record(
            f"{test_refprod}/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_raw.lh5",
            cycle_def="experiment-period-run-datatype-starttype-what",
        )

    # cycle without file in all tiers
    record = CycleRecord.get_cycle_record(
        f"{test_refprod}/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )
    assert record["tier_raw"] is None
    with pytest.raises(FileNotFoundError):
        record = CycleRecord.get_cycle_record(
            f"{test_refprod}/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5",
            raise_on_missing=True,
        )

    # bad tier list
    with pytest.raises(ValueError):
        record = CycleRecord.get_cycle_record(
            f"{test_refprod}/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_raw.lh5",
            tiers=["raw", "dsp", "foo"],
        )
    with pytest.raises(ValueError):
        record = CycleRecord.get_cycle_record(
            f"{test_refprod}/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_raw.lh5",
            tiers=["dsp", "hit"],
        )


def test_get_tier_filepath(test_refprod):
    record_dict = {
        "relpath": "cal/p03/r000",
        "cycle": "l200-p03-r000-cal-20230311T235840Z",
    }
    record = CycleRecord(record_dict)

    assert (
        record.get_tier_filepath("raw")
        == f"{test_refprod}/generated/tier/raw/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_raw.lh5"
    )
    assert (
        CycleRecord.get_tier_filepath(record_dict, "raw")
        == f"{test_refprod}/generated/tier/raw/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_raw.lh5"
    )

    with pytest.raises(ValueError):
        record.get_tier_filepath("psp")

    bad_record = {"cycle": "l200-p03-r000-cal-20230311T235840Z"}
    with pytest.raises(ValueError):
        CycleRecord.get_tier_filepath(bad_record, "raw")
