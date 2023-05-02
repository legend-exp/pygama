import json
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from pygama.flow import FileDB

config_dir = Path(__file__).parent / "configs"


def test_init_partial(lgnd_test_data, test_filedb_full):
    with open(config_dir / "filedb-config.json") as f:
        config = json.load(f)

    config["data_dir"] = lgnd_test_data.get_path("lh5/prod-ref-l200/generated/tier")
    fdb = FileDB(config, scan=False)
    fdb.scan_files("cal/p01/r014")
    fdb.scan_tables_columns()

    assert fdb.df.equals(test_filedb_full.df)


def test_filedb_basics(test_filedb):
    db = test_filedb

    assert list(db.df.keys()) == [
        "exp",
        "period",
        "run",
        "timestamp",
        "type",
        "raw_file",
        "dsp_file",
        "hit_file",
        "tcm_file",
        "evt_file",
        "raw_size",
        "dsp_size",
        "hit_size",
        "tcm_size",
        "evt_size",
        "file_status",
    ]

    assert db.df.values.tolist() == [
        [
            "l60",
            "p01",
            "r014",
            "20220716T104550Z",
            "cal",
            "/cal/p01/r014/l60-p01-r014-cal-20220716T104550Z-tier_raw.lh5",
            "/cal/p01/r014/l60-p01-r014-cal-20220716T104550Z-tier_dsp.lh5",
            "/cal/p01/r014/l60-p01-r014-cal-20220716T104550Z-tier_hit.lh5",
            "/cal/p01/r014/l60-p01-r014-cal-20220716T104550Z-tier_tcm.lh5",
            "/cal/p01/r014/l60-p01-r014-cal-20220716T104550Z-tier_evt.lh5",
            2141080,
            292032,
            280040,
            15728,
            0,
            int("0b11110", 2),
        ],
        [
            "l60",
            "p01",
            "r014",
            "20220716T105236Z",
            "cal",
            "/cal/p01/r014/l60-p01-r014-cal-20220716T105236Z-tier_raw.lh5",
            "/cal/p01/r014/l60-p01-r014-cal-20220716T105236Z-tier_dsp.lh5",
            "/cal/p01/r014/l60-p01-r014-cal-20220716T105236Z-tier_hit.lh5",
            "/cal/p01/r014/l60-p01-r014-cal-20220716T105236Z-tier_tcm.lh5",
            "/cal/p01/r014/l60-p01-r014-cal-20220716T105236Z-tier_evt.lh5",
            2141080,
            292032,
            280040,
            15728,
            0,
            int("0b11110", 2),
        ],
    ]


def test_scan_tables_columns(test_filedb_full):
    db = test_filedb_full

    assert list(db.df.keys()) == [
        "exp",
        "period",
        "run",
        "timestamp",
        "type",
        "raw_file",
        "dsp_file",
        "hit_file",
        "tcm_file",
        "evt_file",
        "raw_size",
        "dsp_size",
        "hit_size",
        "tcm_size",
        "evt_size",
        "file_status",
        "raw_tables",
        "raw_col_idx",
        "dsp_tables",
        "dsp_col_idx",
        "hit_tables",
        "hit_col_idx",
        "tcm_tables",
        "tcm_col_idx",
        "evt_tables",
        "evt_col_idx",
    ]

    assert db.columns == [
        [
            "baseline",
            "card",
            "ch_orca",
            "channel",
            "crate",
            "daqenergy",
            "deadtime",
            "dr_maxticks",
            "dr_start_pps",
            "dr_start_ticks",
            "dr_stop_pps",
            "dr_stop_ticks",
            "eventnumber",
            "fcid",
            "numtraces",
            "packet_id",
            "runtime",
            "timestamp",
            "to_abs_mu_usec",
            "to_dt_mu_usec",
            "to_master_sec",
            "to_mu_sec",
            "to_mu_usec",
            "to_start_sec",
            "to_start_usec",
            "tracelist",
            "ts_maxticks",
            "ts_pps",
            "ts_ticks",
            "waveform",
        ],
        [
            "bl_mean",
            "bl_std",
        ],
        ["hit_par1", "hit_par2"],
        ["array_id", "array_idx", "cumulative_length"],
    ]


def test_serialization(test_filedb_full):
    db = test_filedb_full
    db.to_disk("/tmp/filedb.lh5", wo_mode="of")

    with pytest.raises(RuntimeError):
        db.to_disk("/tmp/filedb.lh5")

    db2 = FileDB("/tmp/filedb.lh5")
    assert_frame_equal(db.df, db2.df)


def test_get_table_columns(test_filedb_full):
    db = test_filedb_full
    cols = db.get_table_columns(1, "dsp")
    assert cols == ["bl_mean", "bl_std"]
    with pytest.raises(KeyError):
        db.get_table_columns(1, "blah")
    with pytest.raises(ValueError):
        db.get_table_columns(9999, "raw")
