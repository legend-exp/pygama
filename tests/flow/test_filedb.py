import json
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from pygama.flow import FileDB

config_dir = Path(__file__).parent / "configs"


def test_chain_filedbs(lgnd_test_data, test_filedb_full, tmptestdir):
    with open(config_dir / "filedb-config.json") as f:
        config = json.load(f)

    config["data_dir"] = lgnd_test_data.get_path("lh5/prod-ref-l200/generated/tier")

    fdb1 = FileDB(config, scan=False)
    fdb1.scan_files("cal/p03/r001")
    fdb1.scan_tables_columns()
    fdb1.to_disk(f"{tmptestdir}/fdb1.h5")

    fdb2 = FileDB(config, scan=False)
    fdb2.scan_files("phy/p03/r001")
    fdb2.scan_tables_columns()
    fdb2.to_disk(f"{tmptestdir}/fdb2.h5")

    # columns from cal and phy data should be different
    assert fdb1.columns
    assert fdb2.columns
    assert fdb1.columns != fdb2.columns

    # test invariance to merge direction
    for fdb in [
        FileDB([f"{tmptestdir}/fdb1.h5", f"{tmptestdir}/fdb2.h5"]),
        FileDB([f"{tmptestdir}/fdb2.h5", f"{tmptestdir}/fdb1.h5"]),
    ]:
        # check that merging columns works
        merged_columns = []
        for it in fdb1.columns + fdb2.columns:
            if it not in merged_columns:
                merged_columns += [it]

        assert sorted(fdb.columns, key=lambda item: item[0]) == sorted(
            merged_columns, key=lambda item: item[0]
        )

        assert len(fdb.df) == len(fdb1.df) + len(fdb2.df)

        # cal comes first in time, then phy

        # hpge channel
        tbl = 1084804
        assert fdb.get_table_columns(tbl, "raw", ifile=0) == fdb1.get_table_columns(
            tbl, "raw", ifile=0
        )
        assert fdb.get_table_columns(tbl, "dsp", ifile=0) == fdb1.get_table_columns(
            tbl, "dsp", ifile=0
        )

        # at a later timestamp (phy), columns in dsp should be the same as in fdb2
        assert fdb.get_table_columns(tbl, "dsp", ifile=2) == fdb2.get_table_columns(
            tbl, "dsp", ifile=0
        )

        # sipm channel, not in cal
        tbl = 1057600
        assert fdb.get_table_columns(tbl, "raw", ifile=2) == fdb2.get_table_columns(
            tbl, "raw", ifile=0
        )
        assert fdb.get_table_columns(tbl, "hit", ifile=2) == fdb2.get_table_columns(
            tbl, "hit", ifile=0
        )


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
            "l200",
            "p03",
            "r001",
            "20230318T012144Z",
            "cal",
            "/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5",
            "/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_dsp.lh5",
            "/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_hit.lh5",
            "/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_tcm.lh5",
            "/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_evt.lh5",
            874456,
            484864,
            224776,
            15728,
            0,
            30,
        ],
        [
            "l200",
            "p03",
            "r001",
            "20230318T012228Z",
            "cal",
            "/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_raw.lh5",
            "/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_dsp.lh5",
            "/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_hit.lh5",
            "/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_tcm.lh5",
            "/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_evt.lh5",
            874456,
            484864,
            224776,
            15728,
            0,
            30,
        ],
        [
            "l200",
            "p03",
            "r001",
            "20230322T160139Z",
            "phy",
            "/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_raw.lh5",
            "/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_dsp.lh5",
            "/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_hit.lh5",
            "/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5",
            "/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5",
            1747968,
            601440,
            388368,
            15728,
            0,
            30,
        ],
        [
            "l200",
            "p03",
            "r001",
            "20230322T170202Z",
            "phy",
            "/phy/p03/r001/l200-p03-r001-phy-20230322T170202Z-tier_raw.lh5",
            "/phy/p03/r001/l200-p03-r001-phy-20230322T170202Z-tier_dsp.lh5",
            "/phy/p03/r001/l200-p03-r001-phy-20230322T170202Z-tier_hit.lh5",
            "/phy/p03/r001/l200-p03-r001-phy-20230322T170202Z-tier_tcm.lh5",
            "/phy/p03/r001/l200-p03-r001-phy-20230322T170202Z-tier_evt.lh5",
            1747968,
            601440,
            388368,
            15728,
            0,
            30,
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
            "abs_delta_mu_usec",
            "baseline",
            "board_id",
            "channel",
            "crate",
            "daqenergy",
            "deadtime",
            "delta_mu_usec",
            "dr_maxticks",
            "dr_start_pps",
            "dr_start_ticks",
            "dr_stop_pps",
            "dr_stop_ticks",
            "event_type",
            "eventnumber",
            "fc_input",
            "fcid",
            "mu_offset_sec",
            "mu_offset_usec",
            "numtraces",
            "packet_id",
            "runtime",
            "slot",
            "timestamp",
            "to_master_sec",
            "to_start_sec",
            "to_start_usec",
            "tracelist",
            "ts_maxticks",
            "ts_pps",
            "ts_ticks",
            "waveform",
        ],
        [
            "A_max",
            "QDrift",
            "baseline",
            "bl_intercept",
            "bl_mean",
            "bl_slope",
            "bl_slope_diff",
            "bl_slope_rms",
            "bl_std",
            "cuspEftp",
            "cuspEmax",
            "dt_eff",
            "dt_eff_invert",
            "lq80",
            "pz_mean",
            "pz_slope",
            "pz_slope_diff",
            "pz_slope_rms",
            "pz_std",
            "t_discharge",
            "t_sat_hi",
            "t_sat_lo",
            "timestamp",
            "tp_01",
            "tp_0_atrap",
            "tp_0_est",
            "tp_0_invert",
            "tp_10",
            "tp_100",
            "tp_100_invert",
            "tp_10_invert",
            "tp_20",
            "tp_20_invert",
            "tp_50",
            "tp_50_invert",
            "tp_80",
            "tp_80_invert",
            "tp_90",
            "tp_90_invert",
            "tp_95",
            "tp_99",
            "tp_99_invert",
            "tp_aoe_max",
            "tp_max",
            "tp_max_win",
            "tp_min",
            "tp_min_win",
            "trapEftp",
            "trapEmax",
            "trapSmax",
            "trapTftp_invert",
            "trapTmax",
            "trapTmax_invert",
            "wf_max",
            "wf_max_win",
            "wf_min",
            "wf_min_win",
            "zacEftp",
            "zacEmax",
        ],
        ["energies", "energies_dplms", "timestamp", "trigger_pos", "trigger_pos_dplms"],
        [
            "AoE_Classifier",
            "AoE_Corrected",
            "AoE_Double_Sided_Cut",
            "AoE_Low_Cut",
            "cuspEmax_ctc_cal",
            "is_discharge",
            "is_downgoing_baseline",
            "is_neg_energy",
            "is_negative",
            "is_negative_crosstalk",
            "is_noise_burst",
            "is_saturated",
            "is_upgoing_baseline",
            "is_valid_0vbb",
            "is_valid_baseline",
            "is_valid_cal",
            "is_valid_dteff",
            "is_valid_ediff",
            "is_valid_efrac",
            "is_valid_rt",
            "is_valid_t0",
            "is_valid_tail",
            "is_valid_tmax",
            "timestamp",
            "trapEmax_ctc_cal",
            "trapTmax_cal",
            "zacEmax_ctc_cal",
        ],
        [
            "energy_in_pe",
            "energy_in_pe_dplms",
            "is_valid_hit",
            "is_valid_hit_dplms",
            "timestamp",
            "trigger_pos",
            "trigger_pos_dplms",
        ],
        ["array_id", "array_idx", "cumulative_length"],
    ]


def test_serialization(test_filedb_full, tmptestdir):
    db = test_filedb_full
    db.to_disk(f"{tmptestdir}/filedb.lh5", wo_mode="of")

    with pytest.raises(RuntimeError):
        db.to_disk(f"{tmptestdir}/filedb.lh5")

    db2 = FileDB(f"{tmptestdir}/filedb.lh5")
    assert_frame_equal(db.df, db2.df)


def test_get_table_columns(test_filedb_full):
    db = test_filedb_full
    cols = db.get_table_columns(1084803, "dsp")
    assert cols == db.columns[1]
    with pytest.raises(KeyError):
        db.get_table_columns(1084803, "blah")
    with pytest.raises(ValueError):
        db.get_table_columns(9999, "raw")
