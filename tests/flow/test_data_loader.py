from pathlib import Path

import pandas as pd
import pytest

from pygama import lgdo
from pygama.flow import DataLoader

config_dir = Path(__file__).parent / "configs"


@pytest.fixture(scope="function")
def test_dl(test_filedb):
    return DataLoader(f"{config_dir}/data-loader-config.json", test_filedb)


def test_init(test_dl):
    pass


def test_init_variants(test_filedb):
    assert DataLoader(str(config_dir), test_filedb).config is not None
    assert (
        DataLoader(
            f"{config_dir}/nested/data-loader-config-nested.json[nest1/nest2]",
            test_filedb,
        ).config
        is not None
    )
    assert (
        DataLoader(f"{config_dir}/nested[nest1/nest2]", test_filedb).config is not None
    )


def test_simple_load(test_dl):
    test_dl.set_files("all")
    test_dl.set_output(columns=["timestamp"])
    data = test_dl.load()

    assert isinstance(data, lgdo.Table)
    assert list(data.keys()) == ["hit_table", "hit_idx", "file", "timestamp"]


def test_simple_chunked_load(test_dl):
    test_dl.set_files("all")
    test_dl.set_output(columns=["timestamp"])
    for data in test_dl.next(chunk_size=2):
        assert len(data) == 2
        assert isinstance(data, lgdo.Table)
        assert list(data.keys()) == ["hit_table", "hit_idx", "file", "timestamp"]


def test_load_wfs(test_dl):
    test_dl.set_files("all")
    test_dl.set_output(columns=["waveform"], fmt="lgdo.Table")
    data = test_dl.load()
    assert isinstance(data, lgdo.Table)
    assert list(data.keys()) == ["hit_table", "hit_idx", "file", "waveform"]

    test_dl.set_output(columns=["waveform"], fmt="pd.DataFrame")
    data = test_dl.load()
    assert isinstance(data, pd.DataFrame)
    assert list(data.keys()) == [
        "hit_table",
        "hit_idx",
        "file",
        "waveform_t0",
        "waveform_dt",
        "waveform_values",
    ]


def test_no_merge(test_dl):
    test_dl.set_files("all")
    test_dl.set_output(columns=["timestamp"], merge_files=False)
    data = test_dl.load()

    assert isinstance(data, dict)
    assert isinstance(data[0], lgdo.Table)
    assert len(data) == 2
    assert list(data[0].keys()) == ["hit_table", "hit_idx", "timestamp"]


def test_outputs(test_dl):
    test_dl.set_files("all")
    test_dl.set_output(
        fmt="pd.DataFrame", columns=["timestamp", "channel", "bl_mean", "hit_par1"]
    )
    data = test_dl.load()

    assert isinstance(data, pd.DataFrame)
    assert list(data.keys()) == [
        "hit_table",
        "hit_idx",
        "file",
        "timestamp",
        "channel",
        "bl_mean",
        "hit_par1",
    ]


def test_any_mode(test_dl):
    test_dl.filedb.scan_tables_columns()
    test_dl.set_files("all")
    test_dl.set_cuts({"hit": "daqenergy == 634"})
    el = test_dl.build_entry_list(tcm_level="tcm", mode="any")

    assert len(el) == 42


def test_set_files(test_dl):
    test_dl.set_files("timestamp == '20220716T104550Z'")
    test_dl.set_output(columns=["timestamp"], merge_files=False)
    data = test_dl.load()

    assert len(data) == 1


def test_set_keylist(test_dl):
    test_dl.set_files(["20220716T104550Z", "20220716T104550Z"])
    test_dl.set_output(columns=["timestamp"], merge_files=False)
    data = test_dl.load()

    assert len(data) == 1


def test_set_datastreams(test_dl):
    test_dl.set_files("all")
    test_dl.set_datastreams([1, 3, 8], "ch")
    test_dl.set_output(columns=["channel"], merge_files=False)
    data = test_dl.load()

    assert (data[0]["hit_table"].nda == [1, 3, 8]).all()
    assert (data[0]["channel"].nda == [1, 3, 8]).all()


def test_set_cuts(test_dl):
    test_dl.set_files("all")
    test_dl.set_cuts({"hit": "card == 3"})
    test_dl.set_output(columns=["card"])
    data = test_dl.load()

    assert (data["hit_table"].nda == [12, 13, 12, 13]).all()


def test_browse(test_dl):
    test_dl.set_files("all")
    test_dl.set_output(
        fmt="pd.DataFrame", columns=["timestamp", "channel", "bl_mean", "hit_par1"]
    )
    wb = test_dl.browse()

    wb.draw_next()
