from pathlib import Path

import lgdo
import numpy as np
import pandas as pd
import pytest

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

    assert isinstance(data, lgdo.Struct)
    assert isinstance(data[0], lgdo.Table)
    assert len(data) == 4  # 4 files
    assert list(data[0].keys()) == ["hit_table", "hit_idx", "timestamp"]


def test_outputs(test_dl):
    test_dl.set_files("type == 'phy'")
    test_dl.set_datastreams([1057600, 1059201], "ch")
    test_dl.set_output(
        fmt="pd.DataFrame", columns=["timestamp", "channel", "energies", "energy_in_pe"]
    )
    data = test_dl.load()

    assert isinstance(data, pd.DataFrame)
    assert list(data.keys()) == [
        "hit_table",
        "hit_idx",
        "file",
        "timestamp",
        "channel",
        "energies",
        "energy_in_pe",
    ]


def test_any_mode(test_dl):
    test_dl.filedb.scan_tables_columns()
    test_dl.set_files("type == 'phy'")
    test_dl.set_cuts({"hit": "daqenergy == 10221"})
    el = test_dl.build_entry_list(tcm_level="tcm", mode="any")

    assert len(el) == 6


def test_set_files(test_dl):
    test_dl.set_files("timestamp == '20230318T012144Z'")
    test_dl.set_output(columns=["timestamp"], merge_files=False)
    data = test_dl.load()

    assert len(data) == 1


def test_set_keylist(test_dl):
    test_dl.set_files(["20230318T012144Z", "20230318T012228Z"])
    test_dl.set_output(columns=["timestamp"], merge_files=False)
    data = test_dl.load()

    assert len(data) == 2


def test_set_datastreams(test_dl):
    channels = [1084803, 1084804, 1121600]
    test_dl.set_files("timestamp == '20230318T012144Z'")
    test_dl.set_datastreams(channels, "ch")
    test_dl.set_output(columns=["eventnumber"], fmt="pd.DataFrame", merge_files=False)
    data = test_dl.load()

    assert np.array_equal(data[0]["hit_table"].unique(), channels)


def test_set_cuts(test_dl):
    test_dl.set_files("type == 'cal'")
    test_dl.set_cuts({"hit": "is_valid_cal == False"})
    test_dl.set_datastreams([1084803], "ch")
    test_dl.set_output(columns=["is_valid_cal"], fmt="pd.DataFrame")
    data = test_dl.load()

    assert (data.is_valid_cal == False).all()  # noqa: E712


def test_setter_overwrite(test_dl):
    test_dl.set_files("all")
    test_dl.set_datastreams([1084803, 1084804, 1121600], "ch")
    test_dl.set_cuts({"hit": "trapEmax > 5000"})
    test_dl.set_output(columns=["trapEmax"])

    data = test_dl.load().get_dataframe()

    test_dl.set_files("timestamp == '20230318T012144Z'")
    test_dl.set_datastreams([1084803, 1121600], "ch")
    test_dl.set_cuts({"hit": "trapEmax > 0"})

    data2 = test_dl.load().get_dataframe()

    assert 1084804 not in data2["hit_table"]
    assert len(pd.unique(data2["file"])) == 1
    assert len(data2.query("hit_table == 1084803")) > len(
        data.query("hit_table == 1084803")
    )


def test_browse(test_dl):
    test_dl.set_files("type == 'phy'")
    test_dl.set_datastreams([1057600, 1059201], "ch")
    test_dl.set_output(
        fmt="pd.DataFrame", columns=["timestamp", "channel", "energies", "energy_in_pe"]
    )
    wb = test_dl.browse()

    wb.draw_next()
