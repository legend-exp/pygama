import os
from pathlib import Path

import pytest

from pygama.lgdo.lh5_store import LH5Store, ls
from pygama.raw import build_raw

config_dir = Path(__file__).parent / "configs"


def test_build_raw_basics(lgnd_test_data):
    with pytest.raises(FileNotFoundError):
        build_raw(in_stream="non-existent-file")

    with pytest.raises(FileNotFoundError):
        build_raw(
            in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
            out_spec="non-existent-file.json",
        )


def test_build_raw_fc(lgnd_test_data):
    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        overwrite=True,
    )

    assert lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.lh5") != ""

    out_file = "/tmp/L200-comm-20211130-phy-spms.lh5"

    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        out_spec=out_file,
        overwrite=True,
    )

    assert os.path.exists("/tmp/L200-comm-20211130-phy-spms.lh5")


def test_build_raw_fc_out_spec(lgnd_test_data):
    out_file = "/tmp/L200-comm-20211130-phy-spms.lh5"
    out_spec = {
        "FCEventDecoder": {"spms": {"key_list": [[2, 4]], "out_stream": out_file}}
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        out_spec=out_spec,
        n_max=10,
        overwrite=True,
    )

    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/spms", out_file)
    assert n_rows == 10
    assert (lh5_obj["channel"].nda == [2, 3, 4, 2, 3, 4, 2, 3, 4, 2]).all()

    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        out_spec=f"{config_dir}/fc-out-spec.json",
        n_max=10,
        overwrite=True,
    )


def test_build_raw_fc_channelwise_out_spec(lgnd_test_data):
    out_file = "/tmp/L200-comm-20211130-phy-spms.lh5"
    out_spec = {
        "FCEventDecoder": {
            "ch{key}": {
                "key_list": [[0, 6]],
                "out_stream": out_file + ":{name}",
                "out_name": "raw",
            }
        }
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        out_spec=out_spec,
        overwrite=True,
    )

    assert ls(out_file) == ["ch0", "ch1", "ch2", "ch3", "ch4", "ch5"]
    assert ls(out_file, "ch0/") == ["ch0/raw"]
    assert ls(out_file, "ch0/raw/waveform") == ["ch0/raw/waveform"]


def test_build_raw_orca(lgnd_test_data):
    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        overwrite=True,
    )

    assert lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.lh5") != ""

    out_file = "/tmp/L200-comm-20220519-phy-geds.lh5"

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        out_spec=out_file,
        overwrite=True,
    )

    assert os.path.exists("/tmp/L200-comm-20220519-phy-geds.lh5")


def test_build_raw_orca_out_spec(lgnd_test_data):
    out_file = "/tmp/L200-comm-20220519-phy-geds.lh5"
    out_spec = {
        "ORFlashCamADCWaveformDecoder": {
            "geds": {"key_list": [[2, 4]], "out_stream": out_file}
        }
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        out_spec=out_spec,
        n_max=10,
        overwrite=True,
    )

    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/geds", out_file)
    assert n_rows == 10
    assert (lh5_obj["channel"].nda == [2, 3, 4, 2, 3, 4, 2, 3, 4, 2]).all()

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        out_spec=f"{config_dir}/orca-out-spec.json",
        n_max=10,
        overwrite=True,
    )


def test_build_raw_overwrite(lgnd_test_data):
    with pytest.raises(FileExistsError):
        build_raw(
            in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")
        )


def test_build_raw_orca_sis3316(lgnd_test_data):
    out_file = "/tmp/coherent-run1141-bkg.lh5"
    out_spec = {
        "ORSIS3316WaveformDecoder": {
            "Card1": {"key_list": [48], "out_stream": out_file}
        }
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/sis3316/coherent-run1141-bkg.orca"),
        out_spec=out_spec,
        n_max=10,
        overwrite=True,
    )

    assert os.path.exists(out_file)
