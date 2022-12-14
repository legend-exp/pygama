import json
import os
import sys
from pathlib import Path

import numpy as np

import pygama.lgdo as lgdo
from pygama.raw.build_raw import build_raw

config_dir = Path(__file__).parent / "test_data_trimmer_configs"


# check that packet indexes match in verification test
def test_data_trimmer_packet_ids(lgnd_test_data):

    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca")
    dsp_config = f"{config_dir}/data_trimmer_config.json"

    trimmed_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca", "L200-comm-20220519-phy-geds_trim.lh5"
    )

    build_raw(in_stream=daq_file, overwrite=True, trim_config=dsp_config)
    build_raw(in_stream=daq_file, overwrite=True)

    raw_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca", "L200-comm-20220519-phy-geds.lh5"
    )

    sto = lgdo.LH5Store()

    raw_group = "ORFlashCamADCWaveform"
    raw_packet_ids, _ = sto.read_object(str(raw_group) + "/packet_id", raw_file)
    trimmed_packet_ids, _ = sto.read_object(str(raw_group) + "/packet_id", trimmed_file)

    assert np.array_equal(raw_packet_ids.nda, trimmed_packet_ids.nda)


# check that packet indexes match in verification test
def test_data_trimmer_waveform_lengths(lgnd_test_data):

    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")
    trimmed_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-20211130-phy-spms_trim.lh5"
    )
    raw_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-20211130-phy-spms.lh5"
    )

    out_spec = {
        "FCEventDecoder": {
            "ch{key}": {
                "key_list": [[0, 6]],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            }
        }
    }

    dsp_config = """
    {
        "outputs" : [ "presummed_waveform" ],
        "processors" : {
            "windowed_waveform": {
            "start_index": 1000,
            "end_index": -1000
            },
            "presummed_waveform": {
                "function": "presum",
                "module": "pygama.dsp.processors",
                "args": ["waveform", "presummed_waveform(len(waveform)/4, 'f')"],
                "unit": "ADC"
            }
        }
    }
    """

    build_raw(
        in_stream=daq_file, out_spec=out_spec, overwrite=True, trim_config=dsp_config
    )
    build_raw(in_stream=daq_file, out_spec=out_spec, overwrite=True)

    lh5_tables = lgdo.ls(raw_file)
    # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
    for i, tb in enumerate(lh5_tables):
        if "raw" not in tb and lgdo.ls(raw_file, f"{tb}/raw"):
            lh5_tables[i] = f"{tb}/raw"
        elif not lgdo.ls(raw_file, tb):
            del lh5_tables[i]

    if isinstance(dsp_config, str) and dsp_config.endswith(".json"):
        f = open(dsp_config)
        jsonfile = json.load(f)
        f.close()
    # If we get a string that is in the correct format as a json file
    elif isinstance(dsp_config, str):
        jsonfile = json.loads(dsp_config)
    # Or we could get a dict as the config
    elif isinstance(dsp_config, dict):
        jsonfile = dsp_config

    # Read in the presummed rate from the config file to modify the clock rate later
    presum_rate_string = jsonfile["processors"]["presummed_waveform"]["args"][1]
    presum_rate_start_idx = presum_rate_string.find("/") + 1
    presum_rate_end_idx = presum_rate_string.find(",")
    presum_rate = int(presum_rate_string[presum_rate_start_idx:presum_rate_end_idx])

    # This needs to be overwritten with the correct windowing values set in data_trimmer.py
    window_start_index = 1000
    window_end_index = 1000

    sto = lgdo.LH5Store()

    for raw_group in lh5_tables:

        raw_packet_waveform_values = sto.read_object(
            str(raw_group) + "/waveform/values", raw_file
        )
        presummed_packet_waveform_values = sto.read_object(
            str(raw_group) + "/presummed_waveform/values", trimmed_file
        )
        windowed_packet_waveform_values = sto.read_object(
            str(raw_group) + "/windowed_waveform/values", trimmed_file
        )

        # Check that the lengths of the waveforms match what we expect
        assert len(raw_packet_waveform_values[0].nda[0]) == presum_rate * len(
            presummed_packet_waveform_values[0].nda[0]
        )
        assert isinstance(
            presummed_packet_waveform_values[0].nda[0][0], np.float32
        )  # change this to np.uint32 when we merge unnorm_presum
        assert len(raw_packet_waveform_values[0].nda[0]) == len(
            windowed_packet_waveform_values[0].nda[0]
        ) + np.abs(window_start_index) + np.abs(window_end_index)
        assert isinstance(windowed_packet_waveform_values[0].nda[0][0], np.uint16)

        raw_packet_waveform_t0s, _ = sto.read_object(
            str(raw_group) + "/waveform/t0", raw_file
        )
        raw_packet_waveform_dts, _ = sto.read_object(
            str(raw_group) + "/waveform/dt", raw_file
        )

        windowed_packet_waveform_t0s, _ = sto.read_object(
            str(raw_group) + "/windowed_waveform/t0", trimmed_file
        )
        presummed_packet_waveform_t0s, _ = sto.read_object(
            str(raw_group) + "/presummed_waveform/t0", trimmed_file
        )

        # Check that the t0s match what we expect, with the correct units
        assert np.array_equal(
            raw_packet_waveform_t0s.nda,
            windowed_packet_waveform_t0s.nda
            - window_start_index * raw_packet_waveform_dts.nda,
        )
        assert (
            windowed_packet_waveform_t0s.attrs["units"]
            == raw_packet_waveform_t0s.attrs["units"]
        )
        assert np.array_equal(
            raw_packet_waveform_t0s.nda, presummed_packet_waveform_t0s.nda
        )
        assert (
            presummed_packet_waveform_t0s.attrs["units"]
            == raw_packet_waveform_t0s.attrs["units"]
        )

        presummed_packet_waveform_dts, _ = sto.read_object(
            str(raw_group) + "/presummed_waveform/dt", trimmed_file
        )

        # Check that the dts match what we expect, with the correct units
        assert np.array_equal(
            raw_packet_waveform_dts.nda, presummed_packet_waveform_dts.nda / presum_rate
        )


def test_data_trimmer_file_size_decrease(lgnd_test_data):
    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("orca/sis3316/coherent-run1141-bkg.orca")
    dsp_config = f"{config_dir}/data_trimmer_config.json"
    trimmed_file = daq_file.replace(
        "coherent-run1141-bkg.orca", "coherent-run1141-bkg_trim.lh5"
    )
    raw_file = daq_file.replace("coherent-run1141-bkg.orca", "coherent-run1141-bkg.lh5")

    out_spec = {
        "ORSIS3316WaveformDecoder": {
            "Card1": {"key_list": [48], "out_stream": raw_file, "out_name": "raw"}
        }
    }

    build_raw(
        in_stream=daq_file, out_spec=out_spec, overwrite=True, trim_config=dsp_config
    )
    build_raw(in_stream=daq_file, out_spec=out_spec, overwrite=True)

    lh5_tables = lgdo.ls(raw_file)
    for i, tb in enumerate(lh5_tables):
        if "raw" not in tb and lgdo.ls(raw_file, f"{tb}/raw"):
            lh5_tables[i] = f"{tb}/raw"
        elif not lgdo.ls(raw_file, tb):
            del lh5_tables[i]
    sto = lgdo.LH5Store()

    wf_size = 0

    for raw_group in lh5_tables:
        wf_size += sys.getsizeof(
            sto.read_object(str(raw_group) + "/waveform/values", raw_file)[0].nda
        )

    # Make sure we are taking up less space than a file that has two copies of the waveform table in it
    assert os.path.getsize(trimmed_file) < os.path.getsize(raw_file) + wf_size


# check that packet indexes match in verification test on file that has both spms and geds
def test_data_trimmer_separate_name_tables(lgnd_test_data):

    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")
    trimmed_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-fake-geds-and-spms_trim.lh5"
    )
    raw_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-fake-geds-and-spms.lh5"
    )

    out_spec = {
        "FCEventDecoder": {
            "geds": {
                "key_list": [[0, 3]],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            },
            "spms": {
                "key_list": [[3, 6]],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            },
        }
    }

    dsp_config = """
    {
        "spms": {
        "outputs" : [ "presummed_waveform" ],
        "processors" : {
            "windowed_waveform": {
            "start_index": 1000,
            "end_index": -1000
            },
            "presummed_waveform": {
                "function": "presum",
                "module": "pygama.dsp.processors",
                "args": ["waveform", "presummed_waveform(len(waveform)/4, 'f')"],
                "unit": "ADC"
                }
            }
        },

        "geds": {
        "outputs" : [ "presummed_waveform" ],
        "processors" : {
            "windowed_waveform": {
            "start_index": 2000,
            "end_index": -1000
            },
            "presummed_waveform": {
                "function": "presum",
                "module": "pygama.dsp.processors",
                "args": ["waveform", "presummed_waveform(len(waveform)/8, 'f')"],
                "unit": "ADC"
                }
            }
        }
    }
    """

    build_raw(
        in_stream=daq_file, out_spec=out_spec, overwrite=True, trim_config=dsp_config
    )
    build_raw(in_stream=daq_file, out_spec=out_spec, overwrite=True)

    lh5_tables = lgdo.ls(raw_file)
    # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
    for i, tb in enumerate(lh5_tables):
        if "raw" not in tb and lgdo.ls(raw_file, f"{tb}/raw"):
            lh5_tables[i] = f"{tb}/raw"
        elif not lgdo.ls(raw_file, tb):
            del lh5_tables[i]

    if isinstance(dsp_config, str) and dsp_config.endswith(".json"):
        f = open(dsp_config)
        jsonfile = json.load(f)
        f.close()
    # If we get a string that is in the correct format as a json file
    elif isinstance(dsp_config, str):
        jsonfile = json.loads(dsp_config)
    # Or we could get a dict as the config
    elif isinstance(dsp_config, dict):
        jsonfile = dsp_config

    sto = lgdo.LH5Store()

    for raw_group in lh5_tables:

        # First, check the packet ids
        raw_packet_ids, _ = sto.read_object(str(raw_group) + "/packet_id", raw_file)
        trimmed_packet_ids, _ = sto.read_object(
            str(raw_group) + "/packet_id", trimmed_file
        )

        assert np.array_equal(raw_packet_ids.nda, trimmed_packet_ids.nda)

        # Read in the presummed rate from the config file to modify the clock rate later
        group_name = raw_group.split("/raw")[0]
        presum_rate_string = jsonfile[group_name]["processors"]["presummed_waveform"][
            "args"
        ][1]
        presum_rate_start_idx = presum_rate_string.find("/") + 1
        presum_rate_end_idx = presum_rate_string.find(",")
        presum_rate = int(presum_rate_string[presum_rate_start_idx:presum_rate_end_idx])

        # This needs to be overwritten with the correct windowing values set in data_trimmer.py
        window_start_index = int(
            jsonfile[group_name]["processors"]["windowed_waveform"]["start_index"]
        )
        window_end_index = int(
            jsonfile[group_name]["processors"]["windowed_waveform"]["end_index"]
        )

        raw_packet_waveform_values = sto.read_object(
            str(raw_group) + "/waveform/values", raw_file
        )
        presummed_packet_waveform_values = sto.read_object(
            str(raw_group) + "/presummed_waveform/values", trimmed_file
        )
        windowed_packet_waveform_values = sto.read_object(
            str(raw_group) + "/windowed_waveform/values", trimmed_file
        )

        # Check that the lengths of the waveforms match what we expect
        assert len(raw_packet_waveform_values[0].nda[0]) == presum_rate * len(
            presummed_packet_waveform_values[0].nda[0]
        )
        assert isinstance(
            presummed_packet_waveform_values[0].nda[0][0], np.float32
        )  # change this to np.uint32 when we merge unnorm_presum
        assert len(raw_packet_waveform_values[0].nda[0]) == len(
            windowed_packet_waveform_values[0].nda[0]
        ) + np.abs(window_start_index) + np.abs(window_end_index)
        assert isinstance(windowed_packet_waveform_values[0].nda[0][0], np.uint16)

        raw_packet_waveform_t0s, _ = sto.read_object(
            str(raw_group) + "/waveform/t0", raw_file
        )
        raw_packet_waveform_dts, _ = sto.read_object(
            str(raw_group) + "/waveform/dt", raw_file
        )

        windowed_packet_waveform_t0s, _ = sto.read_object(
            str(raw_group) + "/windowed_waveform/t0", trimmed_file
        )
        presummed_packet_waveform_t0s, _ = sto.read_object(
            str(raw_group) + "/presummed_waveform/t0", trimmed_file
        )

        # Check that the t0s match what we expect, with the correct units
        assert (
            raw_packet_waveform_t0s.nda[0]
            == windowed_packet_waveform_t0s.nda[0]
            - raw_packet_waveform_dts.nda[0] * window_start_index
        )
        assert (
            windowed_packet_waveform_t0s.attrs["units"]
            == raw_packet_waveform_t0s.attrs["units"]
        )
        assert raw_packet_waveform_t0s.nda[0] == presummed_packet_waveform_t0s.nda[0]
        assert (
            presummed_packet_waveform_t0s.attrs["units"]
            == raw_packet_waveform_t0s.attrs["units"]
        )

        presummed_packet_waveform_dts, _ = sto.read_object(
            str(raw_group) + "/presummed_waveform/dt", trimmed_file
        )

        # Check that the dts match what we expect, with the correct units
        assert (
            raw_packet_waveform_dts.nda[0]
            == presummed_packet_waveform_dts.nda[0] / presum_rate
        )
