import json
import os
import re
import sys
from pathlib import Path

import numpy as np

import pygama.lgdo as lgdo
from pygama.dsp import build_processing_chain as bpc
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


def test_trim_geds_no_trim_spms(lgnd_test_data):
    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")
    trimmed_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-test-pass_trim.lh5"
    )
    raw_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-test-pass.lh5"
    )

    out_spec = {
        "FCEventDecoder": {
            "geds": {
                "key_list": [[0, 1]],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            },
            "spms": {
                "key_list": [[3, 4]],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            },
        }
    }

    dsp_config = """
    {
        "spms": "pass",

        "geds": {
        "outputs": [ "presummed_waveform", "t_sat_lo", "t_sat_hi" ],
        "processors": {
            "windowed_waveform": {
            "start_index": 2000,
            "end_index": -1000
                },
            "presummed_waveform": {
                "function": "presum",
                "module": "pygama.dsp.processors",
                "args": ["waveform", "presummed_waveform(len(waveform)/16, 'f')"],
                "unit": "ADC"
                },
            "t_sat_lo, t_sat_hi": {
                "function": "saturation",
                "module": "pygama.dsp.processors",
                "args": ["waveform", 16, "t_sat_lo", "t_sat_hi"],
                "unit": "ADC"
                }
            }
        }
    }
    """

    raw_dsp_config = """
    {
        "outputs": ["t_sat_lo", "t_sat_hi"],
        "processors": {
            "t_sat_lo, t_sat_hi": {
                "function": "saturation",
                "module": "pygama.dsp.processors",
                "args": ["waveform", 16, "t_sat_lo", "t_sat_hi"],
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
        pass_flag = False
        # If the user passes trimming on a group, then the presum_rate is just 1 and there is no windowing
        if (type(jsonfile[group_name]) == str) & (jsonfile[group_name] == "pass"):
            presum_rate = 1
            window_start_index = 0
            window_end_index = 0
            pass_flag = True
        else:
            presum_rate_string = jsonfile[group_name]["processors"][
                "presummed_waveform"
            ]["args"][1]
            presum_rate_start_idx = presum_rate_string.find("/") + 1
            presum_rate_end_idx = presum_rate_string.find(",")
            presum_rate = int(
                presum_rate_string[presum_rate_start_idx:presum_rate_end_idx]
            )

            # This needs to be overwritten with the correct windowing values set in data_trimmer.py
            window_start_index = int(
                jsonfile[group_name]["processors"]["windowed_waveform"]["start_index"]
            )
            window_end_index = int(
                jsonfile[group_name]["processors"]["windowed_waveform"]["end_index"]
            )

        # Read in the waveforms
        raw_packet_waveform_values = sto.read_object(
            str(raw_group) + "/waveform/values", raw_file
        )
        if pass_flag:
            presummed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/waveform/values", trimmed_file
            )
            windowed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/waveform/values", trimmed_file
            )
        else:
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

        if pass_flag:
            windowed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/waveform/t0", trimmed_file
            )
            presummed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/waveform/t0", trimmed_file
            )
        else:
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

        if pass_flag:

            presummed_packet_waveform_dts, _ = sto.read_object(
                str(raw_group) + "/waveform/dt", trimmed_file
            )
        else:

            presummed_packet_waveform_dts, _ = sto.read_object(
                str(raw_group) + "/presummed_waveform/dt", trimmed_file
            )
        # Check that the dts match what we expect, with the correct units
        assert (
            raw_packet_waveform_dts.nda[0]
            == presummed_packet_waveform_dts.nda[0] / presum_rate
        )

        # check that the t_lo_sat and t_sat_hi are correct
        if not pass_flag:
            wf_table, _ = sto.read_object(str(raw_group), raw_file)
            pc, _, wf_out = bpc(wf_table, json.loads(raw_dsp_config))
            pc.execute()
            raw_sat_lo = wf_out["t_sat_lo"]
            raw_sat_hi = wf_out["t_sat_hi"]

            trim_sat_lo, _ = sto.read_object(str(raw_group) + "/t_sat_lo", trimmed_file)

            trim_sat_hi, _ = sto.read_object(str(raw_group) + "/t_sat_hi", trimmed_file)

            assert np.array_equal(raw_sat_lo.nda, trim_sat_lo.nda)
            assert np.array_equal(raw_sat_hi.nda, trim_sat_hi.nda)
            assert type(trim_sat_lo.nda[0]) == np.uint16


# check that packet indexes match in verification test
def test_data_trimmer_multiple_keys(lgnd_test_data):

    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca")
    trimmed_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca",
        "L200-comm-20220519-phy-geds-key-test_trim.lh5",
    )
    raw_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca", "L200-comm-20220519-phy-geds-key-test.lh5"
    )

    out_spec = {
        "ORFlashCamADCWaveformDecoder": {
            "ch{key}": {
                "key_list": [0, 1, 3, 4],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            }
        }
    }

    dsp_config = """
    {
        "ch3, ch4": "pass",

        "ch0, ch1": {
        "outputs": [ "presummed_waveform", "t_sat_lo", "t_sat_hi" ],
        "processors": {
            "windowed_waveform": {
            "start_index": 2000,
            "end_index": -1000
                },
            "presummed_waveform": {
                "function": "presum",
                "module": "pygama.dsp.processors",
                "args": ["waveform", "presummed_waveform(len(waveform)/16, 'f')"],
                "unit": "ADC"
                },
            "t_sat_lo, t_sat_hi": {
                "function": "saturation",
                "module": "pygama.dsp.processors",
                "args": ["waveform", 16, "t_sat_lo", "t_sat_hi"],
                "unit": "ADC"
                }
            }
        }
    }
    """
    raw_dsp_config = """
    {
        "outputs": ["t_sat_lo", "t_sat_hi"],
        "processors": {
            "t_sat_lo, t_sat_hi": {
                "function": "saturation",
                "module": "pygama.dsp.processors",
                "args": ["waveform", 16, "t_sat_lo", "t_sat_hi"],
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

    multi_key_dsp_dict = {}
    # Now check the RawBuffer's group and see if that there is a matching key in the dsp_dict, then take that sub dictionary.
    for key, node in jsonfile.items():
        # if we have multiple outputs, add each to the processesors list
        keys = [k for k in re.split(",| ", key) if k != ""]
        if len(keys) >= 1:
            for k in keys:
                multi_key_dsp_dict[k] = node

    jsonfile = multi_key_dsp_dict

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

        pass_flag = False
        # If the user passes trimming on a group, then the presum_rate is just 1 and there is no windowing
        if (type(jsonfile[group_name]) == str) & (jsonfile[group_name] == "pass"):
            presum_rate = 1
            window_start_index = 0
            window_end_index = 0
            pass_flag = True
        else:
            presum_rate_string = jsonfile[group_name]["processors"][
                "presummed_waveform"
            ]["args"][1]
            presum_rate_start_idx = presum_rate_string.find("/") + 1
            presum_rate_end_idx = presum_rate_string.find(",")
            presum_rate = int(
                presum_rate_string[presum_rate_start_idx:presum_rate_end_idx]
            )

            # This needs to be overwritten with the correct windowing values set in data_trimmer.py
            window_start_index = int(
                jsonfile[group_name]["processors"]["windowed_waveform"]["start_index"]
            )
            window_end_index = int(
                jsonfile[group_name]["processors"]["windowed_waveform"]["end_index"]
            )

        # Read in the waveforms
        raw_packet_waveform_values = sto.read_object(
            str(raw_group) + "/waveform/values", raw_file
        )
        if pass_flag:
            presummed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/waveform/values", trimmed_file
            )
            windowed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/waveform/values", trimmed_file
            )
        else:
            presummed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/presummed_waveform/values", trimmed_file
            )
            windowed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/windowed_waveform/values", trimmed_file
            )

        # Check that the lengths of the waveforms match what we expect
        assert (
            len(raw_packet_waveform_values[0].nda[0])
            // len(presummed_packet_waveform_values[0].nda[0])
            == presum_rate
        )
        assert len(raw_packet_waveform_values[0].nda[0]) == len(
            windowed_packet_waveform_values[0].nda[0]
        ) + np.abs(window_start_index) + np.abs(window_end_index)
        assert isinstance(windowed_packet_waveform_values[0].nda[0][0], np.uint16)

        # Check that the waveforms match
        if group_name == "ch3" or group_name == "ch4":
            raw_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/waveform/values", raw_file
            )
            windowed_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/waveform/values", trimmed_file
            )
            assert np.array_equal(
                raw_packet_waveform_values.nda, windowed_packet_waveform_values.nda
            )
        else:
            raw_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/waveform/values", raw_file
            )
            windowed_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/windowed_waveform/values", trimmed_file
            )
            assert np.array_equal(
                raw_packet_waveform_values.nda[:, window_start_index:window_end_index],
                windowed_packet_waveform_values.nda,
            )

        # Check the t0 and dts are what we expect
        raw_packet_waveform_t0s, _ = sto.read_object(
            str(raw_group) + "/waveform/t0", raw_file
        )
        raw_packet_waveform_dts, _ = sto.read_object(
            str(raw_group) + "/waveform/dt", raw_file
        )

        if pass_flag:
            windowed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/waveform/t0", trimmed_file
            )
            presummed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/waveform/t0", trimmed_file
            )
        else:
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

        if pass_flag:

            presummed_packet_waveform_dts, _ = sto.read_object(
                str(raw_group) + "/waveform/dt", trimmed_file
            )
        else:

            presummed_packet_waveform_dts, _ = sto.read_object(
                str(raw_group) + "/presummed_waveform/dt", trimmed_file
            )
        # Check that the dts match what we expect, with the correct units
        assert (
            raw_packet_waveform_dts.nda[0]
            == presummed_packet_waveform_dts.nda[0] / presum_rate
        )

        # check that the t_lo_sat and t_sat_hi are correct
        if not pass_flag:
            wf_table, _ = sto.read_object(str(raw_group), raw_file)
            pc, _, wf_out = bpc(wf_table, json.loads(raw_dsp_config))
            pc.execute()
            raw_sat_lo = wf_out["t_sat_lo"]
            raw_sat_hi = wf_out["t_sat_hi"]

            trim_sat_lo, _ = sto.read_object(str(raw_group) + "/t_sat_lo", trimmed_file)

            trim_sat_hi, _ = sto.read_object(str(raw_group) + "/t_sat_hi", trimmed_file)

            assert np.array_equal(raw_sat_lo.nda, trim_sat_lo.nda)
            assert np.array_equal(raw_sat_hi.nda, trim_sat_hi.nda)
            assert type(trim_sat_lo.nda[0]) == np.uint16


def test_data_trimmer_all_pass(lgnd_test_data):
    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca")

    raw_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca", "L200-comm-20220519-phy-geds-all-pass.lh5"
    )

    trimmed_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca",
        "L200-comm-20220519-phy-geds-all-pass_trim.lh5",
    )

    dsp_config = """
    {
        "/": "pass"
    }
    """

    build_raw(
        in_stream=daq_file, out_spec=raw_file, overwrite=True, trim_config=dsp_config
    )
    build_raw(in_stream=daq_file, out_spec=raw_file, overwrite=True)

    # assert filecmp.cmp(raw_file, trimmed_file, shallow=True)
    sto = lgdo.LH5Store()
    raw_tables = lgdo.ls(raw_file)
    for tb in raw_tables:
        raw, _ = sto.read_object(tb, raw_file)
        trim, _ = sto.read_object(tb, trimmed_file)
        if isinstance(raw, lgdo.Scalar):
            raw_value = raw.value
            raw_attrs = raw.attrs
            trim_value = trim.value
            trim_attrs = trim.attrs
            assert raw_value == trim_value
            assert raw_attrs == trim_attrs
        else:
            for obj in raw.keys():
                if not isinstance(raw[obj], lgdo.Table):
                    raw_df = raw.get_dataframe([obj])
                    trim_df = trim.get_dataframe([obj])
                else:
                    for sub_obj in raw[obj].keys():
                        raw_df = raw[obj].get_dataframe([str(sub_obj)])
                        trim_df = trim[obj].get_dataframe([str(sub_obj)])

            assert raw_df.equals(trim_df)
