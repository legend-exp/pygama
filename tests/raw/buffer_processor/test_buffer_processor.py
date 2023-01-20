import json
import os
import sys
from pathlib import Path

import numpy as np

import pygama.lgdo as lgdo
from pygama.dsp import build_processing_chain as bpc
from pygama.raw.build_raw import build_raw

config_dir = Path(__file__).parent / "test_buffer_processor_configs"


# check that packet indexes match in verification test
def test_buffer_processor_packet_ids(lgnd_test_data):

    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca")
    proc_out_spec = f"{config_dir}/buffer_processor_config.json"
    raw_out_spec = f"{config_dir}/raw_out_spec_no_proc.json"

    processed_file = "/tmp/L200-comm-20220519-phy-geds_proc.lh5"

    build_raw(in_stream=daq_file, out_spec=proc_out_spec, overwrite=True)
    build_raw(in_stream=daq_file, out_spec=raw_out_spec, overwrite=True)

    raw_file = "/tmp/L200-comm-20220519-phy-geds.lh5"

    sto = lgdo.LH5Store()

    raw_group = "ORFlashCamADCWaveform"
    raw_packet_ids, _ = sto.read_object(str(raw_group) + "/packet_id", raw_file)
    processed_packet_ids, _ = sto.read_object(
        str(raw_group) + "/packet_id", processed_file
    )

    assert np.array_equal(raw_packet_ids.nda, processed_packet_ids.nda)

    processed_presummed_wfs, _ = sto.read_object(
        str(raw_group) + "/presummed_waveform/values", processed_file
    )
    raw_wfs, _ = sto.read_object(str(raw_group) + "/waveform/values", raw_file)
    assert processed_presummed_wfs.nda[0][0] == np.sum(raw_wfs.nda[0][:4])


# check that packet indexes match in verification test
def test_buffer_processor_waveform_lengths(lgnd_test_data):

    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")
    processed_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-20211130-phy-spms_proc.lh5"
    )
    raw_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-20211130-phy-spms.lh5"
    )

    out_spec = {
        "FCEventDecoder": {
            "ch{key}": {
                "key_list": [[0, 6]],
                "out_stream": processed_file + ":{name}",
                "out_name": "raw",
                "proc_spec": {
                    "window": ["waveform", 1000, -1000, "windowed_waveform"],
                    "dsp_config": {
                        "outputs": ["presum_rate", "presummed_waveform"],
                        "processors": {
                            "presum_rate, presummed_waveform": {
                                "function": "presum",
                                "module": "pygama.dsp.processors",
                                "args": [
                                    "waveform",
                                    0,
                                    "presum_rate",
                                    "presummed_waveform(shape=len(waveform)/16, period=waveform.period*16, offset=waveform.offset)",
                                ],
                                "unit": "ADC",
                            }
                        },
                    },
                    "drop": ["waveform"],
                    "dtype_conv": {
                        "presummed_waveform/values": "uint32",
                        "presum_rate": "uint16",
                    },
                },
            }
        }
    }

    copy_out_spec = {
        "FCEventDecoder": {
            "ch{key}": {
                "key_list": [[0, 6]],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            }
        }
    }

    build_raw(in_stream=daq_file, out_spec=out_spec, overwrite=True)

    proc_spec = out_spec["FCEventDecoder"]["ch0"].pop("proc_spec")
    dsp_config = proc_spec["dsp_config"]
    window_config = proc_spec["window"]

    build_raw(in_stream=daq_file, out_spec=copy_out_spec, overwrite=True)

    lh5_tables = lgdo.ls(raw_file)
    # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
    for i, tb in enumerate(lh5_tables):
        if "raw" not in tb and lgdo.ls(raw_file, f"{tb}/raw"):
            lh5_tables[i] = f"{tb}/raw"
        elif not lgdo.ls(raw_file, tb):
            del lh5_tables[i]

    jsonfile = dsp_config

    # Read in the presummed rate from the config file to modify the clock rate later
    presum_rate_string = jsonfile["processors"]["presum_rate, presummed_waveform"][
        "args"
    ][3]
    presum_rate_start_idx = presum_rate_string.find("/") + 1
    presum_rate_end_idx = presum_rate_string.find(",")
    presum_rate = int(presum_rate_string[presum_rate_start_idx:presum_rate_end_idx])

    # This needs to be overwritten with the correct windowing values set in buffer_processor.py
    window_start_index = window_config[1]
    window_end_index = window_config[2]

    sto = lgdo.LH5Store()

    for raw_group in lh5_tables:

        raw_packet_waveform_values = sto.read_object(
            str(raw_group) + "/waveform/values", raw_file
        )
        presummed_packet_waveform_values = sto.read_object(
            str(raw_group) + "/presummed_waveform/values", processed_file
        )
        windowed_packet_waveform_values = sto.read_object(
            str(raw_group) + "/windowed_waveform/values", processed_file
        )

        # Check that the lengths of the waveforms match what we expect
        assert len(raw_packet_waveform_values[0].nda[0]) == presum_rate * len(
            presummed_packet_waveform_values[0].nda[0]
        )
        assert isinstance(presummed_packet_waveform_values[0].nda[0][0], np.uint32)
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
            str(raw_group) + "/windowed_waveform/t0", processed_file
        )
        presummed_packet_waveform_t0s, _ = sto.read_object(
            str(raw_group) + "/presummed_waveform/t0", processed_file
        )

        windowed_packet_waveform_dts, _ = sto.read_object(
            str(raw_group) + "/windowed_waveform/dt", processed_file
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
            str(raw_group) + "/presummed_waveform/dt", processed_file
        )

        # Check that the dts match what we expect, with the correct units
        assert np.array_equal(
            raw_packet_waveform_dts.nda, presummed_packet_waveform_dts.nda / presum_rate
        )

        assert np.array_equal(
            raw_packet_waveform_dts.nda, windowed_packet_waveform_dts.nda
        )

        # Check that the presum_rate is correctly identified
        presum_rate_from_file, _ = sto.read_object(
            str(raw_group) + "/presum_rate", processed_file
        )
        assert presum_rate_from_file.nda[0] == presum_rate


def test_buffer_processor_file_size_decrease(lgnd_test_data):
    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("orca/sis3316/coherent-run1141-bkg.orca")
    processed_file = daq_file.replace(
        "coherent-run1141-bkg.orca", "coherent-run1141-bkg_proc.lh5"
    )
    raw_file = daq_file.replace("coherent-run1141-bkg.orca", "coherent-run1141-bkg.lh5")

    raw_out_spec = {
        "ORSIS3316WaveformDecoder": {
            "Card1": {"key_list": [48], "out_stream": raw_file, "out_name": "raw"}
        }
    }

    proc_out_spec = {
        "ORSIS3316WaveformDecoder": {
            "Card1": {
                "key_list": [48],
                "out_stream": processed_file,
                "out_name": "raw",
                "proc_spec": {
                    "window": ["waveform", 1000, -1000, "windowed_waveform"],
                    "dsp_config": {
                        "outputs": ["presum_rate", "presummed_waveform"],
                        "processors": {
                            "presum_rate, presummed_waveform": {
                                "function": "presum",
                                "module": "pygama.dsp.processors",
                                "args": [
                                    "waveform",
                                    0,
                                    "presum_rate",
                                    "presummed_waveform(shape=len(waveform)/4, period=waveform.period*4, offset=waveform.offset)",
                                ],
                                "unit": "ADC",
                            }
                        },
                    },
                    "drop": ["waveform"],
                    "dtype_conv": {
                        "presummed_waveform/values": "uint32",
                        "presum_rate": "uint16",
                    },
                },
            }
        }
    }

    build_raw(in_stream=daq_file, out_spec=proc_out_spec, overwrite=True)
    build_raw(in_stream=daq_file, out_spec=raw_out_spec, overwrite=True)

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
    assert os.path.getsize(processed_file) < os.path.getsize(raw_file) + wf_size


# check that packet indexes match in verification test on file that has both spms and geds
def test_buffer_processor_separate_name_tables(lgnd_test_data):

    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")
    processed_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-fake-geds-and-spms_proc.lh5"
    )
    raw_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-fake-geds-and-spms.lh5"
    )

    out_spec = {
        "FCEventDecoder": {
            "geds": {
                "key_list": [[0, 3]],
                "out_stream": processed_file + ":{name}",
                "out_name": "raw",
                "proc_spec": {
                    "window": ["waveform", 2000, -1000, "windowed_waveform"],
                    "dsp_config": {
                        "outputs": ["presum_rate", "presummed_waveform"],
                        "processors": {
                            "presum_rate, presummed_waveform": {
                                "function": "presum",
                                "module": "pygama.dsp.processors",
                                "args": [
                                    "waveform",
                                    0,
                                    "presum_rate",
                                    "presummed_waveform(shape=len(waveform)/8, period=waveform.period*8, offset=waveform.offset)",
                                ],
                                "unit": "ADC",
                            }
                        },
                    },
                    "drop": ["waveform"],
                    "dtype_conv": {
                        "presummed_waveform/values": "uint32",
                        "presum_rate": "uint16",
                    },
                },
            },
            "spms": {
                "key_list": [[3, 6]],
                "out_stream": processed_file + ":{name}",
                "out_name": "raw",
                "proc_spec": {
                    "window": ["waveform", 1000, -1000, "windowed_waveform"],
                    "dsp_config": {
                        "outputs": ["presum_rate", "presummed_waveform"],
                        "processors": {
                            "presum_rate, presummed_waveform": {
                                "function": "presum",
                                "module": "pygama.dsp.processors",
                                "args": [
                                    "waveform",
                                    0,
                                    "presum_rate",
                                    "presummed_waveform(shape=len(waveform)/4, period=waveform.period*4, offset=waveform.offset)",
                                ],
                                "unit": "ADC",
                            }
                        },
                    },
                    "drop": ["waveform"],
                    "dtype_conv": {
                        "presummed_waveform/values": "uint32",
                        "presum_rate": "uint16",
                    },
                },
            },
        }
    }

    copy_out_spec = {
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

    build_raw(in_stream=daq_file, out_spec=out_spec, overwrite=True)

    # Grab the proc_spec after keylist expansion from build_raw
    proc_spec = {}
    for key in out_spec["FCEventDecoder"].keys():
        if "proc_spec" in out_spec["FCEventDecoder"][key].keys():
            proc_spec[key] = out_spec["FCEventDecoder"][key].pop("proc_spec")

    # build the unprocessed raw file
    build_raw(in_stream=daq_file, out_spec=copy_out_spec, overwrite=True)

    lh5_tables = lgdo.ls(raw_file)

    # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
    for i, tb in enumerate(lh5_tables):
        if "raw" not in tb and lgdo.ls(raw_file, f"{tb}/raw"):
            lh5_tables[i] = f"{tb}/raw"
        elif not lgdo.ls(raw_file, tb):
            del lh5_tables[i]

    jsonfile = proc_spec

    sto = lgdo.LH5Store()

    for raw_group in lh5_tables:

        # First, check the packet ids
        raw_packet_ids, _ = sto.read_object(str(raw_group) + "/packet_id", raw_file)
        processed_packet_ids, _ = sto.read_object(
            str(raw_group) + "/packet_id", processed_file
        )

        assert np.array_equal(raw_packet_ids.nda, processed_packet_ids.nda)

        # Read in the presummed rate from the config file to modify the clock rate later
        group_name = raw_group.split("/raw")[0]
        presum_rate_string = jsonfile[group_name]["dsp_config"]["processors"][
            "presum_rate, presummed_waveform"
        ]["args"][3]
        presum_rate_start_idx = presum_rate_string.find("/") + 1
        presum_rate_end_idx = presum_rate_string.find(",")
        presum_rate = int(presum_rate_string[presum_rate_start_idx:presum_rate_end_idx])

        # This needs to be overwritten with the correct windowing values set in buffer_processor.py
        window_start_index = int(jsonfile[group_name]["window"][1])
        window_end_index = int(jsonfile[group_name]["window"][2])

        raw_packet_waveform_values = sto.read_object(
            str(raw_group) + "/waveform/values", raw_file
        )
        presummed_packet_waveform_values = sto.read_object(
            str(raw_group) + "/presummed_waveform/values", processed_file
        )
        windowed_packet_waveform_values = sto.read_object(
            str(raw_group) + "/windowed_waveform/values", processed_file
        )

        # Check that the lengths of the waveforms match what we expect
        assert len(raw_packet_waveform_values[0].nda[0]) == presum_rate * len(
            presummed_packet_waveform_values[0].nda[0]
        )
        assert isinstance(presummed_packet_waveform_values[0].nda[0][0], np.uint32)
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
            str(raw_group) + "/windowed_waveform/t0", processed_file
        )
        presummed_packet_waveform_t0s, _ = sto.read_object(
            str(raw_group) + "/presummed_waveform/t0", processed_file
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
            str(raw_group) + "/presummed_waveform/dt", processed_file
        )

        # Check that the dts match what we expect, with the correct units
        assert (
            raw_packet_waveform_dts.nda[0]
            == presummed_packet_waveform_dts.nda[0] / presum_rate
        )

        # Check that the presum_rate is correctly identified
        presum_rate_from_file, _ = sto.read_object(
            str(raw_group) + "/presum_rate", processed_file
        )
        assert presum_rate_from_file.nda[0] == presum_rate


def test_proc_geds_no_proc_spms(lgnd_test_data):
    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")
    processed_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-test-pass_proc.lh5"
    )
    raw_file = daq_file.replace(
        "L200-comm-20211130-phy-spms.fcio", "L200-comm-test-pass.lh5"
    )

    out_spec = {
        "FCEventDecoder": {
            "geds": {
                "key_list": [[0, 1]],
                "out_stream": processed_file + ":{name}",
                "out_name": "raw",
                "proc_spec": {
                    "window": ["waveform", 2000, -1000, "windowed_waveform"],
                    "dsp_config": {
                        "outputs": [
                            "presum_rate",
                            "presummed_waveform",
                            "t_sat_lo",
                            "t_sat_hi",
                        ],
                        "processors": {
                            "presum_rate, presummed_waveform": {
                                "function": "presum",
                                "module": "pygama.dsp.processors",
                                "args": [
                                    "waveform",
                                    0,
                                    "presum_rate",
                                    "presummed_waveform(shape=len(waveform)/16, period=waveform.period*16, offset=waveform.offset)",
                                ],
                                "unit": "ADC",
                            },
                            "t_sat_lo, t_sat_hi": {
                                "function": "saturation",
                                "module": "pygama.dsp.processors",
                                "args": ["waveform", 16, "t_sat_lo", "t_sat_hi"],
                                "unit": "ADC",
                            },
                        },
                    },
                    "drop": ["waveform"],
                    "dtype_conv": {
                        "presummed_waveform/values": "uint32",
                        "t_sat_lo": "uint16",
                        "t_sat_hi": "uint16",
                        "presum_rate": "uint16",
                    },
                },
            },
            "spms": {
                "key_list": [[3, 4]],
                "out_stream": processed_file + ":{name}",
                "out_name": "raw",
            },
        }
    }

    copy_out_spec = {
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

    # Do the data processing
    build_raw(in_stream=daq_file, out_spec=out_spec, overwrite=True)

    # Grab the proc_spec after keylist expansion from build_raw
    proc_spec = {}
    for key in out_spec["FCEventDecoder"].keys():
        if "proc_spec" in out_spec["FCEventDecoder"][key].keys():
            proc_spec[key] = out_spec["FCEventDecoder"][key].pop("proc_spec")

    # Do the unprocessed build raw
    build_raw(in_stream=daq_file, out_spec=copy_out_spec, overwrite=True)

    lh5_tables = lgdo.ls(raw_file)
    # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
    for i, tb in enumerate(lh5_tables):
        if "raw" not in tb and lgdo.ls(raw_file, f"{tb}/raw"):
            lh5_tables[i] = f"{tb}/raw"
        elif not lgdo.ls(raw_file, tb):
            del lh5_tables[i]

    jsonfile = proc_spec

    sto = lgdo.LH5Store()

    for raw_group in lh5_tables:

        # First, check the packet ids
        raw_packet_ids, _ = sto.read_object(str(raw_group) + "/packet_id", raw_file)
        processed_packet_ids, _ = sto.read_object(
            str(raw_group) + "/packet_id", processed_file
        )

        assert np.array_equal(raw_packet_ids.nda, processed_packet_ids.nda)

        # Read in the presummed rate from the config file to modify the clock rate later
        group_name = raw_group.split("/raw")[0]
        pass_flag = False
        # If the user passes processing on a group, then the presum_rate is just 1 and there is no windowing
        # If a group_name is absent from the jsonfile, then that means no processing was performed
        if group_name not in jsonfile.keys():
            presum_rate = 1
            window_start_index = 0
            window_end_index = 0
            pass_flag = True
        else:
            presum_rate_string = jsonfile[group_name]["dsp_config"]["processors"][
                "presum_rate, presummed_waveform"
            ]["args"][3]
            presum_rate_start_idx = presum_rate_string.find("/") + 1
            presum_rate_end_idx = presum_rate_string.find(",")
            presum_rate = int(
                presum_rate_string[presum_rate_start_idx:presum_rate_end_idx]
            )

            # This needs to be overwritten with the correct windowing values set in buffer_processor.py
            window_start_index = int(jsonfile[group_name]["window"][1])
            window_end_index = int(jsonfile[group_name]["window"][2])

        # Read in the waveforms
        raw_packet_waveform_values = sto.read_object(
            str(raw_group) + "/waveform/values", raw_file
        )
        if pass_flag:
            presummed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/waveform/values", processed_file
            )
            windowed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/waveform/values", processed_file
            )
        else:
            presummed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/presummed_waveform/values", processed_file
            )
            windowed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/windowed_waveform/values", processed_file
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
                str(raw_group) + "/waveform/t0", processed_file
            )
            presummed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/waveform/t0", processed_file
            )
        else:
            windowed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/windowed_waveform/t0", processed_file
            )
            presummed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/presummed_waveform/t0", processed_file
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
                str(raw_group) + "/waveform/dt", processed_file
            )
        else:

            presummed_packet_waveform_dts, _ = sto.read_object(
                str(raw_group) + "/presummed_waveform/dt", processed_file
            )

            # Check that the presum_rate is correctly identified
            presum_rate_from_file, _ = sto.read_object(
                str(raw_group) + "/presum_rate", processed_file
            )
            assert presum_rate_from_file.nda[0] == presum_rate
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

            proc_sat_lo, _ = sto.read_object(
                str(raw_group) + "/t_sat_lo", processed_file
            )

            proc_sat_hi, _ = sto.read_object(
                str(raw_group) + "/t_sat_hi", processed_file
            )

            assert np.array_equal(raw_sat_lo.nda, proc_sat_lo.nda)
            assert np.array_equal(raw_sat_hi.nda, proc_sat_hi.nda)
            assert type(proc_sat_lo.nda[0]) == np.uint16


# check that packet indexes match in verification test
def test_buffer_processor_multiple_keys(lgnd_test_data):

    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca")
    processed_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca",
        "L200-comm-20220519-phy-geds-key-test_proc.lh5",
    )
    raw_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca", "L200-comm-20220519-phy-geds-key-test.lh5"
    )

    out_spec = {
        "ORFlashCamADCWaveformDecoder": {
            "ch{key}": {
                "key_list": [0, 1],
                "out_stream": processed_file + ":{name}",
                "out_name": "raw",
                "proc_spec": {
                    "window": ["waveform", 2000, -1000, "windowed_waveform"],
                    "dsp_config": {
                        "outputs": [
                            "presum_rate",
                            "presummed_waveform",
                            "t_sat_lo",
                            "t_sat_hi",
                        ],
                        "processors": {
                            "presum_rate, presummed_waveform": {
                                "function": "presum",
                                "module": "pygama.dsp.processors",
                                "args": [
                                    "waveform",
                                    0,
                                    "presum_rate",
                                    "presummed_waveform(shape=len(waveform)/16, period=waveform.period*16, offset=waveform.offset)",
                                ],
                                "unit": "ADC",
                            },
                            "t_sat_lo, t_sat_hi": {
                                "function": "saturation",
                                "module": "pygama.dsp.processors",
                                "args": ["waveform", 16, "t_sat_lo", "t_sat_hi"],
                                "unit": "ADC",
                            },
                        },
                    },
                    "drop": ["waveform"],
                    "dtype_conv": {
                        "presummed_waveform/values": "uint32",
                        "t_sat_lo": "uint16",
                        "t_sat_hi": "uint16",
                        "presum_rate": "uint16",
                    },
                },
            },
            "chan{key}": {
                "key_list": [3, 4],
                "out_stream": processed_file + ":{name}",
                "out_name": "raw",
            },
        }
    }

    copy_out_spec = {
        "ORFlashCamADCWaveformDecoder": {
            "ch{key}": {
                "key_list": [0, 1],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            },
            "chan{key}": {
                "key_list": [3, 4],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            },
        }
    }

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
    # Do the data processing
    build_raw(in_stream=daq_file, out_spec=out_spec, overwrite=True)

    # Grab the proc_spec after keylist expansion from build_raw
    proc_spec = {}
    for key in out_spec["ORFlashCamADCWaveformDecoder"].keys():
        if "proc_spec" in out_spec["ORFlashCamADCWaveformDecoder"][key].keys():
            proc_spec[key] = out_spec["ORFlashCamADCWaveformDecoder"][key].pop(
                "proc_spec"
            )

    # Build the unprocessed raw file for comparison
    build_raw(in_stream=daq_file, out_spec=copy_out_spec, overwrite=True)

    lh5_tables = lgdo.ls(raw_file)
    # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
    for i, tb in enumerate(lh5_tables):
        if "raw" not in tb and lgdo.ls(raw_file, f"{tb}/raw"):
            lh5_tables[i] = f"{tb}/raw"
        elif not lgdo.ls(raw_file, tb):
            del lh5_tables[i]

    jsonfile = proc_spec

    sto = lgdo.LH5Store()

    for raw_group in lh5_tables:

        # First, check the packet ids
        raw_packet_ids, _ = sto.read_object(str(raw_group) + "/packet_id", raw_file)
        processed_packet_ids, _ = sto.read_object(
            str(raw_group) + "/packet_id", processed_file
        )

        assert np.array_equal(raw_packet_ids.nda, processed_packet_ids.nda)

        # Read in the presummed rate from the config file to modify the clock rate later
        group_name = raw_group.split("/raw")[0]

        pass_flag = False
        # If the user passes processing on a group, then the presum_rate is just 1 and there is no windowing
        # If the group_name is absent from the jsonfile, then no processing was done
        if group_name not in jsonfile.keys():
            presum_rate = 1
            window_start_index = 0
            window_end_index = 0
            pass_flag = True
        else:
            presum_rate_string = jsonfile[group_name]["dsp_config"]["processors"][
                "presum_rate, presummed_waveform"
            ]["args"][3]
            presum_rate_start_idx = presum_rate_string.find("/") + 1
            presum_rate_end_idx = presum_rate_string.find(",")
            presum_rate = int(
                presum_rate_string[presum_rate_start_idx:presum_rate_end_idx]
            )

            # This needs to be overwritten with the correct windowing values set in buffer_processor.py
            window_start_index = int(jsonfile[group_name]["window"][1])
            window_end_index = int(jsonfile[group_name]["window"][2])

        # Read in the waveforms
        raw_packet_waveform_values = sto.read_object(
            str(raw_group) + "/waveform/values", raw_file
        )
        if pass_flag:
            presummed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/waveform/values", processed_file
            )
            windowed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/waveform/values", processed_file
            )
        else:
            presummed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/presummed_waveform/values", processed_file
            )
            windowed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/windowed_waveform/values", processed_file
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
        # These are the channels that should be unprocessed
        if group_name == "chan3" or group_name == "chan4":
            raw_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/waveform/values", raw_file
            )
            windowed_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/waveform/values", processed_file
            )
            assert np.array_equal(
                raw_packet_waveform_values.nda, windowed_packet_waveform_values.nda
            )
        else:
            raw_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/waveform/values", raw_file
            )
            windowed_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/windowed_waveform/values", processed_file
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
                str(raw_group) + "/waveform/t0", processed_file
            )
            presummed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/waveform/t0", processed_file
            )
        else:
            windowed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/windowed_waveform/t0", processed_file
            )
            presummed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/presummed_waveform/t0", processed_file
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
                str(raw_group) + "/waveform/dt", processed_file
            )
        else:

            presummed_packet_waveform_dts, _ = sto.read_object(
                str(raw_group) + "/presummed_waveform/dt", processed_file
            )

            # Check that the presum_rate is correctly identified
            presum_rate_from_file, _ = sto.read_object(
                str(raw_group) + "/presum_rate", processed_file
            )
            assert presum_rate_from_file.nda[0] == presum_rate
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

            proc_sat_lo, _ = sto.read_object(
                str(raw_group) + "/t_sat_lo", processed_file
            )

            proc_sat_hi, _ = sto.read_object(
                str(raw_group) + "/t_sat_hi", processed_file
            )

            assert np.array_equal(raw_sat_lo.nda, proc_sat_lo.nda)
            assert np.array_equal(raw_sat_hi.nda, proc_sat_hi.nda)
            assert type(proc_sat_lo.nda[0]) == np.uint16


def test_buffer_processor_all_pass(lgnd_test_data):
    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca")

    raw_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca", "L200-comm-20220519-phy-geds-all-pass.lh5"
    )

    processed_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca",
        "L200-comm-20220519-phy-geds-all-pass_proc.lh5",
    )

    proc_out_spec = {
        "*": {
            "{name}": {"key_list": ["*"], "out_stream": processed_file, "proc_spec": {}}
        }
    }

    raw_out_spec = {
        "*": {
            "{name}": {
                "key_list": ["*"],
                "out_stream": raw_file,
            }
        }
    }

    build_raw(in_stream=daq_file, out_spec=proc_out_spec, overwrite=True)

    build_raw(in_stream=daq_file, out_spec=raw_out_spec, overwrite=True)

    # assert filecmp.cmp(raw_file, processed_file, shallow=True)
    sto = lgdo.LH5Store()
    raw_tables = lgdo.ls(raw_file)
    for tb in raw_tables:
        raw, _ = sto.read_object(tb, raw_file)
        proc, _ = sto.read_object(tb, processed_file)
        if isinstance(raw, lgdo.Scalar):
            raw_value = raw.value
            raw_attrs = raw.attrs
            proc_value = proc.value
            proc_attrs = proc.attrs
            assert raw_value == proc_value
            assert raw_attrs == proc_attrs
        else:
            for obj in raw.keys():
                if not isinstance(raw[obj], lgdo.Table):
                    raw_df = raw.get_dataframe([obj])
                    proc_df = proc.get_dataframe([obj])
                else:
                    for sub_obj in raw[obj].keys():
                        raw_df = raw[obj].get_dataframe([str(sub_obj)])
                        proc_df = proc[obj].get_dataframe([str(sub_obj)])

            assert raw_df.equals(proc_df)


# check that packet indexes match in verification test
def test_buffer_processor_drop_waveform_small_buffer(lgnd_test_data):

    # Set up I/O files, including config
    daq_file = lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca")
    processed_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca",
        "L200-comm-20220519-phy-geds-key-test_proc.lh5",
    )
    raw_file = daq_file.replace(
        "L200-comm-20220519-phy-geds.orca", "L200-comm-20220519-phy-geds-key-test.lh5"
    )

    out_spec = {
        "ORFlashCamADCWaveformDecoder": {
            "ch{key}": {
                "key_list": [0, 1],
                "out_stream": processed_file + ":{name}",
                "out_name": "raw",
                "proc_spec": {
                    "window": ["waveform", 2000, -1000, "windowed_waveform"],
                    "dsp_config": {
                        "outputs": [
                            "presum_rate",
                            "presummed_waveform",
                            "t_sat_lo",
                            "t_sat_hi",
                        ],
                        "processors": {
                            "presum_rate, presummed_waveform": {
                                "function": "presum",
                                "module": "pygama.dsp.processors",
                                "args": [
                                    "waveform",
                                    0,
                                    "presum_rate",
                                    "presummed_waveform(shape=len(waveform)/16, period=waveform.period*16, offset=waveform.offset)",
                                ],
                                "unit": "ADC",
                            },
                            "t_sat_lo, t_sat_hi": {
                                "function": "saturation",
                                "module": "pygama.dsp.processors",
                                "args": ["waveform", 16, "t_sat_lo", "t_sat_hi"],
                                "unit": "ADC",
                            },
                        },
                    },
                    "drop": ["waveform"],
                    "dtype_conv": {
                        "presummed_waveform/values": "uint32",
                        "t_sat_lo": "uint16",
                        "t_sat_hi": "uint16",
                        "presum_rate": "uint16",
                    },
                },
            },
            "chan{key}": {
                "key_list": [3, 4],
                "out_stream": processed_file + ":{name}",
                "out_name": "raw",
            },
        }
    }

    copy_out_spec = {
        "ORFlashCamADCWaveformDecoder": {
            "ch{key}": {
                "key_list": [0, 1],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            },
            "chan{key}": {
                "key_list": [3, 4],
                "out_stream": raw_file + ":{name}",
                "out_name": "raw",
            },
        }
    }

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
    # Do the data processing
    build_raw(in_stream=daq_file, out_spec=out_spec, overwrite=True, buffer_size=2)

    # Grab the proc_spec after keylist expansion from build_raw
    proc_spec = {}
    for key in out_spec["ORFlashCamADCWaveformDecoder"].keys():
        if "proc_spec" in out_spec["ORFlashCamADCWaveformDecoder"][key].keys():
            proc_spec[key] = out_spec["ORFlashCamADCWaveformDecoder"][key].pop(
                "proc_spec"
            )

    # Build the unprocessed raw file for comparison
    build_raw(in_stream=daq_file, out_spec=copy_out_spec, overwrite=True, buffer_size=2)

    lh5_tables = lgdo.ls(raw_file)
    # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
    for i, tb in enumerate(lh5_tables):
        if "raw" not in tb and lgdo.ls(raw_file, f"{tb}/raw"):
            lh5_tables[i] = f"{tb}/raw"
        elif not lgdo.ls(raw_file, tb):
            del lh5_tables[i]

    jsonfile = proc_spec

    sto = lgdo.LH5Store()

    for raw_group in lh5_tables:

        # First, check the packet ids
        raw_packet_ids, _ = sto.read_object(str(raw_group) + "/packet_id", raw_file)
        processed_packet_ids, _ = sto.read_object(
            str(raw_group) + "/packet_id", processed_file
        )

        assert np.array_equal(raw_packet_ids.nda, processed_packet_ids.nda)

        # Read in the presummed rate from the config file to modify the clock rate later
        group_name = raw_group.split("/raw")[0]

        pass_flag = False
        # If the user passes processing on a group, then the presum_rate is just 1 and there is no windowing
        # If the group_name is absent from the jsonfile, then no processing was done
        if group_name not in jsonfile.keys():
            presum_rate = 1
            window_start_index = 0
            window_end_index = 0
            pass_flag = True
        else:
            presum_rate_string = jsonfile[group_name]["dsp_config"]["processors"][
                "presum_rate, presummed_waveform"
            ]["args"][3]
            presum_rate_start_idx = presum_rate_string.find("/") + 1
            presum_rate_end_idx = presum_rate_string.find(",")
            presum_rate = int(
                presum_rate_string[presum_rate_start_idx:presum_rate_end_idx]
            )

            # This needs to be overwritten with the correct windowing values set in buffer_processor.py
            window_start_index = int(jsonfile[group_name]["window"][1])
            window_end_index = int(jsonfile[group_name]["window"][2])

        # Read in the waveforms
        raw_packet_waveform_values = sto.read_object(
            str(raw_group) + "/waveform/values", raw_file
        )
        if pass_flag:
            presummed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/waveform/values", processed_file
            )
            windowed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/waveform/values", processed_file
            )
        else:
            presummed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/presummed_waveform/values", processed_file
            )
            windowed_packet_waveform_values = sto.read_object(
                str(raw_group) + "/windowed_waveform/values", processed_file
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
        # These are the channels that should be unprocessed
        if group_name == "chan3" or group_name == "chan4":
            raw_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/waveform/values", raw_file
            )
            windowed_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/waveform/values", processed_file
            )
            assert np.array_equal(
                raw_packet_waveform_values.nda, windowed_packet_waveform_values.nda
            )
        else:
            raw_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/waveform/values", raw_file
            )
            windowed_packet_waveform_values, _ = sto.read_object(
                str(raw_group) + "/windowed_waveform/values", processed_file
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
                str(raw_group) + "/waveform/t0", processed_file
            )
            presummed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/waveform/t0", processed_file
            )
        else:
            windowed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/windowed_waveform/t0", processed_file
            )
            presummed_packet_waveform_t0s, _ = sto.read_object(
                str(raw_group) + "/presummed_waveform/t0", processed_file
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
                str(raw_group) + "/waveform/dt", processed_file
            )
        else:

            presummed_packet_waveform_dts, _ = sto.read_object(
                str(raw_group) + "/presummed_waveform/dt", processed_file
            )

            # Check that the presum_rate is correctly identified
            presum_rate_from_file, _ = sto.read_object(
                str(raw_group) + "/presum_rate", processed_file
            )
            assert presum_rate_from_file.nda[0] == presum_rate
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

            proc_sat_lo, _ = sto.read_object(
                str(raw_group) + "/t_sat_lo", processed_file
            )

            proc_sat_hi, _ = sto.read_object(
                str(raw_group) + "/t_sat_hi", processed_file
            )

            assert np.array_equal(raw_sat_lo.nda, proc_sat_lo.nda)
            assert np.array_equal(raw_sat_hi.nda, proc_sat_hi.nda)
            assert type(proc_sat_lo.nda[0]) == np.uint16
