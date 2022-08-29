def test_decoding(compass_config):
    pass


def test_decoding_no_settings(compass_config_no_settings):
    pass


def test_data_types(compass_config):
    assert isinstance(compass_config, dict)


def test_data_types_no_settings(compass_config_no_settings):
    assert isinstance(compass_config_no_settings, dict)


def test_values(compass_config):
    assert compass_config
    expected_dict = {
        "boards": {
            0: {
                "adcBitCount": "14",
                "channels": {
                    "0": {
                        "SRV_PARAM_CH_BLINE_DCOFFSET": "20.0",
                        "SRV_PARAM_CH_ENABLED": "true",
                        "SRV_PARAM_CH_INDYN": "INDYN_2_0_VPP",
                        "SRV_PARAM_CH_THRESHOLD": "100.0",
                        "SRV_PARAM_CH_TRG_HOLDOFF": "1024.0",
                    },
                    "1": {
                        "SRV_PARAM_CH_ENABLED": "true",
                        "SRV_PARAM_CH_INDYN": "INDYN_0_5_VPP",
                    },
                },
                "modelName": "DT5730",
                "sample_rate": 0.5,
                "wf_len": 1000.0,
            }
        },
        "energy_calibrated": 0,
        "energy_channels": 1,
        "energy_short": 1,
        "waveform_samples": 1,
    }
    for k, v in expected_dict.items():
        assert compass_config[k] == v


def test_values_no_config(compass_config_no_settings):
    expected_dict = {
        "boards": {
            0: {
                "adcBitCount": None,
                "channels": {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}},
                "modelName": None,
                "sample_rate": None,
                "wf_len": 1000.0,
            },
            1: {
                "adcBitCount": None,
                "channels": {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}},
                "modelName": None,
                "sample_rate": None,
                "wf_len": 1000.0,
            },
        },
        "energy_calibrated": 0,
        "energy_channels": 1,
        "energy_short": 1,
        "waveform_samples": 1,
    }

    for k, v in expected_dict.items():
        assert compass_config_no_settings[k] == v
