import numpy as np

from pygama.lgdo import Scalar


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
            "0": {
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
    # We have a nested struct, so we need some nasty recursion
    for key in compass_config.keys():
        if type(compass_config[key]) == Scalar:
            assert compass_config[key].value == expected_dict[key]
        # Nested struct
        else:
            for key_2 in compass_config[key]:
                if type(compass_config[key][key_2]) == Scalar:
                    assert compass_config[key][key_2].value == expected_dict[key][key_2]
                else:
                    for key_3 in compass_config[key][key_2]:
                        if type(compass_config[key][key_2][key_3]) == Scalar:
                            assert (
                                compass_config[key][key_2][key_3].value
                                == expected_dict[key][key_2][key_3]
                            )
                        else:
                            for key_4 in compass_config[key][key_2][key_3]:
                                if (
                                    type(compass_config[key][key_2][key_3][key_4])
                                    == Scalar
                                ):
                                    assert (
                                        compass_config[key][key_2][key_3][key_4].value
                                        == expected_dict[key][key_2][key_3][key_4]
                                    )
                                else:
                                    for key_5 in compass_config[key][key_2][key_3][
                                        key_4
                                    ]:
                                        if (
                                            type(
                                                compass_config[key][key_2][key_3][
                                                    key_4
                                                ][key_5]
                                            )
                                            == Scalar
                                        ):
                                            assert (
                                                compass_config[key][key_2][key_3][
                                                    key_4
                                                ][key_5].value
                                                == expected_dict[key][key_2][key_3][
                                                    key_4
                                                ][key_5]
                                            )
                                        else:
                                            raise AssertionError(
                                                "Got more keys than expected"
                                            )


def test_values_no_config(compass_config_no_settings):
    expected_dict = {
        "boards": {
            "0": {
                "adcBitCount": np.nan,
                "channels": {
                    "0": np.nan,
                    "1": np.nan,
                    "2": np.nan,
                    "3": np.nan,
                    "4": np.nan,
                    "5": np.nan,
                    "6": np.nan,
                    "7": np.nan,
                },
                "modelName": np.nan,
                "sample_rate": np.nan,
                "wf_len": 1000.0,
            },
            "1": {
                "adcBitCount": np.nan,
                "channels": {
                    "0": np.nan,
                    "1": np.nan,
                    "2": np.nan,
                    "3": np.nan,
                    "4": np.nan,
                    "5": np.nan,
                    "6": np.nan,
                    "7": np.nan,
                },
                "modelName": np.nan,
                "sample_rate": np.nan,
                "wf_len": 1000.0,
            },
        },
        "energy_calibrated": 0,
        "energy_channels": 1,
        "energy_short": 1,
        "waveform_samples": 1,
    }
    # We have a nested struct, so we need some nasty recursion
    for key in compass_config_no_settings.keys():
        if type(compass_config_no_settings[key]) == Scalar:
            if (np.isnan(compass_config_no_settings[key].value)) and (
                np.isnan(expected_dict[key])
            ):
                assert True
            else:
                assert compass_config_no_settings[key].value == expected_dict[key]
        else:
            for key_2 in compass_config_no_settings[key]:
                if type(compass_config_no_settings[key][key_2]) == Scalar:
                    if (np.isnan(compass_config_no_settings[key][key_2].value)) and (
                        np.isnan(expected_dict[key][key_2])
                    ):
                        assert True
                    else:
                        assert (
                            compass_config_no_settings[key][key_2].value
                            == expected_dict[key][key_2]
                        )
                else:
                    for key_3 in compass_config_no_settings[key][key_2]:
                        if (
                            type(compass_config_no_settings[key][key_2][key_3])
                            == Scalar
                        ):
                            if (
                                np.isnan(
                                    compass_config_no_settings[key][key_2][key_3].value
                                )
                            ) and (np.isnan(expected_dict[key][key_2][key_3])):
                                assert True
                            else:
                                assert (
                                    compass_config_no_settings[key][key_2][key_3].value
                                    == expected_dict[key][key_2][key_3]
                                )
                        else:
                            for key_4 in compass_config_no_settings[key][key_2][key_3]:
                                if (
                                    type(
                                        compass_config_no_settings[key][key_2][key_3][
                                            key_4
                                        ]
                                    )
                                    == Scalar
                                ):
                                    if np.isnan(
                                        compass_config_no_settings[key][key_2][key_3][
                                            key_4
                                        ].value
                                    ) and (
                                        np.isnan(
                                            expected_dict[key][key_2][key_3][key_4]
                                        )
                                    ):
                                        assert True
                                    else:
                                        assert (
                                            compass_config_no_settings[key][key_2][
                                                key_3
                                            ][key_4].value
                                            == expected_dict[key][key_2][key_3][key_4]
                                        )
                                else:
                                    for key_5 in compass_config_no_settings[key][key_2][
                                        key_3
                                    ][key_4]:
                                        if (
                                            type(
                                                compass_config_no_settings[key][key_2][
                                                    key_3
                                                ][key_4][key_5]
                                            )
                                            == Scalar
                                        ):
                                            if (
                                                np.isnan(
                                                    compass_config_no_settings[key][
                                                        key_2
                                                    ][key_3][key_4][key_5].value
                                                )
                                            ) and (
                                                np.isnan(
                                                    expected_dict[key][key_2][key_3][
                                                        key_4
                                                    ][key_5]
                                                )
                                            ):
                                                assert True
                                            else:
                                                assert (
                                                    compass_config_no_settings[key][
                                                        key_2
                                                    ][key_3][key_4][key_5].value
                                                    == expected_dict[key][key_2][key_3][
                                                        key_4
                                                    ][key_5]
                                                )
                                        else:
                                            raise AssertionError(
                                                "Got more keys than expected"
                                            )
