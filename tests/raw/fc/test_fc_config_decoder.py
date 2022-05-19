from pygama import lgdo


def test_decoding(fcio_config):
    pass


def test_fc_config_is_correct(fcio_config):
    assert isinstance(fcio_config, lgdo.Struct)

    expected_dict = {
        'nsamples': 1836,
        'nadcs': 1,
        'ntriggers': 0,
        'telid': 0,
        'adcbits': 16,
        'sumlength': 1,
        'blprecision': 1,
        'mastercards': 0,
        'triggercards': 0,
        'adccards': 1,
        'gps': 0
    }

    for k, v in expected_dict.items():
        assert fcio_config[k].value == v
