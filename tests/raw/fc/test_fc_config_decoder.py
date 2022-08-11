from pygama import lgdo


def test_decoding(fcio_config):
    pass


def test_data_types(fcio_config):
    assert isinstance(fcio_config, lgdo.Struct)


def test_values(fcio_config):
    expected_dict = {
        "nsamples": 6000,
        "nadcs": 6,
        "ntriggers": 0,
        "telid": 0,
        "adcbits": 16,
        "sumlength": 1,
        "blprecision": 1,
        "mastercards": 1,
        "triggercards": 0,
        "adccards": 1,
        "gps": 0,
    }

    for k, v in expected_dict.items():
        assert fcio_config[k].value == v
