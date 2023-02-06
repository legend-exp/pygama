import pytest

from pygama.raw.compass.compass_header_decoder import CompassHeaderDecoder
from pygama.raw.compass.compass_streamer import CompassStreamer


@pytest.fixture(scope="module")
def packet(lgnd_test_data):
    streamer = CompassStreamer()
    streamer.open_stream(
        lgnd_test_data.get_path("compass/compass_test_data.BIN"), buffer_size=6
    )
    init_rbytes = streamer.n_bytes_read
    pkt = streamer.load_packet()
    assert pkt is not None  # load was successful
    assert (
        streamer.n_bytes_read
        == init_rbytes + 25 + 2 * streamer.header["boards"]["0"]["wf_len"].value
    )
    streamer.close_stream()
    return pkt


@pytest.fixture(scope="module")
def compass_config(lgnd_test_data):
    decoder = CompassHeaderDecoder()
    test_file = lgnd_test_data.get_path("compass/compass_test_data.BIN")
    in_stream = open(test_file, "rb")
    compass_config_file = lgnd_test_data.get_path(
        "compass/compass_test_data_settings.xml"
    )
    decoder.decode_header(in_stream, compass_config_file)
    in_stream.close()
    return decoder.config


@pytest.fixture(scope="module")
def compass_config_no_settings(lgnd_test_data):
    decoder = CompassHeaderDecoder()
    test_file = lgnd_test_data.get_path("compass/compass_test_data.BIN")
    in_stream = open(test_file, "rb")
    compass_config_file = None
    wf_len = 1000
    decoder.decode_header(in_stream, compass_config_file, wf_len)
    in_stream.close()
    return decoder.config
