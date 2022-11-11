from pygama.raw.compass.compass_streamer import CompassStreamer
from pygama.raw.raw_buffer import RawBuffer, RawBufferList


def test_default_rb_lib(lgnd_test_data):
    streamer = CompassStreamer(
        lgnd_test_data.get_path("compass/compass_test_data_settings.xml"),
    )
    streamer.open_stream(
        lgnd_test_data.get_path("compass/compass_test_data.BIN"), buffer_size=6
    )
    rb_lib = streamer.build_default_rb_lib()
    assert "CompassHeaderDecoder" in rb_lib.keys()
    assert "CompassEventDecoder" in rb_lib.keys()
    assert rb_lib["CompassHeaderDecoder"][0].out_name == "CompassHeader"
    assert rb_lib["CompassEventDecoder"][0].out_name == "CompassEvent"
    assert rb_lib["CompassEventDecoder"][0].key_list == [0, 1]
    streamer.close_stream()


def test_open_stream(lgnd_test_data):
    streamer = CompassStreamer()
    res = streamer.open_stream(
        lgnd_test_data.get_path("compass/compass_test_data.BIN"), buffer_size=6
    )
    assert isinstance(res[0], RawBuffer)
    assert streamer.header is not None  # header object is instantiated
    assert streamer.packet_id == 0  # packet id is initialized
    assert streamer.n_bytes_read == 2  # CoMPASS header is read
    assert streamer.event_rbkd is not None  # dict containing event info is initialized
    streamer.close_stream()


def test_read_packet(lgnd_test_data):
    streamer = CompassStreamer()
    streamer.open_stream(
        lgnd_test_data.get_path("compass/compass_test_data.BIN"), buffer_size=6
    )
    init_rbytes = streamer.n_bytes_read
    assert streamer.read_packet() is True  # read was successful
    assert streamer.packet_id == 1  # packet id is incremented
    assert (
        streamer.n_bytes_read
        == init_rbytes + 25 + 2 * streamer.header["boards"]["0"]["wf_len"].value
    )
    streamer.close_stream()


def test_read_packet_partial(lgnd_test_data):
    streamer = CompassStreamer()
    rb_lib = {"CompassEventDecoder": RawBufferList()}
    rb_lib["CompassEventDecoder"].append(
        RawBuffer(key_list=range(0, 1), out_name="events")
    )

    streamer.open_stream(
        lgnd_test_data.get_path("compass/compass_test_data.BIN"),
        rb_lib=rb_lib,
        buffer_size=6,
    )

    assert list(streamer.rb_lib.keys()) == ["CompassEventDecoder"]
    assert streamer.rb_lib["CompassEventDecoder"][0].key_list == range(0, 1)
    assert streamer.rb_lib["CompassEventDecoder"][0].out_name == "events"

    init_rbytes = streamer.n_bytes_read
    assert streamer.read_packet() is True  # read was successful
    assert streamer.packet_id == 1  # packet id is incremented
    assert (
        streamer.n_bytes_read
        == init_rbytes + 25 + 2 * streamer.header["boards"]["0"]["wf_len"].value
    )
    streamer.close_stream()


def test_read_chunk(lgnd_test_data):
    streamer = CompassStreamer()
    streamer.open_stream(
        lgnd_test_data.get_path("compass/compass_test_data.BIN"), buffer_size=6
    )
    streamer.read_chunk()
    streamer.close_stream()
