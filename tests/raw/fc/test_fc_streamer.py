from pygama.raw.fc.fc_config_decoder import FCConfigDecoder
from pygama.raw.fc.fc_event_decoder import FCEventDecoder
from pygama.raw.fc.fc_status_decoder import FCStatusDecoder
from pygama.raw.fc.fc_streamer import FCStreamer
from pygama.raw.raw_buffer import RawBuffer, RawBufferList


def test_get_decoder_list():
    streamer = FCStreamer()
    assert isinstance(streamer.get_decoder_list()[0], FCConfigDecoder)
    assert isinstance(streamer.get_decoder_list()[1], FCStatusDecoder)
    assert isinstance(streamer.get_decoder_list()[2], FCEventDecoder)


def test_default_rb_lib(lgnd_test_data):
    streamer = FCStreamer()
    streamer.open_stream(
        lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"), buffer_size=6
    )
    rb_lib = streamer.build_default_rb_lib()
    assert "FCConfigDecoder" in rb_lib.keys()
    assert "FCStatusDecoder" in rb_lib.keys()
    assert "FCEventDecoder" in rb_lib.keys()
    assert rb_lib["FCConfigDecoder"][0].out_name == "FCConfig"
    assert rb_lib["FCStatusDecoder"][0].out_name == "FCStatus"
    assert rb_lib["FCEventDecoder"][0].out_name == "FCEvent"
    assert rb_lib["FCEventDecoder"][0].key_list == range(0, 6)


def test_open_stream(lgnd_test_data):
    streamer = FCStreamer()
    res = streamer.open_stream(
        lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"), buffer_size=6
    )
    assert isinstance(res[0], RawBuffer)
    assert streamer.fcio is not None  # fcio object is instantiated
    assert streamer.packet_id == 0  # packet id is initialized
    assert streamer.n_bytes_read == 11 * 4  # fc header is read
    assert streamer.event_rbkd is not None  # dict containing event info is initialized


def test_read_packet(lgnd_test_data):
    streamer = FCStreamer()
    streamer.open_stream(
        lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"), buffer_size=6
    )
    init_rbytes = streamer.n_bytes_read
    assert streamer.read_packet() is True  # read was successful
    assert streamer.packet_id == 1  # packet id is incremented
    assert streamer.n_bytes_read == init_rbytes + 144 + 2 * streamer.fcio.numtraces * (
        streamer.fcio.nsamples + 3
    )


def test_read_packet_partial(lgnd_test_data):
    streamer = FCStreamer()
    rb_lib = {"FCEventDecoder": RawBufferList()}
    rb_lib["FCEventDecoder"].append(RawBuffer(key_list=range(2, 3), out_name="events"))

    streamer.open_stream(
        lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        rb_lib=rb_lib,
        buffer_size=6,
    )

    assert list(streamer.rb_lib.keys()) == ["FCEventDecoder"]
    assert streamer.rb_lib["FCEventDecoder"][0].key_list == range(2, 3)
    assert streamer.rb_lib["FCEventDecoder"][0].out_name == "events"

    init_rbytes = streamer.n_bytes_read
    assert streamer.read_packet() is True  # read was successful
    assert streamer.packet_id == 1  # packet id is incremented
    assert streamer.n_bytes_read == init_rbytes + 144 + 2 * streamer.fcio.numtraces * (
        streamer.fcio.nsamples + 3
    )


def test_read_chunk(lgnd_test_data):
    streamer = FCStreamer()
    streamer.open_stream(
        lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"), buffer_size=6
    )
    streamer.read_chunk()
