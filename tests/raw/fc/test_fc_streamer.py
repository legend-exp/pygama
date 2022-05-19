from pygama.raw.fc.fc_streamer import FCStreamer
from pygama.raw.raw_buffer import RawBuffer
from legend_testdata import LegendTestData

ldata = LegendTestData()
ldata.checkout('49c7bdc')


def test_open_fc_stream():
    streamer = FCStreamer()
    res = streamer.open_stream(ldata.get_path('fcio/th228.fcio'), buffer_size=1)
    assert isinstance(res[0], RawBuffer)
    assert streamer.fcio is not None # fcio object is instantiated
    assert streamer.packet_id == 0 # packet id is initialized
    assert streamer.n_bytes_read == 11*4 # fc header is read
    assert streamer.event_rbkd is not None # dict containing event info is initialized

    # relevant raw buffers are initialized
    assert 'FCConfigDecoder' in streamer.rb_lib.keys()
    assert 'FCStatusDecoder' in streamer.rb_lib.keys()
    assert 'FCEventDecoder' in streamer.rb_lib.keys()
    assert streamer.rb_lib['FCConfigDecoder'][0].out_name == 'FCConfig'
    assert streamer.rb_lib['FCStatusDecoder'][0].out_name == 'FCStatus'
    assert streamer.rb_lib['FCEventDecoder'][0].out_name == 'FCEvent'


def test_read_packet():
    streamer = FCStreamer()
    streamer.open_stream(ldata.get_path('fcio/th228.fcio'), buffer_size=1)
    assert streamer.read_packet() is True # read was successful
    assert streamer.packet_id == 1 # packet id is incremented
    assert streamer.event_rbkd[0].loc == 1
