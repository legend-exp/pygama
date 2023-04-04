import pytest

from pygama import lgdo
from pygama.raw.compass.compass_event_decoder import CompassEventDecoder
from pygama.raw.raw_buffer import RawBuffer


@pytest.fixture(scope="module")
def event_rbkd(lgnd_test_data, compass_config, packet):
    decoder = CompassEventDecoder()
    decoder.set_header(compass_config)

    # build raw buffer for each channel in the FC trace list
    rbkd = {}
    rbkd[0] = RawBuffer(lgdo=decoder.make_lgdo(size=1))

    # decode packet into the lgdo's and check if the buffer is full
    assert (
        decoder.decode_packet(packet, evt_rbkd=rbkd, packet_id=1, header=compass_config)
        is True
    )
    return rbkd


def test_decoding(event_rbkd):
    assert event_rbkd != {}


def test_data_types(event_rbkd):
    for _, v in event_rbkd.items():
        tbl = v.lgdo
        assert isinstance(tbl, lgdo.Struct)
        assert isinstance(tbl["packet_id"], lgdo.Array)
        assert isinstance(tbl["board"], lgdo.Array)
        assert isinstance(tbl["channel"], lgdo.Array)
        assert isinstance(tbl["timestamp"], lgdo.Array)
        assert isinstance(tbl["energy"], lgdo.Array)
        assert isinstance(tbl["energy_short"], lgdo.Array)
        assert isinstance(tbl["flags"], lgdo.Array)
        assert isinstance(tbl["num_samples"], lgdo.Array)
        assert isinstance(tbl["waveform"], lgdo.Struct)
        assert isinstance(tbl["waveform"]["t0"], lgdo.Array)
        assert isinstance(tbl["waveform"]["dt"], lgdo.Array)
        assert isinstance(tbl["waveform"]["values"], lgdo.ArrayOfEqualSizedArrays)


def test_values(event_rbkd):
    for _, v in event_rbkd.items():
        tbl = v.lgdo
        assert tbl["packet_id"].nda == [1]
        assert tbl["board"].nda == 0
        assert tbl["channel"].nda == 0
        assert tbl["timestamp"].nda == [9.78762e10]
        assert tbl["energy"].nda == [798]
        assert tbl["energy_short"].nda == [135]
        assert tbl["flags"].nda == [16384]
        assert tbl["num_samples"].nda == 1000
        assert (
            tbl["waveform"]["values"].nda[0][:14]
            == [
                2745,
                2742,
                2745,
                2746,
                2745,
                2743,
                2745,
                2744,
                2746,
                2747,
                2746,
                2743,
                2745,
                2747,
            ]
        ).all()
