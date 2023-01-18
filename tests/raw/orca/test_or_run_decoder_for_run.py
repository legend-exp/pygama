import pytest

from pygama import lgdo
from pygama.raw.orca.orca_run_decoder import ORRunDecoderForRun
from pygama.raw.raw_buffer import RawBuffer


@pytest.fixture(scope="module")
def run_rbkd(orca_stream):

    decoder = ORRunDecoderForRun(header=orca_stream.header)

    rbkd = {}
    rbkd[0] = RawBuffer(lgdo=decoder.make_lgdo(size=1))

    this_packet = orca_stream.load_packet(skip_unknown_ids=False)
    orca_stream.close_stream()  # avoid leaving file open

    # assert correct type for ORRunDecoderForRun
    assert (this_packet[0] >> 18) == 7

    # assert that it worked and the buffer is full
    assert decoder.decode_packet(packet=this_packet, packet_id=1, rbl=rbkd)

    return rbkd, this_packet


def test_decoding(run_rbkd):
    assert run_rbkd[0] != {}


def test_data_types(run_rbkd):

    for _, v in run_rbkd[0].items():
        tbl = v.lgdo
        assert isinstance(tbl, lgdo.Struct)
        assert isinstance(tbl["subrun_number"], lgdo.Array)
        assert isinstance(tbl["runstartorstop"], lgdo.Array)
        assert isinstance(tbl["quickstartrun"], lgdo.Array)
        assert isinstance(tbl["remotecontrolrun"], lgdo.Array)
        assert isinstance(tbl["heartbeatrecord"], lgdo.Array)
        assert isinstance(tbl["endsubrunrecord"], lgdo.Array)
        assert isinstance(tbl["startsubrunrecord"], lgdo.Array)
        assert isinstance(tbl["run_number"], lgdo.Array)
        assert isinstance(tbl["time"], lgdo.Array)


def test_values(run_rbkd):

    this_packet = run_rbkd[1]

    decoded_values = ORRunDecoderForRun().decoded_values

    for _, v in run_rbkd[0].items():
        loc = v.loc - 1
        tbl = v.lgdo

        assert tbl["subrun_number"].nda[loc] == (this_packet[1] & 0xFFFF0000) >> 16

        for i, k in enumerate(decoded_values):
            if 0 < i < 7:
                assert tbl[k].nda[loc] == (this_packet[1] & (1 << (i - 1))) >> (i - 1)

        assert tbl["run_number"].nda[loc] == this_packet[2]
        assert tbl["time"].nda[loc] == this_packet[3]
