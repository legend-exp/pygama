import numpy as np
import pytest

from pygama import lgdo
from pygama.raw.fc.fc_event_decoder import FCEventDecoder
from pygama.raw.raw_buffer import RawBuffer


@pytest.fixture(scope="module")
def event_rbkd(fcio_obj, fcio_config):
    decoder = FCEventDecoder()
    decoder.set_file_config(fcio_config)

    # get just one record (because size=1, see below) and check if it's a (sparse)event
    assert fcio_obj.get_record() == 3 or fcio_obj.get_record() == 6

    # build raw buffer for each channel in the FC trace list
    rbkd = {}
    for i in fcio_obj.tracelist:
        rbkd[i] = RawBuffer(lgdo=decoder.make_lgdo(size=1))

    # decode packet into the lgdo's and check if the buffer is full
    assert decoder.decode_packet(fcio=fcio_obj, evt_rbkd=rbkd, packet_id=69) is True
    return rbkd


def test_decoding(event_rbkd):
    assert event_rbkd != {}


def test_data_types(event_rbkd):

    for _, v in event_rbkd.items():
        # assert v.out_name == 'FCEvent' FIXME: is this a bug?
        tbl = v.lgdo
        assert isinstance(tbl, lgdo.Struct)
        assert isinstance(tbl["packet_id"], lgdo.Array)
        assert isinstance(tbl["eventnumber"], lgdo.Array)
        assert isinstance(tbl["timestamp"], lgdo.Array)
        assert isinstance(tbl["runtime"], lgdo.Array)
        assert isinstance(tbl["numtraces"], lgdo.Array)
        assert isinstance(tbl["tracelist"], lgdo.VectorOfVectors)
        assert isinstance(tbl["baseline"], lgdo.Array)
        assert isinstance(tbl["daqenergy"], lgdo.Array)
        assert isinstance(tbl["channel"], lgdo.Array)
        assert isinstance(tbl["ts_pps"], lgdo.Array)
        assert isinstance(tbl["ts_ticks"], lgdo.Array)
        assert isinstance(tbl["ts_maxticks"], lgdo.Array)
        assert isinstance(tbl["to_mu_sec"], lgdo.Array)
        assert isinstance(tbl["to_mu_usec"], lgdo.Array)
        assert isinstance(tbl["to_master_sec"], lgdo.Array)
        assert isinstance(tbl["to_dt_mu_usec"], lgdo.Array)
        assert isinstance(tbl["to_abs_mu_usec"], lgdo.Array)
        assert isinstance(tbl["to_start_sec"], lgdo.Array)
        assert isinstance(tbl["to_start_usec"], lgdo.Array)
        assert isinstance(tbl["dr_start_pps"], lgdo.Array)
        assert isinstance(tbl["dr_start_ticks"], lgdo.Array)
        assert isinstance(tbl["dr_stop_pps"], lgdo.Array)
        assert isinstance(tbl["dr_stop_ticks"], lgdo.Array)
        assert isinstance(tbl["dr_maxticks"], lgdo.Array)
        assert isinstance(tbl["deadtime"], lgdo.Array)
        assert isinstance(tbl["waveform"], lgdo.Struct)
        assert isinstance(tbl["waveform"]["t0"], lgdo.Array)
        assert isinstance(tbl["waveform"]["dt"], lgdo.Array)
        assert isinstance(tbl["waveform"]["values"], lgdo.ArrayOfEqualSizedArrays)


def test_values(event_rbkd, fcio_obj):

    fc = fcio_obj
    for ch in fc.tracelist:
        loc = event_rbkd[ch].loc - 1
        tbl = event_rbkd[ch].lgdo

        assert event_rbkd[ch].fill_safety == fc.numtraces

        assert tbl["packet_id"].nda[loc] == 69
        assert tbl["eventnumber"].nda[loc] == fc.eventnumber
        assert tbl["timestamp"].nda[loc] == fc.eventtime
        assert tbl["runtime"].nda[loc] == fc.runtime
        assert tbl["numtraces"].nda[loc] == fc.numtraces

        # custom logic for VectorOfVectors
        start = 0 if loc == 0 else tbl["tracelist"].cumulative_length.nda[loc - 1]
        stop = start + len(fc.tracelist)
        assert np.array_equal(
            tbl["tracelist"].flattened_data.nda[start:stop], fc.tracelist
        )

        assert np.array_equal(tbl["baseline"].nda[loc], fc.baseline[ch])
        assert np.array_equal(tbl["daqenergy"].nda[loc], fc.daqenergy[ch])
        assert tbl["channel"].nda[loc] == ch
        assert tbl["ts_pps"].nda[loc] == fc.timestamp_pps
        assert tbl["ts_ticks"].nda[loc] == fc.timestamp_ticks
        assert tbl["ts_maxticks"].nda[loc] == fc.timestamp_maxticks
        assert tbl["to_mu_sec"].nda[loc] == fc.timeoffset_mu_sec
        assert tbl["to_mu_usec"].nda[loc] == fc.timeoffset_mu_usec
        assert tbl["to_master_sec"].nda[loc] == fc.timeoffset_master_sec
        assert tbl["to_dt_mu_usec"].nda[loc] == fc.timeoffset_dt_mu_usec
        assert tbl["to_abs_mu_usec"].nda[loc] == fc.timeoffset_abs_mu_usec
        assert tbl["to_start_sec"].nda[loc] == fc.timeoffset_start_sec
        assert tbl["to_start_usec"].nda[loc] == fc.timeoffset_start_usec
        assert tbl["dr_start_pps"].nda[loc] == fc.deadregion_start_pps
        assert tbl["dr_start_ticks"].nda[loc] == fc.deadregion_start_ticks
        assert tbl["dr_stop_pps"].nda[loc] == fc.deadregion_stop_pps
        assert tbl["dr_stop_ticks"].nda[loc] == fc.deadregion_stop_ticks
        assert tbl["dr_maxticks"].nda[loc] == fc.deadregion_maxticks
        assert tbl["deadtime"].nda[loc] == fc.deadtime
        assert tbl["waveform"]["t0"].nda[loc] == 0
        assert tbl["waveform"]["dt"].nda[loc] == 16
        assert np.array_equal(tbl["waveform"]["values"].nda[loc], fc.traces[ch])
