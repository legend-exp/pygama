from pygama.raw.fc.fc_status_decoder import FCStatusDecoder
from pygama.raw.raw_buffer import RawBuffer
import fcutils
from legend_testdata import LegendTestData

ldata = LegendTestData()
ldata.checkout('49c7bdc')


def test_decode_fc_status():
    decoder = FCStatusDecoder()
    rb = RawBuffer(lgdo=decoder.make_lgdo(size=1))
    fc = fcutils.fcio(ldata.get_path('fcio/th228.fcio'))
    fc.get_record() # TODO: get test file with status record
    decoder.decode_packet(fcio=fc, status_rb=rb, packet_id=0)

    for k in decoder.decoded_values.keys():
        print(k, rb.lgdo[k].nda)
