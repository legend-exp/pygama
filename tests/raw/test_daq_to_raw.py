from io import BytesIO
import json
import plistlib
import re
import struct

from pygama.lgdo import LH5Store
from pygama.raw import build_raw
from pygama.raw.orca.orca_flashcam import ORFlashCamListenerConfigDecoder, ORFlashCamADCWaveformDecoder
from pygama.raw.orca.orca_run_decoder import ORRunDecoderForRun


class OrcaEncoder(object):
    
    def __init__(self, file):

        self.file = file
        self.header = None

    def encode_header(self):

        LH5 = LH5Store()
        test_file, _ = LH5.read_object(
            'OrcaHeader',
            self.file,
        )
        test_str = str(test_file.value, 'utf-8')
        xml_dict = json.loads(test_str)
        xml_header = BytesIO()
        plistlib.dump(xml_dict, xml_header, fmt=plistlib.FMT_XML)
        xml_header.seek(0)
        compiled_header = str(xml_header.read(), 'utf-8')
        xml_header.close()
        fixed_header = re.sub(
            r"([1-9]\d*\.0)+",
            lambda x: str(int(float(x.group(0)))),
            compiled_header,
        ).replace('1', '1.0', 4)

        self.header = bytes(fixed_header, 'utf-8')
        len_header_true = len(self.header)

        if (len_header_true + 8) % 16 != 0:
            extra_zeros = 16 - ((len_header_true + 8) % 16)
            extra_bytes = struct.pack(
                f'{extra_zeros}b',
                *([0] * extra_zeros),
            )
            self.header += extra_bytes

        len_header_full = len(self.header) + 8

        init_header = struct.pack('i', (len_header_full) // 4)
        init_header += struct.pack('i', len_header_true)

        self.header = init_header + self.header

    def encode_ORFlashCamConfig(self):

        LH5 = LH5Store()
        tbl, _ = LH5.read_object(
            'ORFlashCamListenerConfig', self.file,
        )

        packets = []
        packets.append(4 << 18)
        packets.append((tbl['readout_id'].nda[0] << 16) + tbl['fcid'].nda[0])

        decoded_values = ORFlashCamListenerConfigDecoder().get_decoded_values()

        for i, k in enumerate(decoded_values):
            if i < 2:
                continue
            packets.append(tbl[k].nda[0])
            if k == "gps":
                break

        npacks = tbl['ch_boardid'].nda.shape[-1]

        for ii in range(npacks):
            packets.append((tbl['ch_boardid'].nda[0, ii] << 16) + tbl['ch_inputnum'].nda[0, ii])

        packets[0] += len(packets)

        return packets

    def encode_ORFlashCamADCWaveform(self, ii):

        LH5 = LH5Store()
        tbl, _ = LH5.read_object(
            'ORFlashCamADCWaveform', self.file,
        )

        orca_header_length = 3
        fcio_header_length = 17

        packets = []
        packets.append(3 << 18)
        packets.append(1)
        packets[1] += (orca_header_length << 28)
        packets[1] += (fcio_header_length << 22)

        packet3 = 0x80000
        packet3 += tbl['channel'].nda[ii]
        packet3 += tbl['ch_orca'].nda[ii] << 10
        packet3 += tbl['crate'].nda[ii] << 22
        packet3 += tbl['card'].nda[ii] << 27
        packets.append(packet3)

        # time offsets
        packets.append(tbl['to_mu_sec'].nda[ii])
        packets.append(tbl['to_mu_usec'].nda[ii])
        packets.append(tbl['to_master_sec'].nda[ii])
        packets.append(tbl['to_dt_mu_usec'].nda[ii])
        packets.append(tbl['to_abs_mu_usec'].nda[ii])
        packets.append(tbl['to_start_sec'].nda[ii])
        packets.append(tbl['to_start_usec'].nda[ii])

        # set the dead region values
        packets.append(tbl['dr_start_pps'].nda[ii])
        packets.append(tbl['dr_start_ticks'].nda[ii])
        packets.append(tbl['dr_stop_pps'].nda[ii])
        packets.append(tbl['dr_stop_ticks'].nda[ii])
        packets.append(tbl['dr_maxticks'].nda[ii])

        # set event number and clock counters
        packets.append(tbl['eventnumber'].nda[ii])
        packets.append(tbl['ts_pps'].nda[ii])
        packets.append(tbl['ts_ticks'].nda[ii])
        packets.append(tbl['ts_maxticks'].nda[ii])

        packets.append(tbl['baseline'].nda[ii] + (tbl['daqenergy'].nda[ii] << 16))


        packets.extend([xx + (yy << 16) for xx, yy in zip(
            tbl['waveform']['values'].nda[ii, ::2],
            tbl['waveform']['values'].nda[ii, 1::2],
        )])

        wf_samples = 2 * (len(packets) - orca_header_length - fcio_header_length)

        packets[1] += (wf_samples << 6)
        packets[0] += len(packets)

        return packets

    def encode_ORRunDecoderForRun(self, ii):

        LH5 = LH5Store()
        tbl, _ = LH5.read_object(
            'ORRunDecoderForRun', self.file,
        )

        packets = []
        packets.append(7 << 18)
        packets.append(tbl['subrun_number'].nda[ii] << 16)
        decoded_values = ORRunDecoderForRun().get_decoded_values()
        for i, k in enumerate(decoded_values):
            if 0 < i < 7:
                packets[1] += tbl[k].nda[ii] << (i - 1)

        packets.append(tbl["run_number"].nda[ii])
        packets.append(tbl["time"].nda[ii])

        packets[0] += len(packets)

        return packets


def test_daq_to_raw(lgnd_test_data):

    orca_file = lgnd_test_data.get_path(
        'orca/fc/L200-comm-20220519-phy-geds.orca'
    )
    out_spec = '/tmp/L200-comm-20220519-phy-geds_test.lh5'

    build_raw(
        orca_file,
        in_stream_type='ORCA',
        out_spec=out_spec,
        overwrite=True,
    )

    OE = OrcaEncoder(out_spec)
    OE.encode_header()

    packets_run = []
    for ii in range(2):
        packets_run.append(OE.encode_ORRunDecoderForRun(ii))

    packets_fcconfig = OE.encode_ORFlashCamConfig()

    packets_wf = []
    for ii in range(67):
        packets_wf.append(OE.encode_ORFlashCamADCWaveform(ii))

    rebuilt_orca_data = OE.header 
    rebuilt_orca_data += b''.join(
        struct.pack(f'{len(pp)}I', *pp) for pp in packets_run
    )
    rebuilt_orca_data += struct.pack(
        f'{len(packets_fcconfig)}I', *packets_fcconfig
    )
    rebuilt_orca_data += b''.join(
        struct.pack(f'{len(pp)}I', *pp) for pp in packets_wf
    )

    with open(orca_file, 'rb') as ff:
        orig_orca_data = ff.read()

    assert rebuilt_orca_data == orig_orca_data
