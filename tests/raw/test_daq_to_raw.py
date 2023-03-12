import json
import plistlib
import re
import struct
from collections import Counter
from io import BytesIO

from pygama.lgdo import LH5Store
from pygama.raw import build_raw
from pygama.raw.orca import orca_streamer
from pygama.raw.orca.orca_flashcam import ORFlashCamListenerConfigDecoder
from pygama.raw.orca.orca_run_decoder import ORRunDecoderForRun


class OrcaEncoder:
    """Encoding function for recreating raw Orca files."""

    def __init__(self, file):
        """Initialize the encoder for daq to raw testing."""

        self.file = file
        self.header = None

    def encode_header(self):
        """Convert orca header back to a byte string."""

        lh5 = LH5Store()
        test_file, _ = lh5.read_object(
            "OrcaHeader",
            self.file,
        )
        test_str = str(test_file.value, "utf-8")
        xml_dict = json.loads(test_str)
        xml_header = BytesIO()
        plistlib.dump(xml_dict, xml_header, fmt=plistlib.FMT_XML)
        xml_header.seek(0)
        compiled_header = str(xml_header.read(), "utf-8")
        xml_header.close()
        fixed_header = re.sub(
            r"([1-9]\d*\.0)+",
            lambda x: str(int(float(x.group(0)))),
            compiled_header,
        ).replace("1", "1.0", 4)

        self.header = bytes(fixed_header, "utf-8")
        len_header_true = len(self.header)

        if (len_header_true + 8) % 16 != 0:
            extra_zeros = 16 - ((len_header_true + 8) % 16)
            extra_bytes = struct.pack(
                f"{extra_zeros}b",
                *([0] * extra_zeros),
            )
            self.header += extra_bytes

        len_header_full = len(self.header) + 8

        init_header = struct.pack("i", (len_header_full) // 4)
        init_header += struct.pack("i", len_header_true)

        self.header = init_header + self.header

    def encode_orflashcamconfig(self, ii):
        """Convert orca flashcam config data back to byte strings."""

        lh5 = LH5Store()
        tbl, _ = lh5.read_object(
            "ORFlashCamListenerConfig",
            self.file,
        )

        packets = []
        packets.append(4 << 18)
        packets.append((tbl["readout_id"].nda[ii] << 16) + tbl["fcid"].nda[ii])

        decoded_values = ORFlashCamListenerConfigDecoder().get_decoded_values()

        for i, k in enumerate(decoded_values):
            if i < 4:
                continue
            packets.append(tbl[k].nda[ii])
            if k == "gps":
                break

        bvi0 = 0  # start index of board vector-of-vector's
        if ii > 0:
            bvi0 = tbl["ch_board_id"].cumulative_length.nda[ii - 1]
        npacks = tbl["ch_board_id"].cumulative_length.nda[ii] - bvi0

        for jj in range(npacks):
            board_id = tbl["ch_board_id"].flattened_data.nda[bvi0 + jj]
            fc_input = tbl["ch_inputnum"].flattened_data.nda[bvi0 + jj]
            packets.append((board_id << 16) + fc_input)

        while len(packets) < tbl["packet_len"].nda[ii]:
            packets.append(0)

        packets[0] += len(packets)

        return packets

    def encode_orflashcamadcwaveform(self, ii):
        """Convert orca flashcam ADC waveform data back to byte strings."""

        lh5 = LH5Store()
        tbl, _ = lh5.read_object(
            "ORFlashCamADCWaveform",
            self.file,
        )

        orca_header_length = 3
        fcio_header_length = 17

        packets = []
        packets.append(3 << 18)
        packets.append(1)
        packets[1] += orca_header_length << 28
        packets[1] += fcio_header_length << 22

        packet3 = 0
        packet3 += tbl["channel"].nda[ii]
        packet3 += tbl["fc_input"].nda[ii] << 10
        packet3 += (tbl["board_id"].nda[ii] & 0xFF) << 14  # old bad board_id encoding
        packet3 += tbl["slot"].nda[ii] << 22
        packet3 += tbl["crate"].nda[ii] << 27
        packets.append(packet3)

        # time offsets
        packets.append(tbl["mu_offset_sec"].nda[ii])
        packets.append(tbl["mu_offset_usec"].nda[ii])
        packets.append(tbl["to_master_sec"].nda[ii])
        packets.append(tbl["delta_mu_usec"].nda[ii])
        packets.append(tbl["abs_delta_mu_usec"].nda[ii])
        packets.append(tbl["to_start_sec"].nda[ii])
        packets.append(tbl["to_start_usec"].nda[ii])

        # set the dead region values
        packets.append(tbl["dr_start_pps"].nda[ii])
        packets.append(tbl["dr_start_ticks"].nda[ii])
        packets.append(tbl["dr_stop_pps"].nda[ii])
        packets.append(tbl["dr_stop_ticks"].nda[ii])
        packets.append(tbl["dr_maxticks"].nda[ii])

        # set event number and clock counters
        packets.append(tbl["eventnumber"].nda[ii])
        packets.append(tbl["ts_pps"].nda[ii])
        packets.append(tbl["ts_ticks"].nda[ii])
        packets.append(tbl["ts_maxticks"].nda[ii])

        packets.append(tbl["baseline"].nda[ii] + (tbl["daqenergy"].nda[ii] << 16))

        packets.extend(
            [
                xx + (yy << 16)
                for xx, yy in zip(
                    tbl["waveform"]["values"].nda[ii, ::2],
                    tbl["waveform"]["values"].nda[ii, 1::2],
                )
            ]
        )

        wf_samples = 2 * (len(packets) - orca_header_length - fcio_header_length)

        packets[1] += wf_samples << 6
        packets[0] += len(packets)

        return packets

    def encode_orrun(self, ii):
        """Convert orca run data back to byte strings."""

        lh5 = LH5Store()
        tbl, _ = lh5.read_object(
            "ORRunDecoderForRun",
            self.file,
        )

        packets = []
        packets.append(7 << 18)
        packets.append(tbl["subrun_number"].nda[ii] << 16)
        decoded_values = ORRunDecoderForRun().get_decoded_values()
        for i, k in enumerate(decoded_values):
            if 0 < i < 7:
                packets[1] += tbl[k].nda[ii] << (i - 1)

        packets.append(tbl["run_number"].nda[ii])
        packets.append(tbl["time"].nda[ii])

        packets[0] += len(packets)

        return packets


def test_daq_to_raw(lgnd_test_data):
    """Test function for the daq to raw validation."""

    # open orca daq file and create LH5 file
    orca_file = lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca")
    out_spec = "/tmp/L200-comm-20220519-phy-geds_test.lh5"

    build_raw(
        orca_file,
        in_stream_type="ORCA",
        out_spec=out_spec,
        overwrite=True,
    )

    # recreate the header
    encoder = OrcaEncoder(out_spec)
    encoder.encode_header()

    rebuilt_orca_data = encoder.header

    # get the order of all of the data IDs
    orstr = orca_streamer.OrcaStreamer()
    orstr.open_stream(orca_file)
    raw_packets = []

    while True:
        this_packet = orstr.load_packet(skip_unknown_ids=False)
        if this_packet is None:
            break
        raw_packets.append(this_packet.copy())

    orstr.close_stream()

    data_ids = [pack[0] >> 18 for pack in raw_packets]

    # count the number of each ID
    full_count = Counter(data_ids)
    buffer_count = {k: 0 for k in full_count}

    # build the raw orca file in the order of data IDs
    for dd in data_ids:
        buffer_count[dd] += 1
        ii = buffer_count[dd] - 1

        if dd == 7:
            this_packet = encoder.encode_orrun(ii)
        elif dd == 4:
            this_packet = encoder.encode_orflashcamconfig(ii)
        elif dd == 3:
            this_packet = encoder.encode_orflashcamadcwaveform(ii)
        else:
            raise ValueError(f"Encoder does not exist for data ID {dd}")

        rebuilt_orca_data += struct.pack(f"{len(this_packet)}I", *this_packet)

    # load the original raw orca file as a byte string
    with open(orca_file, "rb") as ff:
        orig_orca_data = ff.read()

    # assert the byte strings are the same
    assert len(rebuilt_orca_data) == len(orig_orca_data)
    assert rebuilt_orca_data == orig_orca_data
