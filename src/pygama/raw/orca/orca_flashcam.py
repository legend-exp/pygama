import copy
import gc
import logging
from typing import Any

import numpy as np

from pygama.raw.fc.fc_event_decoder import fc_decoded_values
from pygama.raw.orca.orca_base import OrcaDecoder
from pygama.raw.orca.orca_header import OrcaHeader
from pygama.raw.orca.orca_packet import OrcaPacket
from pygama.raw.raw_buffer import RawBufferLibrary

log = logging.getLogger(__name__)


def get_key(fcid, ch: int) -> int:
    return (fcid - 1) * 1000 + ch


def get_fcid(key: int) -> int:
    return int(np.floor(key / 1000)) + 1


def get_ch(key: int) -> int:
    return key % 1000


class ORFlashCamListenerConfigDecoder(OrcaDecoder):
    """Decoder for FlashCam listener config written by ORCA."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:
        # up through ch_inputnum, these are in order of the fcio data format
        # for similicity.  append any additional values after this.
        self.decoded_values = {
            "readout_id": {"dtype": "uint16"},
            "fcid": {"dtype": "uint16"},
            "telid": {"dtype": "int32"},
            "nadcs": {"dtype": "int32"},
            "ntriggers": {"dtype": "int32"},
            "nsamples": {"dtype": "int32"},
            "adcbits": {"dtype": "int32"},
            "sumlength": {"dtype": "int32"},
            "blprecision": {"dtype": "int32"},
            "mastercards": {"dtype": "int32"},
            "triggercards": {"dtype": "int32"},
            "adccards": {"dtype": "int32"},
            "gps": {"dtype": "int32"},
            "ch_boardid": {
                "dtype": "uint16",
                "datatype": "array_of_equalsized_arrays<1,1>{real}",
                "length": 2400,
            },
            "ch_inputnum": {
                "dtype": "uint16",
                "datatype": "array_of_equalsized_arrays<1,1>{real}",
                "length": 2400,
            },
        }
        super().__init__(header=header, **kwargs)

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        return self.decoded_values

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferLibrary
    ) -> bool:
        if len(rbl) != 1:
            log.warning(
                f"got {len(rbl)} rb's, should have only 1 (no keyed decoded values)"
            )
        rb = rbl[0]
        tbl = rb.lgdo
        ii = rb.loc

        int_packet = packet.astype(np.int32)
        tbl["readout_id"].nda[ii] = (int_packet[1] & 0xFFFF0000) >> 16
        tbl["fcid"].nda[ii] = int_packet[1] & 0x0000FFFF

        for i, k in enumerate(self.decoded_values):
            if i < 2:
                continue
            tbl[k].nda[ii] = int_packet[i]
            if k == "gps":
                break

        packet = packet[list(self.decoded_values.keys()).index("ch_boardid") :]
        for i in range(len(packet)):
            tbl["ch_boardid"].nda[ii][i] = (packet[i] & 0xFFFF0000) >> 16
            tbl["ch_inputnum"].nda[ii][i] = packet[i] & 0x0000FFFF

        # check that the ADC decoder has the right number of samples
        objs = []
        for obj in gc.get_objects():
            try:
                if isinstance(obj, ORFlashCamWaveformDecoder):
                    objs.append(obj)
            except ReferenceError:
                # avoids "weakly-referenced object no longer exists"
                pass
        if len(objs) != 1:
            log.warning(f"got {len(objs)} ORFlashCam Waveform Decoders in memory!")
        else:
            objs[0].assert_nsamples(tbl["nsamples"].nda[ii], tbl["fcid"].nda[ii])

        rb.loc += 1
        return rb.is_full()


class ORFlashCamListenerStatusDecoder(OrcaDecoder):
    """
    Decoder for FlashCam status packets written by ORCA

    Some of the card level status data contains an  array of values
    (temperatures for instance) for each card.  Since lh5 currently only
    supports a 1d vector of 1d vectors, this (card,value) data has to be
    flattened before populating the lh5 table.
    """

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:
        raise NotImplementedError(
            "decoding of FlashCam status packets (written by ORCA) is not implemented yet"
        )


class ORFlashCamWaveformDecoder(OrcaDecoder):
    """Decoder for FlashCam ADC data written by ORCA."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:
        # start with the values defined in fcdaq
        self.decoded_values_template = copy.deepcopy(fc_decoded_values)
        # add header values from Orca
        self.decoded_values_template.update(
            {
                "crate": {"dtype": "uint8"},
                "card": {"dtype": "uint8"},
                "ch_orca": {"dtype": "uint8"},
                "fcid": {"dtype": "uint8"},
            }
        )
        self.decoded_values = {}  # dict[fcid]
        self.fcid = {}  # dict[crate][card]
        self.nadc = {}  # dict[fcid]
        super().__init__(header=header, **kwargs)
        self.skipped_channels = {}
        self.ch_orca_mask = 0x00003E00
        self.ch_orca_shift = 9
        self.channel_mask = 0x000001FF

    def set_header(self, header: OrcaHeader) -> None:
        self.header = header

        # set up decoded values, key list, fcid map, etc. based on header info
        fc_listener_info = header.get_readout_info("ORFlashCamListenerModel")
        for info in fc_listener_info:
            fcid = info["uniqueID"]
            # we are going to subtract 1 from fcid for the keys, so it better start from 1
            if fcid == 0:
                raise ValueError("got fcid=0 unexpectedly!")

            obj_info_dict = header.get_object_info("ORFlashCamADCModel")
            obj_info_dict.update(header.get_object_info("ORFlashCamADCStdModel"))
            self.nadc[fcid] = 0
            for child in info["children"]:
                # load self.fcid
                crate = child["crate"]
                if crate not in self.fcid:
                    self.fcid[crate] = {}
                card = child["station"]
                self.fcid[crate][card] = fcid

                if crate not in obj_info_dict:
                    raise RuntimeError(f"no crate {crate} in obj_info_dict")
                if card not in obj_info_dict[crate]:
                    raise RuntimeError(f"no card {card} in obj_info_dict[{crate}]")

                # load self.nadc
                self.nadc[fcid] += np.count_nonzero(
                    obj_info_dict[crate][card]["Enabled"]
                )

            # we are going to shift by 1000 for each fcid, so we better not have that many adcs!
            if self.nadc[fcid] > 1000:
                raise ValueError(f"got too many adc's! ({self.nadc[fcid]})")

            # get the wf len for this fcid and set up decoded_values
            wf_len = header.get_auxhw_info("ORFlashCamListenerModel", fcid)[
                "eventSamples"
            ]
            self.decoded_values[fcid] = copy.deepcopy(self.decoded_values_template)
            self.decoded_values[fcid]["waveform"]["wf_len"] = wf_len
            log.debug(f"fcid {fcid}: {self.nadc[fcid]} adcs, wf_len = {wf_len}")

    def get_key_lists(self) -> list[list[int]]:
        key_lists = []
        for fcid, nadc in self.nadc.items():
            key_lists.append(list(get_key(fcid, np.array(range(nadc)))))
        return key_lists

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        if key is None:
            dec_vals_list = list(self.decoded_values.values())
            if len(dec_vals_list) > 0:
                return dec_vals_list[0]
            raise RuntimeError("decoded_values not built")
        fcid = get_fcid(key)
        if fcid in self.decoded_values:
            return self.decoded_values[fcid]
        raise KeyError(f"no decoded values for key {key} (fcid {fcid})")

    def assert_nsamples(self, nsamples: int, fcid: int) -> None:
        orca_nsamples = self.decoded_values[fcid]["waveform"]["wf_len"]
        if orca_nsamples != nsamples:
            log.warning(
                f"orca miscalculated nsamples = {orca_nsamples} for fcid {fcid}, updating to {nsamples}"
            )
            self.decoded_values[fcid]["waveform"]["wf_len"] = nsamples

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferLibrary
    ) -> bool:
        """Decode the ORCA FlashCam ADC packet."""
        evt_rbkd = rbl.get_keyed_dict()

        # unpack lengths and ids from the header words
        orca_header_length = (packet[1] & 0xF0000000) >> 28
        fcio_header_length = (packet[1] & 0x0FC00000) >> 22
        wf_samples = (packet[1] & 0x003FFFC0) >> 6
        crate = (packet[2] & 0xF8000000) >> 27
        card = (packet[2] & 0x07C00000) >> 22
        fcid = self.fcid[crate][card]
        ch_orca = (packet[2] & self.ch_orca_mask) >> self.ch_orca_shift
        channel = packet[2] & self.channel_mask
        key = get_key(fcid, channel)

        # get the table for this crate/card/channel
        if key not in evt_rbkd:
            if key not in self.skipped_channels:
                self.skipped_channels[key] = 0
            self.skipped_channels[key] += 1
            return False
        tbl = evt_rbkd[key].lgdo
        ii = evt_rbkd[key].loc

        # check that the waveform length is as expected
        rb_wf_len = tbl["waveform"]["values"].nda.shape[1]
        if wf_samples != rb_wf_len:
            if not hasattr(self, "wf_len_errs"):
                self.wf_len_errs = {}
            # if dec_vals has been updated, orca miscalc'd and a warning has
            # already been emitted.  Otherwise, emit a new warning.
            if wf_samples != self.decoded_values[fcid]["waveform"]["wf_len"]:
                if fcid not in self.wf_len_errs:
                    log.warning(
                        f"got waveform from fcid {fcid} of length {wf_samples} with expected length {rb_wf_len}"
                    )
                    self.wf_len_errs[fcid] = True
            # Now resize buffer only if it is still empty.
            # Otherwise emit a warning and keep the smaller length
            if ii != 0:
                if fcid not in self.wf_len_errs:
                    log.warning(
                        "tried to resize buffer according to config record but it was not empty!"
                    )
                    self.wf_len_errs[fcid] = True
                if wf_samples > rb_wf_len:
                    wf_samples = rb_wf_len
            else:
                tbl["waveform"].resize_wf_len(wf_samples)

        # set the values decoded from the header words
        tbl["packet_id"].nda[ii] = packet_id
        tbl["crate"].nda[ii] = crate
        tbl["card"].nda[ii] = card
        tbl["ch_orca"].nda[ii] = ch_orca
        tbl["channel"].nda[ii] = channel
        tbl["fcid"].nda[ii] = fcid
        tbl["numtraces"].nda[ii] = 1

        # set the time offsets
        offset = orca_header_length
        tbl["to_mu_sec"].nda[ii] = np.int32(packet[offset])
        tbl["to_mu_usec"].nda[ii] = np.int32(packet[offset + 1])
        tbl["to_master_sec"].nda[ii] = np.int32(packet[offset + 2])
        tbl["to_dt_mu_usec"].nda[ii] = np.int32(packet[offset + 3])
        tbl["to_abs_mu_usec"].nda[ii] = np.int32(packet[offset + 4])
        tbl["to_start_sec"].nda[ii] = np.int32(packet[offset + 5])
        tbl["to_start_usec"].nda[ii] = np.int32(packet[offset + 6])
        toff = np.float64(packet[offset + 2]) + np.float64(packet[offset + 3]) * 1e-6

        # set the dead region values
        offset += 7
        tbl["dr_start_pps"].nda[ii] = np.int32(packet[offset])
        tbl["dr_start_ticks"].nda[ii] = np.int32(packet[offset + 1])
        tbl["dr_stop_pps"].nda[ii] = np.int32(packet[offset + 2])
        tbl["dr_stop_ticks"].nda[ii] = np.int32(packet[offset + 3])
        tbl["dr_maxticks"].nda[ii] = np.int32(packet[offset + 4])

        # set the event number and clock counters
        offset += 5
        tbl["eventnumber"].nda[ii] = np.int32(packet[offset])
        tbl["ts_pps"].nda[ii] = np.int32(packet[offset + 1])
        tbl["ts_ticks"].nda[ii] = np.int32(packet[offset + 2])
        tbl["ts_maxticks"].nda[ii] = np.int32(packet[offset + 3])

        # set the runtime and timestamp
        tstamp = np.float64(packet[offset + 1])
        tstamp += np.float64(packet[offset + 2]) / (np.int32(packet[offset + 3]) + 1)
        tbl["runtime"].nda[ii] = tstamp
        tbl["timestamp"].nda[ii] = tstamp + toff

        # set the fpga baseline/energy and waveform
        offset = orca_header_length + fcio_header_length
        tbl["baseline"].nda[ii] = packet[offset - 1] & 0x0000FFFF
        tbl["daqenergy"].nda[ii] = (packet[offset - 1] & 0xFFFF0000) >> 16
        wf = np.frombuffer(packet, dtype=np.uint16)[
            offset * 2 : offset * 2 + wf_samples
        ]

        tbl["waveform"]["values"].nda[ii][:wf_samples] = wf

        evt_rbkd[key].loc += 1
        return evt_rbkd[key].is_full()


class ORFlashCamADCWaveformDecoder(ORFlashCamWaveformDecoder):
    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:
        super().__init__(header=header, **kwargs)
        self.ch_orca_mask = 0x00003C00
        self.ch_orca_shift = 10
        self.channel_mask = 0x000003FF
