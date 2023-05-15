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


def get_key(fcid, board_id, fc_input: int) -> int:
    return fcid * 1000000 + board_id * 100 + fc_input


def get_fcid(key: int) -> int:
    return int(np.floor(key / 1000000))


def get_board_id(key: int) -> int:
    return int(np.floor(key / 100)) & 0xFFF


def get_fc_input(key: int) -> int:
    return int(key & 0xFF)


class ORFlashCamListenerConfigDecoder(OrcaDecoder):
    """Decoder for FlashCam listener config written by ORCA."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:
        # up through ch_inputnum, these are in order of the fcio data format
        # for simplicity.  append any additional values after this.
        # note: channel / board level info is vov since there are in principle
        # different boards / channels on each FC system readout by ORCA
        self.decoded_values = {
            "packet_id": {"dtype": "uint32"},
            "packet_len": {"dtype": "uint32"},
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
            "ch_board_id": {
                "dtype": "uint16",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": 2400,  # max number of channels
            },
            "ch_inputnum": {
                "dtype": "uint16",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": 2400,  # max number of channels
            },
            "board_rev": {
                "dtype": "uint8",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "board_uid": {
                "dtype": "uint64",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
        }
        """Default FlashCam (read out by ORCA) config decoded values.

        Warning
        -------
        This configuration can be dynamically modified by the decoder at runtime.
        """
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

        tbl["packet_id"].nda[ii] = packet_id
        tbl["packet_len"].nda[ii] = len(packet)
        tbl["readout_id"].nda[ii] = (packet[1] & 0xFFFF0000) >> 16
        fcid = packet[1] & 0x0000FFFF
        tbl["fcid"].nda[ii] = fcid

        for i, k in enumerate(self.decoded_values):
            if i < 4:
                continue
            tbl[k].nda[ii] = int_packet[i - 2]
            if k == "gps":
                break

        # store gps in the header for building timestamps in the event decoder
        if not hasattr(self.header, "fc_gps"):
            self.header.fc_gps = {}
        self.header.fc_gps[fcid] = tbl["gps"].nda[ii]

        # while decoding this list, store a map of (fcid,fc_channel) to FBI
        # (fcid - board address - input number) in the orca header for use in
        # the waveform packet decoders.
        n_adc = packet[3]
        if not hasattr(self.header, "fbi_map"):
            self.header.fbi_map = {}
        if fcid not in self.header.fbi_map:
            self.header.fbi_map[fcid] = np.empty(n_adc, dtype=np.uint32)

        fd0 = 0 if ii == 0 else tbl["ch_board_id"].cumulative_length.nda[ii - 1]
        tbl["ch_board_id"].cumulative_length.nda[ii] = fd0 + n_adc
        tbl["ch_inputnum"].cumulative_length.nda[ii] = fd0 + n_adc
        for i in range(n_adc):  # i is FC channel
            board_id = (packet[13 + i] & 0xFFFF0000) >> 16
            tbl["ch_board_id"].flattened_data.nda[fd0 + i] = board_id
            fc_input = packet[13 + i] & 0x0000FFFF
            tbl["ch_inputnum"].flattened_data.nda[fd0 + i] = fc_input
            fbi = get_key(fcid, board_id, fc_input)
            self.header.fbi_map[fcid][i] = fbi

        if len(packet) > 13 + n_adc:
            n_boards = packet[11]
            fd0 = 0 if ii == 0 else tbl["board_rev"].cumulative_length.nda[ii - 1]
            tbl["board_rev"].cumulative_length.nda[ii] = fd0 + n_boards
            n_board_rev_words = int(np.ceil(n_boards / 4))
            for i in range(n_board_rev_words):
                for j in range(4):
                    tbl["board_rev"].flattened_data.nda[fd0 + i * 4 + j] = (
                        packet[13 + n_adc + i] >> 4 * j
                    ) & 0xFF

            tbl["board_uid"].cumulative_length.nda[ii] = fd0 + n_boards
            for i in range(n_boards):
                tbl["board_uid"].flattened_data.nda[fd0 + i] = (
                    packet[13 + n_adc + n_board_rev_words + 2 * i] << 32
                )
                tbl["board_uid"].flattened_data.nda[fd0 + i] |= packet[
                    13 + n_adc + n_board_rev_words + 2 * i + 1
                ]

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
        self.decoded_values = {
            "packet_id": {"dtype": "uint32"},
            "readout_id": {"dtype": "uint16"},
            "fcid": {"dtype": "uint16"},
            # 0: Errors occurred, 1: no errors
            "status": {"dtype": "int32"},
            # fc250 seconds, microseconds
            "statustime": {"dtype": "float32", "units": "s"},
            # CPU seconds, microseconds
            "cputime": {"dtype": "float64", "units": "s"},
            # startsec startusec
            "startoffset": {"dtype": "float32", "units": "s"},
            # Total number of cards (number of status data to follow)
            "n_cards": {"dtype": "int32"},
            # Size of each status data
            "card_data_size": {"dtype": "int32"},
            "card_id": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_status": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_eventnumber": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_pps_count": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_clock_tick_count": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_max_clock_ticks": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_n_env_vals": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_n_cti_links": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_n_link_states": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_n_errors_tot": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_n_env_errors": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_n_cti_errors": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_n_link_errors": {
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_n_other_errors": {  # FIXME: should be v of v of array[5]
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(5 * 2400 / 6),  # max number of boards
            },
            "card_temps": {  # FIXME: should be v of v of array[5]
                "dtype": "int32",
                "units": "mC",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(5 * 2400 / 6),  # max number of boards
            },
            "card_voltages": {  # FIXME: should be v of v of array[6]
                "dtype": "int32",
                "units": "mV",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(6 * 2400 / 6),  # max number of boards
            },
            "card_main_current": {
                "dtype": "int32",
                "units": "mA",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_humidity": {
                "dtype": "int32",
                "units": "o/oo",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2400 / 6),  # max number of boards
            },
            "card_adc_temps": {  # FIXME: should be v of v of array[2]
                "dtype": "int32",
                "units": "mC",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(2 * 2400 / 6),  # max number of boards
            },
            "card_cti_links": {  # FIXME: should be v of v of v[n_cti_links]
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(4 * 2400 / 6),  # max number of boards
            },
            "card_link_states": {  # FIXME: should be v of v of v[n_link_states]
                "dtype": "uint32",
                "datatype": "array<1>{array<1>{real}}",  # vector of vectors
                "length_guess": int(256 * 2400 / 6),  # 256*max number of boards
            },
        }
        """Default FlashCam (read out by ORCA) status decoded values.

        Warning
        -------
        This configuration can be dynamically modified by the decoder at runtime.
        """
        super().__init__(header=header, **kwargs)

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        return self.decoded_values

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferLibrary
    ) -> bool:
        """Decode the ORCA FlashCam Status packet."""
        # aliases for brevity
        if len(rbl) != 1:
            log.warning(
                f"got {len(rbl)} rb's, should have only 1 (no keyed decoded values)"
            )
        rb = rbl[0]
        tbl = rb.lgdo
        ii = rb.loc

        tbl["packet_id"].nda[ii] = packet_id
        tbl["readout_id"].nda[ii] = (packet[1] & 0xFFFF0000) >> 16
        fcid = packet[1] & 0x0000FFFF
        tbl["fcid"].nda[ii] = fcid

        int_packet = packet.astype(np.int32)

        # status -- 0: Errors occurred, 1: no errors
        tbl["status"].nda[ii] = int_packet[2]

        # times
        tbl["statustime"].nda[ii] = np.float32(int_packet[3]) + int_packet[4] / 1e6
        tbl["cputime"].nda[ii] = np.float64(int_packet[5]) + int_packet[6] / 1e6
        # packet[7] = empty / dummy
        tbl["startoffset"].nda[ii] = np.float32(int_packet[8]) + int_packet[9] / 1e6

        # note: packet[10-12] are empty

        # Total number of cards (number of status data to follow)
        n_cards = int_packet[13]
        tbl["n_cards"].nda[ii] = n_cards

        # Size of each status data
        cd_size = int_packet[14]
        tbl["card_data_size"].nda[ii] = cd_size

        fd0 = 0 if ii == 0 else tbl["card_id"].cumulative_length.nda[ii - 1]
        tbl["card_id"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_status"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_eventnumber"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_pps_count"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_clock_tick_count"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_max_clock_ticks"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_n_env_vals"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_n_cti_links"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_n_link_states"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_n_errors_tot"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_n_env_errors"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_n_cti_errors"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_n_link_errors"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_n_other_errors"].cumulative_length.nda[ii] = (fd0 + n_cards) * 5
        tbl["card_temps"].cumulative_length.nda[ii] = (fd0 + n_cards) * 5
        tbl["card_voltages"].cumulative_length.nda[ii] = (fd0 + n_cards) * 6
        tbl["card_main_current"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_humidity"].cumulative_length.nda[ii] = fd0 + n_cards
        tbl["card_adc_temps"].cumulative_length.nda[ii] = (fd0 + n_cards) * 2
        # FIXME: can't write multiple rows for some elements and not for others.
        # It breaks the table structure. Need to fix the FIXMEs above: make all
        # of these vector elements vectors of vectors of data. Can't just fix it
        # by implementing get_max_rows_in_packet()
        for i_card in range(n_cards):
            i_cd = 15 + i_card * cd_size  # start index of this card's data
            tbl["card_id"].flattened_data.nda[fd0 + i_card] = packet[i_cd + 0]
            tbl["card_status"].flattened_data.nda[fd0 + i_card] = packet[i_cd + 1]
            tbl["card_eventnumber"].flattened_data.nda[fd0 + i_card] = packet[i_cd + 2]
            tbl["card_pps_count"].flattened_data.nda[fd0 + i_card] = packet[i_cd + 3]
            tbl["card_clock_tick_count"].flattened_data.nda[fd0 + i_card] = packet[
                i_cd + 4
            ]
            tbl["card_max_clock_ticks"].flattened_data.nda[fd0 + i_card] = packet[
                i_cd + 5
            ]
            tbl["card_n_env_vals"].flattened_data.nda[fd0 + i_card] = packet[i_cd + 6]
            tbl["card_n_cti_links"].flattened_data.nda[fd0 + i_card] = packet[i_cd + 7]
            n_link_states = packet[i_cd + 7]
            tbl["card_n_link_states"].flattened_data.nda[fd0 + i_card] = n_link_states
            # packet[i_cd + 9] = empty / dummy
            tbl["card_n_errors_tot"].flattened_data.nda[fd0 + i_card] = packet[
                i_cd + 10
            ]
            tbl["card_n_env_errors"].flattened_data.nda[fd0 + i_card] = packet[
                i_cd + 11
            ]
            tbl["card_n_cti_errors"].flattened_data.nda[fd0 + i_card] = packet[
                i_cd + 12
            ]
            tbl["card_n_link_errors"].flattened_data.nda[fd0 + i_card] = packet[
                i_cd + 13
            ]
            for j in range(5):
                tbl["card_n_other_errors"].flattened_data.nda[
                    (fd0 + i_card) * 5 + j
                ] = packet[i_cd + 14 + j]
            for j in range(5):
                tbl["card_temps"].flattened_data.nda[
                    (fd0 + i_card) * 5 + j
                ] = int_packet[i_cd + 19 + j]
            for j in range(6):
                tbl["card_voltages"].flattened_data.nda[
                    (fd0 + i_card) * 6 + j
                ] = int_packet[i_cd + 24 + j]
            tbl["card_main_current"].flattened_data.nda[fd0 + i_card] = int_packet[
                i_cd + 30
            ]
            tbl["card_humidity"].flattened_data.nda[fd0 + i_card] = int_packet[
                i_cd + 31
            ]
            for j in range(2):
                tbl["card_adc_temps"].flattened_data.nda[
                    (fd0 + i_card) * 2 + j
                ] = int_packet[i_cd + 32 + j]
            # packet[34] is empty / dummy
            for j in range(4):
                tbl["card_cti_links"].flattened_data.nda[
                    (fd0 + i_card) * 4 + j
                ] = packet[i_cd + 35 + j]
            for j in range(256):
                value = packet[i_cd + 39 + j] if j < n_link_states else 0
                tbl["card_link_states"].flattened_data.nda[
                    (fd0 + i_card) * 256 + j
                ] = value

        rb.loc += 1
        return rb.is_full()


class ORFlashCamWaveformDecoder(OrcaDecoder):
    """Decoder for FlashCam ADC data written by ORCA."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:
        # start with the values defined in fcdaq
        self.decoded_values_template = copy.deepcopy(fc_decoded_values)
        """A custom copy of :obj:`.raw.fc.fc_event_decoder.fc_decoded_values`."""
        # add header values from Orca
        self.decoded_values_template.update(
            {
                "event_type": {"dtype": "uint8"},
                "fcid": {"dtype": "uint8"},
                "crate": {"dtype": "uint8"},
                "slot": {"dtype": "uint8"},
                "board_id": {"dtype": "uint16"},
                "fc_input": {"dtype": "uint8"},
            }
        )
        self.decoded_values = {}  # dict[fcid]
        self.fcid = {}  # dict[crate][card]
        self.board_id = {}  # dict[crate][card]
        self.n_adc = {}  # dict[fcid]
        self.key_list = {}  # dict[fcid]
        super().__init__(header=header, **kwargs)
        self.skipped_channels = {}
        self.fc_input_mask = 0x00003E00
        self.fc_input_shift = 9
        self.channel_mask = 0x000001FF

    def set_header(self, header: OrcaHeader) -> None:
        self.header = header

        # set up decoded values, key list, fcid map, etc. based on header info
        fc_listener_info_list = header.get_readout_info("ORFlashCamListenerModel")
        for fc_listener_info in fc_listener_info_list:
            fcid = fc_listener_info["uniqueID"]
            # we are going to subtract 1 from fcid for the keys, so it better start from 1
            if fcid == 0:
                raise ValueError("got fcid=0 unexpectedly!")

            # get FC card object info from header to use below
            # gives access like fc_info[crate][card]
            fc_card_info_dict = header.get_object_info("ORFlashCamADCModel")
            fc_card_info_dict.update(header.get_object_info("ORFlashCamADCStdModel"))

            # build a map of (crate, card) to fcid, and count the number of
            # enabled channels on this fcid. Also build the key list
            self.n_adc[fcid] = 0
            self.key_list[fcid] = []
            for child in fc_listener_info["children"]:
                # load self.fcid
                crate = child["crate"]
                if crate not in self.fcid:
                    self.fcid[crate] = {}
                card = child["station"]
                self.fcid[crate][card] = fcid

                # load self.board_id
                board_id = fc_card_info_dict[crate][card]["CardAddress"]
                if crate not in self.board_id:
                    self.board_id[crate] = {}
                self.board_id[crate][card] = board_id

                if crate not in fc_card_info_dict:
                    raise RuntimeError(f"no crate {crate} in fc_card_info_dict")
                if card not in fc_card_info_dict[crate]:
                    raise RuntimeError(f"no card {card} in fc_card_info_dict[{crate}]")

                # load self.n_adc
                for fc_input in range(len(fc_card_info_dict[crate][card]["Enabled"])):
                    if not fc_card_info_dict[crate][card]["Enabled"][fc_input]:
                        continue
                    self.n_adc[fcid] += 1
                    key = get_key(fcid, board_id, fc_input)
                    if key in self.key_list[fcid]:
                        log.warning(f"key {key} already in key_list...")
                    else:
                        self.key_list[fcid].append(key)

            # FC is only supposed to have 2400 channels max. Double-check it
            if self.n_adc[fcid] > 2400:
                raise ValueError(f"got too many adc's! ({self.n_adc[fcid]})")

            # get the wf len for this fcid and set up decoded_values
            wf_len = header.get_auxhw_info("ORFlashCamListenerModel", fcid)[
                "eventSamples"
            ]
            self.decoded_values[fcid] = copy.deepcopy(self.decoded_values_template)
            self.decoded_values[fcid]["waveform"]["wf_len"] = wf_len
            log.debug(f"fcid {fcid}: {self.n_adc[fcid]} adcs, wf_len = {wf_len}")

    def get_key_lists(self) -> list[list[int]]:
        return list(self.key_list.values())

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
        int_packet = packet.astype(np.int32)

        # unpack lengths and ids from the header words
        orca_header_length = (packet[1] & 0xF0000000) >> 28
        fcio_header_length = (packet[1] & 0x0FC00000) >> 22
        wf_samples = (packet[1] & 0x003FFFC0) >> 6
        event_type = packet[1] & 0x0000001F
        crate = (packet[2] & 0xF8000000) >> 27
        slot = (packet[2] & 0x07C00000) >> 22
        board_id = (packet[2] & 0x003FC000) >> 10
        expected_board_id = self.board_id[crate][slot]
        if board_id != expected_board_id:
            if board_id == (expected_board_id << 4) & 0xFFF:  # just an old ORCA shift
                board_id = expected_board_id
            else:
                if not hasattr(self, "warned_board_id"):
                    self.warned_board_id = {}
                if board_id not in self.warned_board_id:
                    log.warning(
                        f"decoded board_id {board_id} when {expected_board_id} was expected"
                    )
                    self.warned_board_id[board_id] = True

        fc_input = (packet[2] & self.fc_input_mask) >> self.fc_input_shift
        channel = packet[2] & self.channel_mask
        fcid = self.fcid[crate][slot]

        # get the table for this fcid/board_id/fc_input
        key = get_key(fcid, board_id, fc_input)
        # check that the key matches the FC expectation
        if (
            hasattr(self.header, "fbi_map")
            and key != self.header.fbi_map[fcid][channel]
        ):
            # if it doesn't match, try to recover:
            # need to use orca key for looking up the buffer
            # but can correct what gets written to file using the fc_config data
            # this works as long as the the orca <-> fc board ID mapping is 1-to-1
            # So track the matches and emit warnings as they occur
            if not hasattr(self, "key_mismatches"):
                self.key_mismatches = {}
            if key not in self.key_mismatches:
                self.key_mismatches[key] = []
            fc_fbi = self.header.fbi_map[fcid][channel]
            if fc_fbi not in self.key_mismatches[key]:
                log.warning(f"orca key {key} doesn't match FC key {fc_fbi}")
                if len(self.key_mismatches[key]) == 1:
                    log.error(
                        f"orca key {key} corresponds to multiple FC keys! Channel data is mixed"
                    )
                self.key_mismatches[key].append(fc_fbi)
                board_id = get_board_id(fc_fbi)
        if key not in evt_rbkd:
            if key not in self.skipped_channels:
                log.debug(f"skipping key {key}")
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
        tbl["event_type"].nda[ii] = event_type
        tbl["crate"].nda[ii] = crate
        tbl["slot"].nda[ii] = slot
        tbl["fc_input"].nda[ii] = fc_input
        tbl["fcid"].nda[ii] = fcid
        tbl["board_id"].nda[ii] = board_id
        tbl["channel"].nda[ii] = channel
        tbl["numtraces"].nda[ii] = 1

        # set the time offsets
        offset = orca_header_length
        tbl["mu_offset_sec"].nda[ii] = int_packet[offset]
        tbl["mu_offset_usec"].nda[ii] = int_packet[offset + 1]
        tbl["to_master_sec"].nda[ii] = int_packet[offset + 2]
        tbl["delta_mu_usec"].nda[ii] = int_packet[offset + 3]
        tbl["abs_delta_mu_usec"].nda[ii] = int_packet[offset + 4]
        tbl["to_start_sec"].nda[ii] = int_packet[offset + 5]
        tbl["to_start_usec"].nda[ii] = int_packet[offset + 6]

        # set the dead region values
        offset += 7
        tbl["dr_start_pps"].nda[ii] = int_packet[offset]
        tbl["dr_start_ticks"].nda[ii] = int_packet[offset + 1]
        tbl["dr_stop_pps"].nda[ii] = int_packet[offset + 2]
        tbl["dr_stop_ticks"].nda[ii] = int_packet[offset + 3]
        tbl["dr_maxticks"].nda[ii] = int_packet[offset + 4]
        dr_dpps = np.float64(int_packet[offset + 2] - int_packet[offset])
        dr_dticks = np.float64(int_packet[offset + 3] - int_packet[offset + 1])
        tbl["deadtime"].nda[ii] = dr_dpps + dr_dticks / np.float64(
            int_packet[offset + 4]
        )

        # set the event number and clock counters
        offset += 5
        tbl["eventnumber"].nda[ii] = int_packet[offset]
        tbl["ts_pps"].nda[ii] = int_packet[offset + 1]
        tbl["ts_ticks"].nda[ii] = int_packet[offset + 2]
        tbl["ts_maxticks"].nda[ii] = int_packet[offset + 3]

        # set the runtime and timestamp
        tstamp = np.float64(packet[offset + 1])
        tstamp += np.float64(packet[offset + 2]) / (int_packet[offset + 3] + 1)
        tbl["runtime"].nda[ii] = tstamp

        # make a timestamp useful for sorting
        if not hasattr(self.header, "fc_gps"):
            log.warning(
                "didn't decode the FC config record -- timestamps may be miscalculated"
            )
            self.header.fc_gps = {}
        if fcid not in self.header.fc_gps:
            log.warning(f"didn't find fcid {fcid} in fc_gps, adding 0")
            self.header.fc_gps[fcid] = 0
        if self.header.fc_gps[fcid] == 0:
            toff = (
                np.float64(tbl["mu_offset_sec"].nda[ii])
                + np.float64(tbl["mu_offset_usec"].nda[ii]) * 1e-6
            )
        else:
            if tbl["abs_delta_mu_usec"].nda[ii] >= self.header.fc_gps[fcid]:
                log.warning(
                    f"delta {tbl['abs_delta_mu_usec'].nda[ii]} > gps drift allowance {self.header.fc_gps[fcid]}"
                )
            toff = np.float64(tbl["to_master_sec"].nda[ii])
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
        self.fc_input_mask = 0x00003C00
        self.fc_input_shift = 10
        self.channel_mask = 0x000003FF
