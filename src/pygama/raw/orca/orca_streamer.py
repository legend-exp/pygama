from __future__ import annotations

import gzip
import json
import logging

import numpy as np

from pygama.raw.data_streamer import DataStreamer
from pygama.raw.orca import orca_packet
from pygama.raw.orca.orca_base import OrcaDecoder
from pygama.raw.orca.orca_flashcam import (
    ORFlashCamADCWaveformDecoder,
    ORFlashCamListenerConfigDecoder,
)
from pygama.raw.orca.orca_header_decoder import OrcaHeaderDecoder
from pygama.raw.raw_buffer import RawBuffer, RawBufferLibrary

log = logging.getLogger(__name__)


class OrcaStreamer(DataStreamer):
    """Data streamer for ORCA data."""
    def __init__(self) -> None:
        super().__init__()
        self.in_stream = None
        self.buffer = np.empty(1024, dtype='uint32') # start with a 4 kB packet buffer
        self.header = None
        self.header_decoder = OrcaHeaderDecoder()
        self.decoder_id_dict = {} # dict of data_id to decoder object
        self.decoder_name_dict = {} # dict of name to decoder object
        self.rbl_id_dict = {} # dict of RawBufferLists for each data_id

    # TODO: need to correct for endianness?
    def load_packet(self, skip_unknown_ids: bool = False) -> np.uint32 | None:
        """Loads the next packet into the internal buffer.

        Returns packet as a :class:`numpy.uint32` view of the buffer (a slice),
        returns ``None`` at EOF.
        """
        if self.in_stream is None:
            raise RuntimeError('self.in_stream is None')

        # read packet header
        pkt_hdr = self.buffer[:1]
        n_bytes_read = self.in_stream.readinto(pkt_hdr) # buffer is at least 4 kB long
        self.n_bytes_read += n_bytes_read
        if n_bytes_read == 0: return None
        if n_bytes_read != 4:
            raise RuntimeError(f'only got {n_bytes_read} bytes for packet header')

        # if it's a short packet, we are done
        if orca_packet.is_short(pkt_hdr): return pkt_hdr

        # long packet: get length and check if we can skip it
        n_words = orca_packet.get_n_words(pkt_hdr)
        if skip_unknown_ids and orca_packet.get_data_id(pkt_hdr, shift=False) not in self.decoder_id_dict:
            self.in_stream.seek((n_words-1)*4, 1)
            self.n_bytes_read += (n_words-1)*4 # well, we didn't really read it...
            return pkt_hdr

        # load into buffer, resizing as necessary
        if len(self.buffer) < n_words: self.buffer.resize(n_words, refcheck=False)
        n_bytes_read = self.in_stream.readinto(self.buffer[1:n_words])
        self.n_bytes_read += n_bytes_read
        if n_bytes_read != (n_words-1)*4:
            raise RuntimeError(f'only got {n_bytes_read} bytes for packet read when {(n_words-1)*4} were expected.')

        # return just the packet
        return self.buffer[:n_words]

    def get_decoder_list(self) -> list[OrcaDecoder]:
        return list(self.decoder_id_dict.values())


    def set_in_stream(self, stream_name: str) -> None:
        if self.in_stream is not None: self.close_in_stream()
        if stream_name.endswith('.gz'):
            self.in_stream = gzip.open(stream_name.encode('utf-8'), 'rb')
        else: self.in_stream = open(stream_name.encode('utf-8'), 'rb')
        self.n_bytes_read = 0


    def close_in_stream(self) -> None:
        if self.in_stream is None:
            raise RuntimeError("tried to close an unopened stream")
        self.in_stream.close()
        self.in_stream = None

    def close_stream(self) -> None:
        self.close_in_stream()

    def is_orca_stream(stream_name: str) -> bool: # static function
        orca = OrcaStreamer()
        orca.set_in_stream(stream_name)
        first_bytes = orca.in_stream.read(12)
        orca.close_in_stream()

        # that read should have succeeded
        if len(first_bytes) != 12: return False

        # first 14 bits should be zero
        uints = np.frombuffer(first_bytes, dtype='uint32')
        if (uints[0] & 0xfffc0000) != 0: return False

        # xml header length should fit within header packet length
        pad = uints[0] * 4 - 8 - uints[1]
        if pad < 0 or pad > 3: return False

        # last 4 chars should be '<?xm'
        if first_bytes[8:].decode() != '<?xm': return False

        # it must be an orca stream
        return True

    def hex_dump(self, stream_name: str, n_packets: int = np.inf,
                 skip_header: bool = False, shift_data_id: bool = True,
                 print_n_words: bool = False, max_words: int = np.inf,
                 as_int: bool = False, as_short: bool = False) -> None:
        self.set_in_stream(stream_name)
        if skip_header: self.load_packet()
        while n_packets > 0:
            packet = self.load_packet()
            if packet is None:
                self.close_in_stream()
                return
            orca_packet.hex_dump(packet, shift_data_id=shift_data_id,
                                 print_n_words=print_n_words,
                                 max_words=max_words, as_int=as_int,
                                 as_short=as_short)
            n_packets -= 1

    def open_stream(self,
                    stream_name: str,
                    rb_lib: RawBufferLibrary = None,
                    buffer_size: int = 8192,
                    chunk_mode: str = 'any_full',
                    out_stream: str = '') -> list[RawBuffer]:
        """Initialize the ORCA data stream.

        Parameters
        ----------
        stream_name
            The ORCA filename. Only file streams are currently supported.
            Socket stream reading can be added later.
        rb_lib
            library of buffers for this stream.
        buffer_size
            length of tables to be read out in :meth:`read_chunk`.
        chunk_mode : 'any_full', 'only_full', or 'single_packet'
            sets the mode use for :meth:`read_chunk`.
        out_stream
            optional name of output stream for default `rb_lib` generation.

        Returns
        -------
        header_data
            a list of length 1 containing the raw buffer holding the ORCA header.
        """

        self.set_in_stream(stream_name)

        # read in the header
        packet = self.load_packet()
        if orca_packet.get_data_id(packet) != 0:
            raise RuntimeError(f'got data id {orca_packet.get_data_id(packet)} for header')

        self.packet_id = 0
        self.any_full |= self.header_decoder.decode_packet(packet, self.packet_id)
        self.header = self.header_decoder.header

        # instantiate decoders listed in the header AND in the rb_lib (if specified)
        decoder_names = ['OrcaHeaderDecoder']
        decoder_names += self.header.get_decoder_list()
        if rb_lib is not None and '*' not in rb_lib:
            keep_decoders = []
            for name in decoder_names:
                if name in rb_lib: keep_decoders.append(name)
            decoder_names = keep_decoders
            # check that all requested decoders are present
            for name in rb_lib.keys():
                if name not in keep_decoders:
                    log.warning(f'decoder {name} (requested in rb_lib) not in data description in header')
        for name in decoder_names:
            # handle header decoder specially
            if name == 'OrcaHeaderDecoder':
                self.decoder_id_dict[0] = self.header_decoder
                self.decoder_name_dict['OrcaHeaderDecoder'] = 0
                continue
            # instantiate other decoders by name
            if name not in globals():
                log.warning(f'no implementation of {name}, corresponding packets will be skipped')
                continue
            decoder = globals()[name]
            decoder.data_id = self.header.get_data_id(name)
            self.decoder_id_dict[decoder.data_id] = decoder(header=self.header)
            self.decoder_name_dict[name] = decoder.data_id

        # initialize the buffers in rb_lib. Store them for fast lookup
        super().open_stream(stream_name, rb_lib, buffer_size=buffer_size,
                            chunk_mode=chunk_mode, out_stream=out_stream)
        if rb_lib is None: rb_lib = self.rb_lib
        for name in self.rb_lib.keys():
            data_id = self.decoder_name_dict[name]
            self.rbl_id_dict[data_id] = self.rb_lib[name]
        log.debug(f"rb_lib = {self.rb_lib}")

        # return header raw buffer
        if 'OrcaHeaderDecoder' in rb_lib:
            header_rb_list = rb_lib['OrcaHeaderDecoder']
            if len(header_rb_list) != 1:
                log.warning(f'header_rb_list had length {len(header_rb_list)}, ignoring all but the first')
            rb = header_rb_list[0]
        else: rb = RawBuffer(lgdo=self.header_decoder.make_lgdo())
        rb.lgdo.value = json.dumps(self.header)
        rb.loc = 1 # we have filled this buffer
        return [rb]

    def read_packet(self) -> bool:
        """Read a packet of data.

        Data written to the `rb_lib` attribute.
        """
        # read until we get a decodeable packet
        while True:
            packet = self.load_packet(skip_unknown_ids=True)
            if packet is None: return False
            self.packet_id += 1

            # look up the data id, decoder, and rbl
            data_id = orca_packet.get_data_id(packet, shift=False)
            log.debug(f'packet {self.packet_id}: data_id = {data_id}')
            if data_id in self.decoder_id_dict: break

        # now decode
        decoder = self.decoder_id_dict[data_id]
        rbl = self.rbl_id_dict[data_id]
        self.any_full |= decoder.decode_packet(packet, self.packet_id, rbl)
        return True

        '''
# Import orca_digitizers so that the list of OrcaDecoder.__subclasses__ gets populated
# Do it here so that orca_digitizers can import the functions above here
from . import orca_digitizers, orca_flashcam

def process_orca(daq_filename, raw_file_pattern, n_max=np.inf, ch_groups_dict=None, verbose=False, buffer_size=1024):
    """
    convert ORCA DAQ data to "raw" lh5

    ch_groups_dict: keyed by decoder_name
    """
    lh5_store = lh5.Store()

    f_in = open_orca(daq_filename)
    if f_in == None:
        print("Couldn't find the file %s" % daq_filename)
        sys.exit(0)

    # parse the header. save the length so we can jump past it later
    reclen, header_nbytes, header_dict = parse_header(daq_filename)

    # figure out the total size
    SEEK_END = 2
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)  # rewind
    file_size_MB = file_size / 1e6
    print("Total file size: {:.3f} MB".format(file_size_MB))
    print("Run number:", get_run_number(header_dict))


    # Build the dict used in the inner loop for passing data packets to decoders
    decoders = {}

    # First build a list of all decoder names that might be in the data
    # This is a dict of names keyed off of data_id
    id2dn_dict = get_id_to_decoder_name_dict(header_dict)
    if verbose:
        print("Data IDs present in ORCA file header are:")
        for data_id in id2dn_dict:
            print(f"    {data_id}: {id2dn_dict[data_id]}")

    # Invert the previous list, to get a list of decoder ids keyed off of
    # decoder names
    dn2id_dict = {name:data_id for data_id, name in id2dn_dict.items()}

    # By default we decode all data for which we have decoders. If the user
    # provides a ch_group_dict, we will only decode data from decoders keyed in
    # the dict.
    decode_all_data = True
    decoders_to_run = dn2id_dict.keys()
    if ch_groups_dict is not None:
        decode_all_data = False
        decoders_to_run = ch_groups_dict.keys()

    # Now get the actual requested decoders
    for sub in OrcaDecoder.__subclasses__():
        decoder = sub() # instantiate the class
        if decoder.decoder_name in decoders_to_run:
            decoder.dataID = dn2id_dict[decoder.decoder_name]
            decoder.set_header_dict(header_dict)
            decoders[decoder.dataID] = decoder
    if len(decoders) == 0:
        print("No decoders. Exiting...")
        sys.exit(1)
    if verbose:
        print("pygama will run these decoders:")
        for data_id, dec in decoders.items():
            print("   ", dec.decoder_name+ ", id =", data_id)

    # Now cull the decoders_to_run list
    new_dtr = []
    for decoder_name in decoders_to_run:
        data_id = dn2id_dict[decoder_name]
        if data_id not in decoders.keys():
            print("warning: no decoder exists for", decoder_name, "... will skip its data.")
        else: new_dtr.append(decoder_name)
    decoders_to_run = new_dtr

    # prepare ch groups
    if ch_groups_dict is None:
        ch_groups_dict = {}
        for decoder_name in decoders_to_run:
            ch_groups = create_dummy_ch_groups()
            ch_groups_dict[decoder_name] = ch_groups
            grp_path_template = f'{decoder_name}/raw'
            set_outputs(ch_groups, out_file_template=raw_file_pattern, grp_path_template=grp_path_template)
    else:
        for decoder_name, ch_groups in ch_groups_dict.items():
            expand_ch_groups(ch_groups)
            set_outputs(ch_groups, out_file_template=raw_file_pattern, grp_path_template='{system}/{group_name}/raw')

    # Set up tables for data
    ch_tables_dict = {}
    for data_id, dec in decoders.items():
        decoder_name = id2dn_dict[data_id]
        ch_groups = ch_groups_dict[decoder_name]
        ch_tables_dict[data_id] = build_tables(ch_groups, buffer_size, dec)
    max_tbl_size = 0

    # -- scan over raw data --
    print("Beginning daq-to-raw processing ...")

    packet_id = 0  # number of events decoded
    unrecognized_data_ids = []

    # skip the header using reclen from before
    # reclen is in number of longs, and we want to skip a number of bytes
    f_in.seek(reclen * 4)

    n_entries = 0
    unit = "B"
    if n_max < np.inf and n_max > 0:
        n_entries = n_max
        unit = "id"
    else:
        n_entries = file_size
    progress_bar = tqdm_range(0, int(n_entries), text="Processing", verbose=verbose, unit=unit)
    file_position = 0

    # start scanning
    while (packet_id < n_max and f_in.tell() < file_size):
        packet_id += 1

        try:
            packet, data_id = get_next_packet(f_in)
        except EOFError:
            break
        except Exception as e:
            print("Failed to get the next event ... Exception:", e)
            break

        if decode_all_data and data_id not in decoders:
            if data_id not in unrecognized_data_ids:
                unrecognized_data_ids.append(data_id)
            continue

        if data_id not in decoders: continue
        decoder = decoders[data_id]

        # Clear the tables if the next read could overflow them.
        # Only have to check this when the max table size is within
        # max_n_rows_per_packet of being full.
        if max_tbl_size + decoder.max_n_rows_per_packet() >= buffer_size:
            ch_groups = ch_groups_dict[id2dn_dict[data_id]]
            max_tbl_size = 0
            for group_info in ch_groups.values():
                tbl = group_info['table']
                if tbl.is_full():
                    group_path = group_info['group_path']
                    out_file = group_info['out_file']
                    lh5_store.write_object(tbl, group_path, out_file, n_rows=tbl.loc)
                    tbl.clear()
                if tbl.loc > max_tbl_size: max_tbl_size = tbl.loc
        else: max_tbl_size += decoder.max_n_rows_per_packet()

        tables = ch_tables_dict[data_id]
        decoder.decode_packet(packet, tables, packet_id, header_dict)

        if verbose:
            if n_max < np.inf and n_max > 0:
                update_len = 1
            else:
                update_len = f_in.tell() - file_position
                file_position = f_in.tell()
            update_progress(progress_bar, update_len)


    print("Done. Last packet ID:", packet_id)
    f_in.close()

    # final write to file
    for dec_name, ch_groups in ch_groups_dict.items():
        for group_info in ch_groups.values():
            tbl = group_info['table']
            if tbl.loc == 0: continue
            group_path = group_info['group_path']
            out_file = group_info['out_file']
            lh5_store.write_object(tbl, group_path, out_file, n_rows=tbl.loc)
            print('last write')
            tbl.clear()

    if len(unrecognized_data_ids) > 0:
        print("WARNING, Found the following unknown data IDs:")
        for data_id in unrecognized_data_ids:
            try:
                print("  {}: {}".format(data_id, id2dn_dict[data_id]))
            except KeyError:
                print("  {}: Unknown".format(data_id))
        print("hopefully they weren't important!\n")

    print("Wrote RAW File:\n    {}\nFILE INFO:".format(raw_file_pattern))
        '''
