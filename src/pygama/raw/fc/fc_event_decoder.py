from __future__ import annotations

import copy
import logging
from typing import Any

import fcutils

from pygama import lgdo
from pygama.raw.data_decoder import DataDecoder

log = logging.getLogger(__name__)

# put decoded values here where they can be used also by the orca decoder
fc_decoded_values = {
    # packet index in file
    "packet_id": {
        "dtype": "uint32",
    },
    # index of event
    "eventnumber": {
        "dtype": "int32",
    },
    # time since epoch
    "timestamp": {
        "dtype": "float64",
        "units": "s",
    },
    # time since beginning of file
    "runtime": {
        "dtype": "float64",
        "units": "s",
    },
    # number of triggered adc channels
    "numtraces": {
        "dtype": "int32",
    },
    # list of triggered adc channels
    "tracelist": {
        "dtype": "int16",
        "datatype": "array<1>{array<1>{real}}",  # vector of vectors
        "length_guess": 16,
    },
    # fpga baseline
    "baseline": {
        "dtype": "uint16",
    },
    # fpga energy
    "daqenergy": {
        "dtype": "uint16",
    },
    # right now, index of the trigger (trace)
    "channel": {
        "dtype": "uint32",
    },
    # PPS timestamp in sec
    "ts_pps": {
        "dtype": "int32",
    },
    # clock ticks
    "ts_ticks": {
        "dtype": "int32",
    },
    # max clock ticks
    "ts_maxticks": {
        "dtype": "int32",
    },
    # the offset in sec between the master and unix
    "to_mu_sec": {
        "dtype": "int64",
    },
    # the offset in usec between master and unix
    "to_mu_usec": {
        "dtype": "int32",
    },
    # the calculated sec which must be added to the master
    "to_master_sec": {
        "dtype": "int64",
    },
    # the delta time between master and unix in usec
    "to_dt_mu_usec": {
        "dtype": "int32",
    },
    # the abs(time) between master and unix in usec
    "to_abs_mu_usec": {
        "dtype": "int32",
    },
    # startsec
    "to_start_sec": {
        "dtype": "int64",
    },
    # startusec
    "to_start_usec": {
        "dtype": "int32",
    },
    # start pps of the next dead window
    "dr_start_pps": {
        "dtype": "int32",
    },
    # start ticks of the next dead window
    "dr_start_ticks": {
        "dtype": "int32",
    },
    # stop pps of the next dead window
    "dr_stop_pps": {
        "dtype": "int32",
    },
    # stop ticks of the next dead window
    "dr_stop_ticks": {
        "dtype": "int32",
    },
    # maxticks of the dead window
    "dr_maxticks": {
        "dtype": "int32",
    },
    # current dead time calculated from deadregion (dr) fields.
    # Give the total dead time if summed up.
    "deadtime": {
        "dtype": "float64",
    },
    # waveform data
    "waveform": {
        "dtype": "uint16",
        "datatype": "waveform",
        "wf_len": 65532,  # max value. override this before initializing buffers to save RAM
        "dt": 16,  # override if a different clock rate is used
        "dt_units": "ns",
        "t0_units": "ns",
    },
}


class FCEventDecoder(DataDecoder):
    """
    Decode FlashCam digitizer event data.
    """

    def __init__(self, *args, **kwargs) -> None:
        # these are read for every event (decode_event)
        self.decoded_values = copy.deepcopy(fc_decoded_values)
        super().__init__(*args, **kwargs)
        self.skipped_channels = {}
        self.fc_config = None
        self.max_numtraces = 1

    def get_key_lists(self) -> range:
        return [range(self.fc_config["nadcs"].value)]

    def get_decoded_values(self, channel: int = None) -> dict[str, dict[str, Any]]:
        # FC uses the same values for all channels
        return self.decoded_values

    def set_file_config(self, fc_config: lgdo.Struct) -> None:
        """Access ``FCIOConfig`` members once when each file is opened.

        Parameters
        ----------
        fc_config
            extracted via :meth:`~.fc_config_decoder.FCConfigDecoder.decode_config`.
        """
        self.fc_config = fc_config
        self.decoded_values["waveform"]["wf_len"] = self.fc_config["nsamples"].value

    def decode_packet(
        self,
        fcio: fcutils.fcio,
        evt_rbkd: lgdo.Table | dict[int, lgdo.Table],
        packet_id: int,
    ) -> bool:
        """Access ``FCIOEvent`` members for each event in the DAQ file.

        Parameters
        ----------
        fcio
            The interface to the ``fcio`` data. Enters this function after a
            call to ``fcio.get_record()`` so that data for `packet_id` ready to
            be read out.
        evt_rbkd
            A single table for reading out all data, or a dictionary of tables
            keyed by channel number.
        packet_id
            The index of the packet in the `fcio` stream. Incremented by
            :class:`~.raw.fc.fc_streamer.FCStreamer`.

        Returns
        -------
        n_bytes
            (estimated) number of bytes in the packet that was just decoded.
        """
        if fcio.numtraces > self.max_numtraces:
            self.max_numtraces = fcio.numtraces
            # The buffer might be storing all channels' data, so set the
            # fill_safety to the max number of traces we've seen so far.
            for rb in evt_rbkd.values():
                rb.fill_safety = self.max_numtraces
        any_full = False

        # a list of channels is read out simultaneously for each event
        for iwf in fcio.tracelist:
            if iwf not in evt_rbkd:
                if iwf not in self.skipped_channels:
                    # TODO: should this be a warning instead?
                    log.debug(f"skipping packets from channel {iwf}...")
                    self.skipped_channels[iwf] = 0
                self.skipped_channels[iwf] += 1
                continue
            tbl = evt_rbkd[iwf].lgdo
            if fcio.nsamples != tbl["waveform"]["values"].nda.shape[1]:
                log.warning(
                    "event wf length was",
                    fcio.nsamples,
                    "when",
                    self.decoded_values["waveform"]["wf_len"],
                    "were expected",
                )
            ii = evt_rbkd[iwf].loc

            # fill the table
            tbl["channel"].nda[ii] = iwf
            tbl["packet_id"].nda[ii] = packet_id
            tbl["eventnumber"].nda[
                ii
            ] = fcio.eventnumber  # the eventnumber since the beginning of the file
            tbl["timestamp"].nda[ii] = fcio.eventtime  # the time since epoch in seconds
            tbl["runtime"].nda[
                ii
            ] = fcio.runtime  # the time since the beginning of the file in seconds
            tbl["numtraces"].nda[ii] = fcio.numtraces  # number of triggered adcs
            tbl["tracelist"].set_vector(ii, fcio.tracelist)  # list of triggered adcs
            tbl["baseline"].nda[ii] = fcio.baseline[
                iwf
            ]  # the fpga baseline values for each channel in LSB
            tbl["daqenergy"].nda[ii] = fcio.daqenergy[
                iwf
            ]  # the fpga energy values for each channel in LSB
            tbl["ts_pps"].nda[ii] = fcio.timestamp_pps
            tbl["ts_ticks"].nda[ii] = fcio.timestamp_ticks
            tbl["ts_maxticks"].nda[ii] = fcio.timestamp_maxticks
            tbl["to_mu_sec"].nda[ii] = fcio.timeoffset_mu_sec
            tbl["to_mu_usec"].nda[ii] = fcio.timeoffset_mu_usec
            tbl["to_master_sec"].nda[ii] = fcio.timeoffset_master_sec
            tbl["to_dt_mu_usec"].nda[ii] = fcio.timeoffset_dt_mu_usec
            tbl["to_abs_mu_usec"].nda[ii] = fcio.timeoffset_abs_mu_usec
            tbl["to_start_sec"].nda[ii] = fcio.timeoffset_start_sec
            tbl["to_start_usec"].nda[ii] = fcio.timeoffset_start_usec
            tbl["dr_start_pps"].nda[ii] = fcio.deadregion_start_pps
            tbl["dr_start_ticks"].nda[ii] = fcio.deadregion_start_ticks
            tbl["dr_stop_pps"].nda[ii] = fcio.deadregion_stop_pps
            tbl["dr_stop_ticks"].nda[ii] = fcio.deadregion_stop_ticks
            tbl["dr_maxticks"].nda[ii] = fcio.deadregion_maxticks
            tbl["deadtime"].nda[ii] = fcio.deadtime

            # if len(traces[iwf]) != fcio.nsamples: # number of sample per trace check
            tbl["waveform"]["values"].nda[ii][:] = fcio.traces[iwf]

            evt_rbkd[iwf].loc += 1
            any_full |= evt_rbkd[iwf].is_full()

        return any_full
