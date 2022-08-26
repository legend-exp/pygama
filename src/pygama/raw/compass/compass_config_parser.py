from __future__ import annotations

import logging

import xmltodict

log = logging.getLogger(__name__)


max_number_channels = 8
max_number_of_boards = 2  # overload this value if you are using more boards and don't want to supply a config file!
DT5730_sample_freq = 0.5  # sample/ns, because wf_len is stored in ns in the xml file
DT5725_sample_freq = 0.25  # sample/ns


def compass_config_to_dict(compass_config_file: str = None, wf_len: int = None) -> dict:
    """
    Read run-level data from a CoMPASS configuration file.

    Parameters
    ----------
    compass_config_file
        An xml config file for a CoMPASS data run

    Returns
    -------
    config_dict
        A dictionary of configuration data to pass to the streamer

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        {
            "boards" : {
                "0" : {
                    "modelName" : "DT5730",
                    "adcBitCount" : 14,
                    "sample_rate" : 0.5,
                    "wf_len" : 1000,
                    "channels": {
                        "0" : {
                            "Vpp": 2,
                        }
                    }
                }
            }
        }

    Notes
    -----
    First, if the CoMPASS xml config is not none, it is turned into a dictionary.
    If the CoMPASS config file is none, then a generic dictionary with the
    maximum number of channels and boards is returned, in order to create the
    most generic raw buffer to fill.
    """
    # If the config file is not none, read in the xml and convert it to dict
    if compass_config_file is not None:

        with open(compass_config_file) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())

        # Initialize a dictionary to hold the config data
        config_dict = {}
        config_dict["boards"] = {}

        # get the number of boards enabled
        # "configuration/board" is a dictionary if there is only one board
        if isinstance(data_dict["configuration"]["board"], dict):
            config_dict["boards"][0] = {}
            # Read in board and adc
            config_dict["boards"][0]["modelName"] = data_dict["configuration"]["board"][
                "modelName"
            ]
            config_dict["boards"][0]["adcBitCount"] = data_dict["configuration"][
                "board"
            ]["adcBitCount"]

            # Write the sampling rate, in samples/ns and store for conversion of the wf_len
            if config_dict["boards"][0]["modelName"] == "DT5730":
                config_dict["boards"][0]["sample_rate"] = DT5730_sample_freq
            elif config_dict["boards"][0]["modelName"] == "DT5725":
                config_dict["boards"][0]["sample_rate"] = DT5725_sample_freq
            else:
                raise NotImplementedError(
                    f'warning! no sample rate defined yet for board {config_dict["boards"][0]["modelName"]}'
                )
                config_dict["boards"][0]["sample_rate"] = 1

            # The parameters tag is a list of dictionaries, turn it into one dictionary
            param_dict = {}
            for item in data_dict["configuration"]["board"]["parameters"]["entry"]:
                param_name = item["key"]
                param_dict[param_name] = item["value"]

            # Read in wf_len
            if "SRV_PARAM_RECLEN" in param_dict.keys():
                # all xml values are stored with the key "#text" in the config_file
                # the wf_len are stored in ns, but we need the sample length to decode the packets, so convert using the sample rate
                config_dict["boards"][0]["wf_len"] = float(
                    param_dict["SRV_PARAM_RECLEN"]["value"]["#text"]
                ) * float(config_dict["boards"][0]["sample_rate"])
            else:
                raise RuntimeError("wf_len not in config file!")

            # Now, loop through the channels and update the dictionary
            config_dict["boards"][0]["channels"] = {}

            for channel_dictionary in data_dict["configuration"]["board"]["channel"]:

                channel_number = channel_dictionary["index"]
                # first check to make sure that the channel wasn't disabled from the compass interface,
                # the xml file would then have <index>channel_no</index></values>
                if channel_dictionary["values"] is None:
                    continue
                # if channel_dictionary["values"]["entry"] is a dictionary, then that channel is not enabled
                if isinstance(channel_dictionary["values"]["entry"], dict):
                    continue
                # initialize the sub dictionary
                else:
                    config_dict["boards"][0]["channels"][channel_number] = {}

                for x in channel_dictionary["values"]["entry"]:

                    # if a value isn't present, then the key is just set to the default in the param_dict
                    if (x["key"] in param_dict.keys()) and ("value" not in x.keys()):
                        config_dict["boards"][0]["channels"][channel_number][
                            x["key"]
                        ] = param_dict[x["key"]]["value"]["#text"]
                    else:

                        config_dict["boards"][0]["channels"][channel_number][
                            x["key"]
                        ] = x["value"]["#text"]

        # if there is more than one board, loop over each board and get the values
        if isinstance(data_dict["configuration"]["board"], list):
            for i in range(len(data_dict["configuration"]["board"])):
                config_dict["boards"][i] = {}
                # Read in board and adc
                config_dict["boards"][i]["modelName"] = data_dict["configuration"][
                    "board"
                ][i]["modelName"]
                config_dict["boards"][i]["adcBitCount"] = data_dict["configuration"][
                    "board"
                ][i]["adcBitCount"]

                # Write the sampling rate, in samples/ns and store for conversion of the wf_len
                if config_dict["boards"][i]["modelName"] == "DT5730":
                    config_dict["boards"][i]["sample_rate"] = DT5730_sample_freq
                elif config_dict["boards"][i]["modelName"] == "DT5725":
                    config_dict["boards"][i]["sample_rate"] = DT5725_sample_freq
                else:
                    raise NotImplementedError(
                        f'warning! no sample rate defined yet for board {config_dict["boards"][i]["modelName"]}'
                    )
                    config_dict["boards"][i]["sample_rate"] = 1

                # The parameters tag is a list of dictionaries, turn it into one dictionary
                param_dict = {}
                for item in data_dict["configuration"]["board"][i]["parameters"][
                    "entry"
                ]:
                    param_name = item["key"]
                    param_dict[param_name] = item["value"]

                # Read in wf_len
                if "SRV_PARAM_RECLEN" in param_dict.keys():
                    # all xml values are stored with the key "#text" in the config_file
                    # the wf_len are stored in ns, but we need the sample length to decode the packets, so convert using the sample rate
                    config_dict["boards"][i]["wf_len"] = float(
                        param_dict["SRV_PARAM_RECLEN"]["value"]["#text"]
                    ) * float(
                        config_dict["boards"][i]["sample_rate"]
                    )  # all xml values are stored with the key "#text" in the config_file
                else:
                    raise RuntimeError("wf_len not in config file!")

                config_dict["boards"][i]["channels"] = {}

                # Now, loop through the channels and update the dictionary
                for channel_dictionary in data_dict["configuration"]["board"][i][
                    "channel"
                ]:

                    channel_number = channel_dictionary["index"]
                    # if channel_dictionary["values"]["entry"] is a dictionary, then that channel is not enabled
                    if isinstance(channel_dictionary["values"]["entry"], dict):
                        continue
                    # initialize the sub dictionary
                    else:
                        config_dict["boards"][i]["channels"][channel_number] = {}

                    for x in channel_dictionary["values"]["entry"]:

                        # if a value isn't present, then the key is just set to the default in the param_dict
                        if (x["key"] in param_dict.keys()) and (
                            "value" not in x.keys()
                        ):
                            config_dict["boards"][i]["channels"][channel_number][
                                x["key"]
                            ] = param_dict[x["key"]]["value"]["#text"]
                        else:

                            config_dict["boards"][i]["channels"][channel_number][
                                x["key"]
                            ] = x["value"]["#text"]

        return config_dict

    # if the config file is none, return a generic dict with the most channels and boards enabled possible, in order to create a alrge enough raw buffer
    else:
        config_dict = {}
        config_dict["boards"] = {}
        for i in range(max_number_of_boards):
            config_dict["boards"][i] = {}

            # Set board and adc to none
            config_dict["boards"][i]["modelName"] = None
            config_dict["boards"][i]["adcBitCount"] = None
            config_dict["boards"][i]["sample_rate"] = None

            # set the wf_len
            if wf_len is not None:
                config_dict["boards"][i]["wf_len"] = float(wf_len)
            else:
                raise RuntimeError(
                    "wf_len not passed correctly when config file is absent!"
                )

            config_dict["boards"][i]["channels"] = {}
            for j in range(max_number_channels):
                # initialize the sub-dict
                config_dict["boards"][i]["channels"][j] = {}

        return config_dict
