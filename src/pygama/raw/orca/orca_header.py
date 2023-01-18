from __future__ import annotations

import json
import logging

from pygama import lgdo

log = logging.getLogger(__name__)


class OrcaHeader(dict):
    """ORCA file header object."""

    def __init__(self, jsons: str = None, lgdo_scalar: lgdo.Scalar = None) -> None:
        if jsons is not None:
            self.update(json.loads(jsons))
        elif lgdo_scalar is not None:
            self.set_from_lgdo(lgdo_scalar)

    def set_from_lgdo(self, lgdo_scalar: lgdo.Scalar) -> None:
        if not isinstance(lgdo_scalar, lgdo.Scalar):
            raise ValueError(f"can't instantiate a header from a {type(lgdo_scalar)}")
        else:
            self.update(json.loads(lgdo_scalar.value))

    def get_decoder_list(self) -> list[str]:
        return list(set(self.get_id_to_decoder_name_dict().values()))

    def get_id_to_decoder_name_dict(self, shift_data_id: bool = True) -> dict[int, str]:
        id_dict = {0: "OrcaHeaderDecoder"}
        dd = self["dataDescription"]
        for class_key in dd.keys():
            for super_key in dd[class_key].keys():
                data_id = dd[class_key][super_key]["dataId"]
                if shift_data_id:
                    if data_id < 0:  # short record
                        data_id = (-data_id) >> 26
                    else:
                        data_id = data_id >> 18
                id_dict[data_id] = dd[class_key][super_key]["decoder"]
        return id_dict

    def get_run_number(self) -> int:
        for d in self["ObjectInfo"]["DataChain"]:
            if "Run Control" in d:
                return d["Run Control"]["RunNumber"]
        raise ValueError("No run number found in header!")

    def get_object_info(self, orca_class_name: str) -> dict[int, dict[int, dict]]:
        """Returns a ``dict[crate][card]`` with all info from the header for
        each card with name `orca_class_name`.
        """
        object_info_dict = {}

        crates = self["ObjectInfo"]["Crates"]
        for crate in crates:
            for card in crate["Cards"]:
                if card["Class Name"] == orca_class_name:
                    if crate["CrateNumber"] not in object_info_dict:
                        object_info_dict[crate["CrateNumber"]] = {}
                    object_info_dict[crate["CrateNumber"]][card["Card"]] = card

        return object_info_dict

    def get_readout_info(self, orca_class_name: str, unique_id: int = -1) -> list:
        """Returns a list with all the readout list info from the header with
        name `orca_class_name`.

        Optionally, if `unique_id` is greater or equal than zero, only return
        the readout info for that ORCA unique ID number.
        """
        readout_info_list = []
        try:
            readouts = self["ReadoutDescription"]
            for readout in readouts:
                try:
                    if readout["name"] == orca_class_name:
                        if unique_id >= 0 and readout["uniqueID"] == unique_id:
                            return readout
                        readout_info_list.append(readout)
                except KeyError:
                    continue
        except KeyError:
            pass
        if len(readout_info_list) == 0:
            log.warning(f"no readout info for class name '{orca_class_name}'")
        return readout_info_list

    def get_auxhw_info(self, orca_class_name: str, unique_id: int = -1) -> list:
        """Returns a list with all the info from the ``AuxHw`` table of the header
        with name `orca_class_name`.

        Optionally, if `unique_id` is greater or equal to zero, only return the
        object for that ORCA unique ID number.
        """
        auxhw_info_list = []
        try:
            objs = self["ObjectInfo"]["AuxHw"]
            for obj in objs:
                try:
                    if obj["Class Name"] == orca_class_name:
                        if unique_id >= 0 and obj["uniqueID"] == unique_id:
                            return obj
                        auxhw_info_list.append(obj)
                except KeyError:
                    continue
        except KeyError:
            pass
        if len(auxhw_info_list) == 0:
            log.warning(f"no object info for class name '{orca_class_name}'")
        return auxhw_info_list
