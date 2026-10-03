from __future__ import annotations

import re
from abc import ABCMeta
from collections.abc import Collection, Mapping
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
from dbetto import AttrsDict

from pygama.datatools.utils import _read_dataflow_config, _tiers_to_dict


class CycleRecordMeta(ABCMeta):
    """Recognize mappings containing the fields required for cycle records."""

    def __instancecheck__(cls, instance):
        try:
            if isinstance(instance, (np.ndarray, np.void)):
                return (
                    instance.shape == ()
                    and instance["relpath"].dtype.type is np.str_
                    and instance["cycle"].dtype.type is np.str_
                )
            if not isinstance(instance, (Mapping, pd.Series, ak.Record)):
                return False
            return isinstance(instance["relpath"], (str, Path)) and isinstance(
                instance["cycle"], str
            )
        except (KeyError, AttributeError, IndexError, ak.errors.FieldNotFoundError):
            return False


class CycleRecord(AttrsDict, metaclass=CycleRecordMeta):
    """`RunRecord` is a read-only wrapper for rows in a run table built by `query_runs`.
    This class will validate the data in the row and provide a configurable
    (via dataflow_config) interface to the data in the row.
    """

    parse_cycle = re.compile(r"(\w+(?:-\w+)*)-tier_(\w+)\.lh5")

    def __init__(self, record):
        """Copy constructor; check if record a ``CycleRecord`` or compliant
        ``ak.Array``, ``pd.Series`` or ``Mapping`` and copy into ``AttrsDict``
        based implementation
        """
        if not isinstance(record, CycleRecord):
            msg = "record must contain 'cycle' and 'relpath' fields"
            raise ValueError(msg)

        if isinstance(record, Mapping):
            super().__init__(record)
        elif isinstance(record, (np.ndarray, np.void)):
            super().__init__({k: record[k].item() for k in record.dtype.names})
        elif isinstance(record, pd.Series):
            super().__init__(record.to_dict())
        elif isinstance(record, ak.Record):
            super().__init__(record.to_list())

    @classmethod
    def get_cycle_record(
        cls,
        cycle_path: str | Path,
        *,
        dataflow_config: Path | str | Mapping = "$REFPROD/dataflow-config.yaml",
        tiers: str | Collection[str] | Mapping[str, str] | None = None,
        cycle_def: Collection[str] | None = None,
        raise_on_missing: bool = False,
    ):
        """
        Build a ``CycleRecord`` from a file path and tier mapping

        Parameters
        ----------
        cycle_path
            path to the cycle file.
        tiers
            mapping from tier name ``tier_[name]`` to base path for tier
        cycle_def
            list of field names for information parsed from cycle name
        raise_on_missing
            if True, raise a ``FileNotFoundError`` if a tier file is missing.
            Otherwise fields will be filled with ``None``.
        """
        cycle_path = Path(cycle_path)
        cycle_file = cycle_path.name

        # split the cycle name from the data tier
        try:
            cycle_name, tier = cls.parse_cycle.fullmatch(cycle_file).groups()
        except AttributeError as e:
            msg = f"invalid file name: {cycle_file}"
            raise ValueError(msg) from e

        if not isinstance(tiers, Mapping) or cycle_def is None:
            dataflow_config, df_paths, query_config = _read_dataflow_config(
                dataflow_config
            )

        tiers = _tiers_to_dict(tiers, df_paths, query_config)

        # get tier basepath using either tier_[name] or just [name]...
        tier_path = tiers.get(f"tier_{tier}", tiers.get(tier))
        if tier_path is None:
            msg = f"tier {tier} not found in tiers"
            raise ValueError(msg)

        cycle_relpath = cycle_path.parent.relative_to(tiers[tier])
        record = {"cycle": cycle_name, "relpath": str(cycle_relpath)}
        cls.update_tiers(record, tiers, raise_on_missing=raise_on_missing)

        if cycle_def is None:
            if "cycle_def" not in query_config:
                msg = "cycle_def must be provided either as kwarg or in dataflow_config"
                raise ValueError(msg)
            cycle_def = query_config["cycle_def"]
        cls.update_cycle_fields(record, cycle_def)

        return CycleRecord(record)

    def update_cycle_fields(
        record,
        cycle_def: str | Collection[str],
    ):
        """Update a record to add fields from cycle name

        Parameters
        ----------
        record
            CycleRecord instance (can also be compliant ``ak.Array``, ``pd.Series`` or ``Mapping``)
        cycle_def
            list of field names for information parsed from cycle name
        """
        if isinstance(cycle_def, str):
            cycle_def = cycle_def.split("-")

        try:
            for f, v in zip(cycle_def, record["cycle"].split("-"), strict=True):
                record[f] = v
        except ValueError as e:
            msg = f"cycle name {record['cycle']} has different number of fields from cycle_def {list(cycle_def)}"
            raise ValueError(msg) from e

    def update_tiers(
        record,
        tiers: Mapping[str, str],
        raise_on_missing: bool = False,
    ):
        """Update a record to add files found for given tiers

        Parameters
        ----------
        record
            CycleRecord instance (can also be compliant ``ak.Array``, ``pd.Series`` or ``Mapping``)
        tiers
            mapping from tier name to base path
        """
        for t in tiers:
            tier_path = CycleRecord.get_tier_filepath(record, t, tiers=tiers)
            if Path(tier_path).exists():
                record[f"tier_{t}"] = tier_path
            elif raise_on_missing:
                msg = f"tier {t} for cycle {record['cycle']}"
                raise FileNotFoundError(msg)
            else:
                record[f"tier_{t}"] = None

    def get_tier_filepath(
        record,
        tier: str,
        *,
        dataflow_config: Path | str | Mapping = "$REFPROD/dataflow-config.yaml",
        tiers: Mapping[str, str] | None = None,
    ):
        """Build the filename for a tier in a given ``CycleRecord``

        Parameters
        ----------
        record
            CycleRecord instance (can also be compliant ``ak.Array``, ``pd.Series`` or ``Mapping``)
        tier
            tier name (e.g. ``tier_[name]``)
        dataflow_config
            path to dataflow config file or dictionary. This will be ignored if tiers is provided
            as a ``Mapping``
        tiers
            mapping from tier name to base path; ignore ``dataflow_config``
        """
        if not isinstance(record, CycleRecord):
            msg = "record must contain 'cycle' and 'relpath' fields"
            raise ValueError(msg)

        if not isinstance(tiers, Mapping):
            _, tiers, _ = _read_dataflow_config(dataflow_config)
        # get tier basepath using either tier_[name] or just [name]...
        tier_path = tiers.get(f"tier_{tier}", tiers.get(tier))
        if tier_path is None:
            msg = f"tier {tier} not found in tiers"
            raise ValueError(msg)

        return f"{tier_path}/{record['relpath']}/{record['cycle']}-tier_{tier}.lh5"
