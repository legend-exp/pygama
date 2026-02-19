import json
from pathlib import Path

import pytest

from pygama.flow import FileDB

config_dir = Path(__file__).parent / "configs"


@pytest.fixture(scope="module")
def test_filedb(lgnd_test_data):
    with open(config_dir / "filedb-config.json") as f:
        config = json.load(f)

    config["data_dir"] = lgnd_test_data.get_path("lh5/prod-ref-l200/generated/tier")
    db = FileDB(config, scan=False)
    db.scan_files(["cal/p03/r001", "phy/p03/r001"])
    return db


@pytest.fixture(scope="module")
def test_filedb_full(lgnd_test_data):
    with open(config_dir / "filedb-config.json") as f:
        config = json.load(f)

    config["data_dir"] = lgnd_test_data.get_path("lh5/prod-ref-l200/generated/tier")
    db = FileDB(config, scan=False)
    db.scan_files(["cal/p03/r001", "phy/p03/r001"])
    db.scan_tables_columns()
    return db
