import pytest

import pygama.dsp.processors  # noqa: F401
from pygama.lgdo import LH5Store
from pygama.raw.build_raw import build_raw


@pytest.fixture(scope="session")
def geds_raw_tbl(lgnd_test_data):
    store = LH5Store()
    obj, _ = store.read_object(
        "/geds/raw",
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        n_rows=10,
    )
    return obj


@pytest.fixture(scope="session")
def spms_raw_tbl(lgnd_test_data):

    out_file = "/tmp/L200-comm-20211130-phy-spms.lh5"
    out_spec = {
        "FCEventDecoder": {
            "raw": {"key_list": [[0, 6]], "out_stream": f"{out_file}:/spms"}
        }
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        out_spec=out_spec,
        n_max=10,
        overwrite=True,
    )

    store = LH5Store()
    obj, _ = store.read_object("/spms/raw", out_file, n_rows=10)
    return obj
