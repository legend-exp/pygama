import os
from pathlib import Path

from pygama.hit import build_hit

config_dir = Path(__file__).parent / "configs"


def test_build_hit_basic(dsp_test_file):
    build_hit(
        dsp_test_file,
        f"{config_dir}/basic-hit-config.json",
        outfile="/tmp/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5"
    )

    assert os.path.exists("/tmp/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5")
