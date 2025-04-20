from pathlib import Path

import numpy as np
import pytest
from lgdo import lh5

from pygama.evt import utils
from pygama.evt.modules import xtalk

config_dir = Path(__file__).parent / "configs"


@pytest.fixture(scope="module")
def files_config(lgnd_test_data, tmp_dir):
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    outfile = f"{tmp_dir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"

    return {
        "tcm": (lgnd_test_data.get_path(tcm_path), "hardware_tcm_1"),
        "dsp": (lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")), "dsp", "ch{}"),
        "hit": (lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")), "hit", "ch{}"),
        "evt": (outfile, "evt"),
    }


def test_xtalk_corrected_energy(lgnd_test_data, files_config):

    energy = np.array([[1, 2, 3], [4, 5, 6], [2, 0, 1], [0, 1, 0]])
    matrix = np.array([[0, 0, 1], [1, 0, 2], [0, 2, 0]])
    energy_corrected_zero_threshold = xtalk.xtalk_correct_energy_impl(
        energy, energy, matrix, None
    )

    assert np.all(
        energy_corrected_zero_threshold
        == (energy - np.array([[3, 7, 4], [6, 16, 10], [1, 4, 0], [0, 0, 2]]))
    )

    # test with a 2.1 threshold
    energy_corrected_two_threshold = xtalk.xtalk_correct_energy_impl(
        energy, energy, matrix, 2.1
    )
    assert np.all(
        energy_corrected_two_threshold
        == (energy - np.array([[3, 6, 0], [6, 16, 10], [0, 0, 0], [0, 0, 0]]))
    )


def test_gather_energy(lgnd_test_data, files_config):
    f = utils.make_files_config(files_config)
    tcm = utils.TCMData(
        table_key=lh5.read_as(f"/{f.tcm.group}/table_key", f.tcm.file, library="ak"),
        row_in_table=lh5.read_as(
            f"/{f.tcm.group}/row_in_table", f.tcm.file, library="ak"
        ),
    )
    energy = xtalk.gather_energy(
        "hit.cuspEmax_ctc_cal", tcm, f, np.array([1084803, 1084804])
    )
    n_rows = np.max(tcm.row_in_table) + 1
    assert isinstance(energy, np.ndarray)
    assert energy.ndim == 2
    assert np.shape(energy) == (n_rows, 2)


def test_filter_hits(lgnd_test_data, files_config):
    f = utils.make_files_config(files_config)
    tcm = utils.TCMData(
        table_key=lh5.read_as(f"/{f.tcm.group}/table_key", f.tcm.file, library="ak"),
        row_in_table=lh5.read_as(
            f"/{f.tcm.group}/row_in_table", f.tcm.file, library="ak"
        ),
    )
    n_rows = np.max(tcm.row_in_table) + 1

    filter = xtalk.filter_hits(
        f,
        tcm,
        "hit.cuspEmax_ctc_cal>5",
        np.zeros((n_rows, 2)),
        np.array([1084803, 1084804]),
    )

    assert isinstance(filter, np.ndarray)
    assert filter.ndim == 2
    assert np.shape(filter) == (n_rows, 2)
