import os
from pathlib import Path

import awkward as ak
import lgdo
import pytest
from lgdo.lh5 import LH5Store

from pygama.evt import build_evt
from pygama.skm import build_skm

config_dir = Path(__file__).parent / "configs"
evt_config_dir = Path(__file__).parent.parent / "evt" / "configs"
store = LH5Store()


@pytest.fixture(scope="module")
def files_config(lgnd_test_data, tmptestdir):
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"

    return {
        "tcm": (lgnd_test_data.get_path(tcm_path), "hardware_tcm_1"),
        "dsp": (lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")), "dsp", "ch{}"),
        "hit": (lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")), "hit", "ch{}"),
        "evt": (outfile, "evt"),
    }


def test_basics(tmptestdir, files_config):
    build_evt(
        files_config,
        config=f"{evt_config_dir}/vov-test-evt-config.json",
        wo_mode="of",
    )
    outfile = files_config["evt"][0]

    skm_conf = f"{config_dir}/basic-skm-config.json"
    skm_out = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_skm.lh5"

    result = build_skm(
        outfile,
        files_config["hit"][0],
        files_config["dsp"][0],
        files_config["tcm"][0],
        skm_conf,
    )

    assert isinstance(result, lgdo.Table)

    build_skm(
        outfile,
        files_config["hit"][0],
        files_config["dsp"][0],
        files_config["tcm"][0],
        skm_conf,
        skm_out,
        wo_mode="o",
    )

    assert os.path.exists(skm_out)
    obj, _ = store.read("/skm/", skm_out)

    assert obj == result

    df = obj.view_as("pd")
    assert "timestamp" in df.keys()
    assert "energy_0" in df.keys()
    assert "energy_1" in df.keys()
    assert "energy_2" in df.keys()
    assert "energy_id_0" in df.keys()
    assert "energy_id_1" in df.keys()
    assert "energy_id_2" in df.keys()
    assert "multiplicity" in df.keys()
    assert "energy_sum" in df.keys()
    assert (df.multiplicity.to_numpy() <= 3).all()
    assert (
        df.energy_0.to_numpy() + df.energy_1.to_numpy() + df.energy_2.to_numpy()
        == df.energy_sum.to_numpy()
    ).all()

    vov_eid = ak.to_numpy(
        ak.fill_none(
            ak.pad_none(
                store.read("/evt/energy_id", outfile)[0].view_as("ak"), 3, clip=True
            ),
            0,
        ),
        allow_missing=False,
    )
    assert (vov_eid[:, 0] == df.energy_id_0.to_numpy()).all()
    assert (vov_eid[:, 1] == df.energy_id_1.to_numpy()).all()
    assert (vov_eid[:, 2] == df.energy_id_2.to_numpy()).all()


def test_attribute_passing(tmptestdir, files_config):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)

    build_evt(
        files_config,
        config=f"{evt_config_dir}/vov-test-evt-config.json",
        wo_mode="of",
    )

    skm_conf = f"{config_dir}/basic-skm-config.json"

    skm_out = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_skm.lh5"

    build_skm(
        outfile,
        files_config["hit"][0],
        files_config["dsp"][0],
        files_config["tcm"][0],
        skm_conf,
        f_skm=skm_out,
        wo_mode="o",
    )

    assert os.path.exists(skm_out)
    assert "info" in store.read("/skm/timestamp", skm_out)[0].getattrs().keys()
    assert store.read("/skm/timestamp", skm_out)[0].getattrs()["info"] == "pk was here"
