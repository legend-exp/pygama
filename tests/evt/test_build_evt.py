import os
from pathlib import Path

import lgdo.lh5_store as store
import numpy as np
import pytest
from lgdo import Array, VectorOfVectors, load_nda, ls

from pygama.evt import build_evt

config_dir = Path(__file__).parent / "configs"


def test_basics(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    build_evt(
        f_tcm=lgnd_test_data.get_path(tcm_path),
        f_dsp=lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")),
        f_hit=lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")),
        f_evt=outfile,
        meta_path=None,
        evt_config=f"{config_dir}/basic-evt-config.json",
        wo_mode="o",
        group="/evt/",
    )

    assert os.path.exists(outfile)
    assert (
        len(ls(outfile, "/evt/")) == 9
    )  # 7 operations of which 2 are requesting channel field
    nda = load_nda(
        outfile, ["energy", "energy_aux", "energy_sum", "multiplicity"], "/evt/"
    )
    assert (
        nda["energy"][nda["multiplicity"] == 1]
        == nda["energy_aux"][nda["multiplicity"] == 1]
    ).all()
    assert (
        nda["energy"][nda["multiplicity"] == 1]
        == nda["energy_sum"][nda["multiplicity"] == 1]
    ).all()
    assert (
        nda["energy_aux"][nda["multiplicity"] == 1]
        == nda["energy_sum"][nda["multiplicity"] == 1]
    ).all()


def test_lar_module(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    build_evt(
        f_tcm=lgnd_test_data.get_path(tcm_path),
        f_dsp=lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")),
        f_hit=lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")),
        f_evt=outfile,
        meta_path=None,
        evt_config=f"{config_dir}/module-test-evt-config.json",
        wo_mode="o",
        group="/evt/",
    )

    assert os.path.exists(outfile)
    assert len(ls(outfile, "/evt/")) == 7
    assert (
        np.max(load_nda(outfile, ["lar_multiplicity"], "/evt/")["lar_multiplicity"])
        <= 3
    )


def test_vov(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    build_evt(
        f_tcm=lgnd_test_data.get_path(tcm_path),
        f_dsp=lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")),
        f_hit=lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")),
        f_evt=outfile,
        meta_path=None,
        evt_config=f"{config_dir}/vov-test-evt-config.json",
        wo_mode="o",
        group="/evt/",
    )

    assert os.path.exists(outfile)
    assert len(ls(outfile, "/evt/")) == 4
    lstore = store.LH5Store()
    vov_ene, _ = lstore.read_object("/evt/energy", outfile)
    vov_aoe, _ = lstore.read_object("/evt/aoe", outfile)
    arr_ac, _ = lstore.read_object("/evt/multiplicity", outfile)
    assert isinstance(vov_ene, VectorOfVectors)
    assert isinstance(vov_aoe, VectorOfVectors)
    assert isinstance(arr_ac, Array)
    assert (np.diff(vov_ene.cumulative_length.nda, prepend=[0]) == arr_ac.nda).all()


def test_graceful_crashing(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    f_tcm = lgnd_test_data.get_path(tcm_path)
    f_dsp = lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp"))
    f_hit = lgnd_test_data.get_path(tcm_path.replace("tcm", "hit"))
    meta_path = None
    f_config = f"{config_dir}/basic-evt-config.json"

    with pytest.raises(ValueError):
        build_evt(f_dsp, f_tcm, f_hit, outfile, f_config, meta_path)

    with pytest.raises(NameError):
        build_evt(f_tcm, f_hit, f_dsp, outfile, f_config, meta_path)

    with pytest.raises(TypeError):
        build_evt(f_tcm, f_dsp, f_hit, outfile, None, meta_path)

    conf = {"operations": {}}
    with pytest.raises(ValueError):
        build_evt(f_tcm, f_dsp, f_hit, outfile, conf, meta_path)

    conf = {"channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]}}
    with pytest.raises(ValueError):
        build_evt(f_tcm, f_dsp, f_hit, outfile, conf, meta_path)

    conf = {
        "channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]},
        "operations": {},
    }
    build_evt(f_tcm, f_dsp, f_hit, outfile, conf, meta_path)
    assert not os.path.exists(outfile)

    conf = {
        "channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]},
        "operations": {
            "energy": {
                "channels": "geds_on",
                "mode": "first>pineapple",
                "get_ch": True,
                "expression": "cuspEmax_ctc_cal",
                "initial": "np.nan",
            }
        },
    }
    with pytest.raises(ValueError):
        build_evt(f_tcm, f_dsp, f_hit, outfile, conf, meta_path)

    conf = {
        "channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]},
        "operations": {
            "energy": {
                "channels": "geds_on",
                "mode": "first>25",
                "get_ch": True,
                "expression": "cuspEmax_ctc_cal$cuspEmax_ctc_cal",
                "initial": "np.nan",
            }
        },
    }
    with pytest.raises(SyntaxError):
        build_evt(f_tcm, f_dsp, f_hit, outfile, conf, meta_path)
