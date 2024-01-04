import os
from pathlib import Path

import lgdo.lh5_store as store
import numpy as np
import pytest
from lgdo import Array, VectorOfVectors, load_nda, ls

from pygama.evt import build_evt, skim_evt

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
        evt_config=f"{config_dir}/basic-evt-config.json",
        wo_mode="o",
        group="/evt/",
        tcm_group="hardware_tcm_1",
    )

    assert os.path.exists(outfile)
    assert len(ls(outfile, "/evt/")) == 10
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
        evt_config=f"{config_dir}/module-test-evt-config.json",
        wo_mode="o",
        group="/evt/",
    )

    assert os.path.exists(outfile)
    assert len(ls(outfile, "/evt/")) == 10
    nda = load_nda(
        outfile,
        ["lar_multiplicity", "lar_multiplicity_dplms", "t0", "lar_time_shift"],
        "/evt/",
    )
    assert np.max(nda["lar_multiplicity"]) <= 3
    assert np.max(nda["lar_multiplicity_dplms"]) <= 3
    assert ((nda["lar_time_shift"] + nda["t0"]) >= 0).all()


def test_lar_t0_vov_module(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    build_evt(
        f_tcm=lgnd_test_data.get_path(tcm_path),
        f_dsp=lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")),
        f_hit=lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")),
        f_evt=outfile,
        evt_config=f"{config_dir}/module-test-t0-vov-evt-config.json",
        wo_mode="o",
        group="/evt/",
    )

    assert os.path.exists(outfile)
    assert len(ls(outfile, "/evt/")) == 10
    nda = load_nda(
        outfile,
        ["lar_multiplicity", "lar_multiplicity_dplms", "lar_time_shift"],
        "/evt/",
    )
    assert np.max(nda["lar_multiplicity"]) <= 3
    assert np.max(nda["lar_multiplicity_dplms"]) <= 3


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
        evt_config=f"{config_dir}/vov-test-evt-config.json",
        wo_mode="o",
        group="/evt/",
    )

    assert os.path.exists(outfile)
    assert len(ls(outfile, "/evt/")) == 9
    lstore = store.LH5Store()
    vov_ene, _ = lstore.read("/evt/energy", outfile)
    vov_aoe, _ = lstore.read("/evt/aoe", outfile)
    arr_ac, _ = lstore.read("/evt/multiplicity", outfile)
    vov_aoeene, _ = lstore.read("/evt/energy_times_aoe", outfile)
    vov_eneac, _ = lstore.read("/evt/energy_times_multiplicity", outfile)
    arr_ac2, _ = lstore.read("/evt/multiplicity_squared", outfile)
    assert isinstance(vov_ene, VectorOfVectors)
    assert isinstance(vov_aoe, VectorOfVectors)
    assert isinstance(arr_ac, Array)
    assert isinstance(vov_aoeene, VectorOfVectors)
    assert isinstance(vov_eneac, VectorOfVectors)
    assert isinstance(arr_ac2, Array)
    assert (np.diff(vov_ene.cumulative_length.nda, prepend=[0]) == arr_ac.nda).all()


def test_graceful_crashing(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    f_tcm = lgnd_test_data.get_path(tcm_path)
    f_dsp = lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp"))
    f_hit = lgnd_test_data.get_path(tcm_path.replace("tcm", "hit"))
    f_config = f"{config_dir}/basic-evt-config.json"

    with pytest.raises(RuntimeError):
        build_evt(f_dsp, f_tcm, f_hit, outfile, f_config)

    with pytest.raises(RuntimeError):
        build_evt(f_tcm, f_hit, f_dsp, outfile, f_config)

    with pytest.raises(TypeError):
        build_evt(f_tcm, f_dsp, f_hit, outfile, None)

    conf = {"operations": {}}
    with pytest.raises(ValueError):
        build_evt(f_tcm, f_dsp, f_hit, outfile, conf)

    conf = {"channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]}}
    with pytest.raises(ValueError):
        build_evt(f_tcm, f_dsp, f_hit, outfile, conf)

    conf = {
        "channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]},
        "outputs": ["foo"],
        "operations": {
            "foo": {
                "channels": "geds_on",
                "aggregation_mode": "banana",
                "expression": "hit.cuspEmax_ctc_cal > a",
                "parameters": {"a": 25},
                "initial": 0,
            }
        },
    }
    with pytest.raises(ValueError):
        build_evt(f_tcm, f_dsp, f_hit, outfile, conf)


def test_query(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    build_evt(
        f_tcm=lgnd_test_data.get_path(tcm_path),
        f_dsp=lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")),
        f_hit=lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")),
        f_evt=outfile,
        evt_config=f"{config_dir}/query-test-evt-config.json",
        wo_mode="o",
        group="/evt/",
        tcm_group="hardware_tcm_1",
    )
    assert len(ls(outfile, "/evt/")) == 12


def test_vector_sort(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    f_tcm = lgnd_test_data.get_path(tcm_path)
    f_dsp = lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp"))
    f_hit = lgnd_test_data.get_path(tcm_path.replace("tcm", "hit"))

    conf = {
        "channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]},
        "outputs": ["acend_id", "t0_acend", "decend_id", "t0_decend"],
        "operations": {
            "acend_id": {
                "channels": "geds_on",
                "aggregation_mode": "gather",
                "query": "hit.cuspEmax_ctc_cal>25",
                "expression": "tcm.array_id",
                "sort": "ascend_by:dsp.tp_0_est",
            },
            "t0_acend": {
                "aggregation_mode": "keep_at:evt.acend_id",
                "expression": "dsp.tp_0_est",
            },
            "decend_id": {
                "channels": "geds_on",
                "aggregation_mode": "gather",
                "query": "hit.cuspEmax_ctc_cal>25",
                "expression": "tcm.array_id",
                "sort": "descend_by:dsp.tp_0_est",
            },
            "t0_decend": {
                "aggregation_mode": "keep_at:evt.acend_id",
                "expression": "dsp.tp_0_est",
            },
        },
    }
    build_evt(f_tcm, f_dsp, f_hit, outfile, conf)

    assert os.path.exists(outfile)
    assert len(ls(outfile, "/evt/")) == 4
    lstore = store.LH5Store()
    vov_t0, _ = lstore.read("/evt/t0_acend", outfile)
    nda_t0 = vov_t0.to_aoesa().nda
    assert ((np.diff(nda_t0) >= 0) | (np.isnan(np.diff(nda_t0)))).all()
    vov_t0, _ = lstore.read("/evt/t0_decend", outfile)
    nda_t0 = vov_t0.to_aoesa().nda
    assert ((np.diff(nda_t0) <= 0) | (np.isnan(np.diff(nda_t0)))).all()


def test_skimming(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    f_tcm = lgnd_test_data.get_path(tcm_path)
    f_dsp = lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp"))
    f_hit = lgnd_test_data.get_path(tcm_path.replace("tcm", "hit"))
    f_config = f"{config_dir}/vov-test-evt-config.json"
    build_evt(f_tcm, f_dsp, f_hit, outfile, f_config)

    lstore = store.LH5Store()
    ac = lstore.read("/evt/multiplicity", outfile)[0].nda
    ac = len(ac[ac == 3])

    outfile_skm = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_skm.lh5"

    skim_evt(outfile, "multiplicity == 3", None, outfile_skm, "n")
    assert ac == len(lstore.read("/evt/energy", outfile_skm)[0].to_aoesa().nda)

    skim_evt(outfile, "multiplicity == 3", None, None, "o")
    assert ac == len(lstore.read("/evt/energy", outfile)[0].to_aoesa().nda)

    with pytest.raises(ValueError):
        skim_evt(outfile, "multiplicity == 3", None, None, "bla")