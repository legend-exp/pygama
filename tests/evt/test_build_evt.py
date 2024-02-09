import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from lgdo import Array, VectorOfVectors, lh5
from lgdo.lh5 import LH5Store

from pygama.evt import build_evt

config_dir = Path(__file__).parent / "configs"
store = LH5Store()


def test_basics(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)

    build_evt(
        f_tcm=lgnd_test_data.get_path(tcm_path),
        f_dsp=lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")),
        f_hit=lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")),
        evt_config=f"{config_dir}/basic-evt-config.json",
        f_evt=outfile,
        wo_mode="o",
        evt_group="evt",
        hit_group="hit",
        dsp_group="dsp",
        tcm_group="hardware_tcm_1",
    )

    assert "statement" in store.read("/evt/multiplicity", outfile)[0].getattrs().keys()
    assert (
        store.read("/evt/multiplicity", outfile)[0].getattrs()["statement"]
        == "0bb decay is real"
    )
    assert os.path.exists(outfile)
    assert len(lh5.ls(outfile, "/evt/")) == 11
    nda = {
        e: store.read(f"/evt/{e}", outfile)[0].view_as("np")
        for e in ["energy", "energy_aux", "energy_sum", "multiplicity"]
    }
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

    eid = store.read("/evt/energy_id", outfile)[0].view_as("np")
    eidx = store.read("/evt/energy_idx", outfile)[0].view_as("np")
    eidx = eidx[eidx != 999999999999]

    ids = store.read("hardware_tcm_1/array_id", lgnd_test_data.get_path(tcm_path))[
        0
    ].view_as("np")
    ids = ids[eidx]
    assert ak.all(ids == eid[eid != 0])


def test_lar_module(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    build_evt(
        f_tcm=lgnd_test_data.get_path(tcm_path),
        f_dsp=lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")),
        f_hit=lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")),
        evt_config=f"{config_dir}/module-test-evt-config.json",
        f_evt=outfile,
        wo_mode="o",
        evt_group="evt",
        hit_group="hit",
        dsp_group="dsp",
        tcm_group="hardware_tcm_1",
    )

    assert os.path.exists(outfile)
    assert len(lh5.ls(outfile, "/evt/")) == 10
    nda = {
        e: store.read(f"/evt/{e}", outfile)[0].view_as("np")
        for e in ["lar_multiplicity", "lar_multiplicity_dplms", "t0", "lar_time_shift"]
    }
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
        evt_config=f"{config_dir}/module-test-t0-vov-evt-config.json",
        f_evt=outfile,
        wo_mode="o",
        evt_group="evt",
        hit_group="hit",
        dsp_group="dsp",
        tcm_group="hardware_tcm_1",
    )

    assert os.path.exists(outfile)
    assert len(lh5.ls(outfile, "/evt/")) == 12
    nda = {
        e: store.read(f"/evt/{e}", outfile)[0].view_as("np")
        for e in ["lar_multiplicity", "lar_multiplicity_dplms", "lar_time_shift"]
    }
    assert np.max(nda["lar_multiplicity"]) <= 3
    assert np.max(nda["lar_multiplicity_dplms"]) <= 3

    ch_idx = store.read("/evt/lar_tcm_index", outfile)[0].view_as("ak")
    pls_idx = store.read("/evt/lar_pulse_index", outfile)[0].view_as("ak")
    assert ak.count(ch_idx) == ak.count(pls_idx)
    assert ak.all(ak.count(ch_idx, axis=-1) == ak.count(pls_idx, axis=-1))


def test_vov(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    build_evt(
        f_tcm=lgnd_test_data.get_path(tcm_path),
        f_dsp=lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")),
        f_hit=lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")),
        evt_config=f"{config_dir}/vov-test-evt-config.json",
        f_evt=outfile,
        wo_mode="o",
        evt_group="evt",
        hit_group="hit",
        dsp_group="dsp",
        tcm_group="hardware_tcm_1",
    )

    assert os.path.exists(outfile)
    assert len(lh5.ls(outfile, "/evt/")) == 12
    vov_ene, _ = store.read("/evt/energy", outfile)
    vov_aoe, _ = store.read("/evt/aoe", outfile)
    arr_ac, _ = store.read("/evt/multiplicity", outfile)
    vov_aoeene, _ = store.read("/evt/energy_times_aoe", outfile)
    vov_eneac, _ = store.read("/evt/energy_times_multiplicity", outfile)
    arr_ac2, _ = store.read("/evt/multiplicity_squared", outfile)
    assert isinstance(vov_ene, VectorOfVectors)
    assert isinstance(vov_aoe, VectorOfVectors)
    assert isinstance(arr_ac, Array)
    assert isinstance(vov_aoeene, VectorOfVectors)
    assert isinstance(vov_eneac, VectorOfVectors)
    assert isinstance(arr_ac2, Array)
    assert (np.diff(vov_ene.cumulative_length.nda, prepend=[0]) == arr_ac.nda).all()

    vov_eid = store.read("/evt/energy_id", outfile)[0].view_as("ak")
    vov_eidx = store.read("/evt/energy_idx", outfile)[0].view_as("ak")
    vov_aoe_idx = store.read("/evt/aoe_idx", outfile)[0].view_as("ak")

    ids = store.read("hardware_tcm_1/array_id", lgnd_test_data.get_path(tcm_path))[
        0
    ].view_as("ak")
    ids = ak.unflatten(ids[ak.flatten(vov_eidx)], ak.count(vov_eidx, axis=-1))
    assert ak.all(ids == vov_eid)

    arr_ene = store.read("/evt/energy_sum", outfile)[0].view_as("ak")
    assert ak.all(arr_ene == ak.nansum(vov_ene.view_as("ak"), axis=-1))
    assert ak.all(vov_aoe.view_as("ak") == vov_aoe_idx)


def test_graceful_crashing(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    f_tcm = lgnd_test_data.get_path(tcm_path)
    f_dsp = lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp"))
    f_hit = lgnd_test_data.get_path(tcm_path.replace("tcm", "hit"))
    f_config = f"{config_dir}/basic-evt-config.json"

    with pytest.raises(KeyError):
        build_evt(f_dsp, f_tcm, f_hit, f_config, outfile)

    with pytest.raises(KeyError):
        build_evt(f_tcm, f_hit, f_dsp, f_config, outfile)

    with pytest.raises(TypeError):
        build_evt(f_tcm, f_dsp, f_hit, None, outfile)

    conf = {"operations": {}}
    with pytest.raises(ValueError):
        build_evt(f_tcm, f_dsp, f_hit, conf, outfile)

    conf = {"channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]}}
    with pytest.raises(ValueError):
        build_evt(f_tcm, f_dsp, f_hit, conf, outfile)

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
        build_evt(f_tcm, f_dsp, f_hit, conf, outfile)


def test_query(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    build_evt(
        f_tcm=lgnd_test_data.get_path(tcm_path),
        f_dsp=lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")),
        f_hit=lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")),
        evt_config=f"{config_dir}/query-test-evt-config.json",
        f_evt=outfile,
        wo_mode="o",
        evt_group="evt",
        hit_group="hit",
        dsp_group="dsp",
        tcm_group="hardware_tcm_1",
    )
    assert len(lh5.ls(outfile, "/evt/")) == 12


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
                "aggregation_mode": "keep_at_ch:evt.acend_id",
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
                "aggregation_mode": "keep_at_ch:evt.acend_id",
                "expression": "dsp.tp_0_est",
            },
        },
    }
    build_evt(f_tcm, f_dsp, f_hit, conf, outfile)

    assert os.path.exists(outfile)
    assert len(lh5.ls(outfile, "/evt/")) == 4
    vov_t0, _ = store.read("/evt/t0_acend", outfile)
    nda_t0 = vov_t0.to_aoesa().view_as("np")
    assert ((np.diff(nda_t0) >= 0) | (np.isnan(np.diff(nda_t0)))).all()
    vov_t0, _ = store.read("/evt/t0_decend", outfile)
    nda_t0 = vov_t0.to_aoesa().view_as("np")
    assert ((np.diff(nda_t0) <= 0) | (np.isnan(np.diff(nda_t0)))).all()


def test_tcm_id_table_pattern(lgnd_test_data, tmptestdir):
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    if os.path.exists(outfile):
        os.remove(outfile)
    f_tcm = lgnd_test_data.get_path(tcm_path)
    f_dsp = lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp"))
    f_hit = lgnd_test_data.get_path(tcm_path.replace("tcm", "hit"))
    f_config = f"{config_dir}/basic-evt-config.json"

    with pytest.raises(ValueError):
        build_evt(f_tcm, f_dsp, f_hit, f_config, outfile, tcm_id_table_pattern="ch{{}}")
    with pytest.raises(ValueError):
        build_evt(f_tcm, f_dsp, f_hit, f_config, outfile, tcm_id_table_pattern="ch{}{}")
    with pytest.raises(NotImplementedError):
        build_evt(
            f_tcm, f_dsp, f_hit, f_config, outfile, tcm_id_table_pattern="ch{tcm_id}"
        )
    with pytest.raises(ValueError):
        build_evt(
            f_tcm, f_dsp, f_hit, f_config, outfile, tcm_id_table_pattern="apple{}banana"
        )
