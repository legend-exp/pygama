import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from lgdo import Array, Table, VectorOfVectors, lh5
from lgdo.lh5 import LH5Store

from pygama.evt import build_evt

config_dir = Path(__file__).parent / "configs"
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


def test_basics(lgnd_test_data, files_config):
    build_evt(
        files_config,
        config=f"{config_dir}/basic-evt-config.yaml",
        wo_mode="of",
    )

    outfile = files_config["evt"][0]
    f_tcm = files_config["tcm"][0]

    evt = lh5.read("evt", outfile)

    assert "statement" in evt.multiplicity.attrs
    assert evt.multiplicity.attrs["statement"] == "0bb decay is real"

    assert os.path.exists(outfile)
    assert sorted(evt.keys()) == [
        "aoe",
        "energy",
        "energy_all_above1MeV",
        "energy_any_above1MeV",
        "energy_hit_idx",
        "energy_id",
        "energy_idx",
        "energy_sum",
        "is_aoe_rejected",
        "is_usable_aoe",
        "multiplicity",
        "timestamp",
    ]

    ak_evt = evt.view_as("ak")

    assert ak.all(ak_evt.energy_sum == ak.sum(ak_evt.energy, axis=-1))

    eid = store.read("/evt/energy_id", outfile)[0].view_as("np")
    eidx = store.read("/evt/energy_idx", outfile)[0].view_as("np")
    eidx = eidx[eidx != 999999999999]

    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    ids = ids[eidx]
    assert ak.all(ids == eid[eid != 0])

    ehidx = store.read("/evt/energy_hit_idx", outfile)[0].view_as("np")
    ids = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")
    ids = ids[eidx]
    assert ak.all(ids == ehidx[ehidx != 999999999999])


def test_field_nesting(lgnd_test_data, files_config):
    config = {
        "channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]},
        "outputs": [
            "sub1___timestamp",
            "sub2___multiplicity",
            "sub2___dummy",
        ],
        "operations": {
            "sub1___timestamp": {
                "channels": "geds_on",
                "aggregation_mode": "first_at:dsp.tp_0_est",
                "expression": "dsp.timestamp",
            },
            "sub2___multiplicity": {
                "channels": "geds_on",
                "aggregation_mode": "sum",
                "expression": "hit.cuspEmax_ctc_cal > 25",
                "initial": 0,
            },
            "sub2___dummy": {
                "channels": "geds_on",
                "aggregation_mode": "sum",
                "expression": "hit.cuspEmax_ctc_cal > evt.sub1___timestamp",
                "initial": 0,
            },
        },
    }

    build_evt(
        files_config,
        config=config,
        wo_mode="of",
    )

    outfile = files_config["evt"][0]
    evt = lh5.read("/evt", outfile)

    assert isinstance(evt, Table)
    assert isinstance(evt.sub1, Table)
    assert isinstance(evt.sub2, Table)
    assert isinstance(evt.sub1.timestamp, Array)

    assert sorted(evt.keys()) == ["sub1", "sub2"]
    assert sorted(evt.sub1.keys()) == ["timestamp"]
    assert sorted(evt.sub2.keys()) == ["dummy", "multiplicity"]


# FIXME: this can't be properly tested until proper testdata is available
# def test_spms_module(lgnd_test_data, files_config):
#     build_evt(
#         files_config,
#         config=f"{config_dir}/spms-module-config.yaml",
#         wo_mode="of",
#     )

#     outfile = files_config["evt"][0]

#     evt = lh5.read("/evt", outfile)

#     t0 = ak.fill_none(ak.nan_to_none(evt.t0.view_as("ak")), 48_000)
#     tr_pos = evt.trigger_pos.view_as("ak") * 16
#     assert ak.all(tr_pos > t0 - 30_000)
#     assert ak.all(tr_pos < t0 + 30_000)

#     mask = evt._pulse_mask
#     assert isinstance(mask, VectorOfVectors)
#     assert len(mask) == 10
#     assert mask.ndim == 3

#     full = evt.spms_amp_full.view_as("ak")
#     amp = evt.spms_amp.view_as("ak")
#     assert ak.all(amp > 0.1)

#     assert ak.all(full[mask.view_as("ak")] == amp)

#     wo_empty = evt.spms_amp_wo_empty.view_as("ak")
#     assert ak.all(wo_empty == amp[ak.count(amp, axis=-1) > 0])

#     rawids = evt.rawid.view_as("ak")
#     assert rawids.ndim == 2
#     assert ak.count(rawids) == 30

#     idx = evt.hit_idx.view_as("ak")
#     assert idx.ndim == 2
#     assert ak.count(idx) == 30

#     rawids_wo_empty = evt.rawid_wo_empty.view_as("ak")
#     assert ak.count(rawids_wo_empty) == 7

#     vhit = evt.is_valid_hit.view_as("ak")
#     vhit.show()
#     assert ak.all(ak.num(vhit, axis=-1) == ak.num(full, axis=-1))


def test_vov(lgnd_test_data, files_config):
    build_evt(
        files_config,
        config=f"{config_dir}/vov-test-evt-config.json",
        wo_mode="of",
    )

    outfile = files_config["evt"][0]
    f_tcm = files_config["tcm"][0]

    assert os.path.exists(outfile)
    assert len(lh5.ls(outfile, "/evt/")) == 12

    timestamp, _ = store.read("/evt/timestamp", outfile)
    assert np.all(~np.isnan(timestamp.nda))

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

    assert vov_ene.dtype == "float32"
    assert vov_aoe.dtype == "float64"
    assert arr_ac.dtype == "int16"

    assert (np.diff(vov_ene.cumulative_length.nda, prepend=[0]) == arr_ac.nda).all()

    vov_eid = store.read("/evt/energy_id", outfile)[0].view_as("ak")
    vov_eidx = store.read("/evt/energy_idx", outfile)[0].view_as("ak")
    vov_aoe_idx = store.read("/evt/aoe_idx", outfile)[0].view_as("ak")

    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("ak")
    ids = ak.unflatten(ids[ak.flatten(vov_eidx)], ak.count(vov_eidx, axis=-1))
    assert ak.all(ids == vov_eid)

    arr_ene = store.read("/evt/energy_sum", outfile)[0].view_as("ak")
    assert ak.all(
        ak.isclose(arr_ene, ak.nansum(vov_ene.view_as("ak"), axis=-1), rtol=1e-3)
    )
    assert ak.all(vov_aoe.view_as("ak") == vov_aoe_idx)


def test_graceful_crashing(lgnd_test_data, files_config):
    with pytest.raises(TypeError):
        build_evt(files_config, None, wo_mode="of")

    conf = {"operations": {}}
    with pytest.raises(ValueError):
        build_evt(files_config, conf, wo_mode="of")

    conf = {"channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]}}
    with pytest.raises(ValueError):
        build_evt(files_config, conf, wo_mode="of")

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
        build_evt(
            files_config,
            conf,
            wo_mode="of",
        )


def test_query(lgnd_test_data, files_config):
    build_evt(
        files_config,
        config=f"{config_dir}/query-test-evt-config.json",
        wo_mode="of",
    )
    outfile = files_config["evt"][0]

    assert len(lh5.ls(outfile, "/evt/")) == 12


def test_vector_sort(lgnd_test_data, files_config):
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

    build_evt(
        files_config,
        conf,
        wo_mode="of",
    )

    outfile = files_config["evt"][0]

    assert os.path.exists(outfile)
    assert len(lh5.ls(outfile, "/evt/")) == 4
    vov_t0, _ = store.read("/evt/t0_acend", outfile)
    nda_t0 = vov_t0.to_aoesa().view_as("np")
    assert ((np.diff(nda_t0) >= 0) | (np.isnan(np.diff(nda_t0)))).all()
    vov_t0, _ = store.read("/evt/t0_decend", outfile)
    nda_t0 = vov_t0.to_aoesa().view_as("np")
    assert ((np.diff(nda_t0) <= 0) | (np.isnan(np.diff(nda_t0)))).all()
