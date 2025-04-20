from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from lgdo import Array, Table, VectorOfVectors, lh5

from pygama.evt import build_evt

config_dir = Path(__file__).parent / "configs"


@pytest.fixture(scope="module")
def files_config_nowrite(lgnd_test_data):
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    return {
        "tcm": (lgnd_test_data.get_path(tcm_path), "hardware_tcm_1"),
        "dsp": (lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")), "dsp", "ch{}"),
        "hit": (lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")), "hit", "ch{}"),
        "evt": (None, "evt"),
    }


@pytest.fixture(scope="module")
def files_config_write(lgnd_test_data, tmp_dir):
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    outfile = f"{tmp_dir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"
    return {
        "tcm": (lgnd_test_data.get_path(tcm_path), "hardware_tcm_1"),
        "dsp": (lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")), "dsp", "ch{}"),
        "hit": (lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")), "hit", "ch{}"),
        "evt": (outfile, "evt"),
    }


def test_basics(files_config_nowrite):
    evt = build_evt(
        files_config_nowrite,
        config=f"{config_dir}/basic-evt-config.yaml",
    )

    assert "statement" in evt.multiplicity.attrs
    assert evt.multiplicity.attrs["statement"] == "0bb decay is real"
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

    eid = evt.energy_id.view_as("np")
    eidx = evt.energy_idx.view_as("np")
    eidx = eidx[eidx != 999999999999]

    ids = lh5.read(
        "hardware_tcm_1/table_key", files_config_nowrite["tcm"][0]
    ).flattened_data.nda
    ids = ids[eidx]
    assert ak.all(ids == eid[eid != 0])

    ehidx = evt.energy_hit_idx.view_as("np")
    ids = lh5.read(
        "hardware_tcm_1/row_in_table", files_config_nowrite["tcm"][0]
    ).flattened_data.nda
    ids = ids[eidx]
    assert ak.all(ids == ehidx[ehidx != 999999999999])


def test_field_nesting(files_config_nowrite):
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

    evt = build_evt(
        files_config_nowrite,
        config=config,
    )

    assert isinstance(evt, Table)
    assert isinstance(evt.sub1, Table)
    assert isinstance(evt.sub2, Table)
    assert isinstance(evt.sub1.timestamp, Array)

    assert sorted(evt.keys()) == ["sub1", "sub2"]
    assert sorted(evt.sub1.keys()) == ["timestamp"]
    assert sorted(evt.sub2.keys()) == ["dummy", "multiplicity"]


# FIXME: this can't be properly tested until proper testdata is available
# def test_spms_module(lgnd_test_data, files_config_nowrite):
#     build_evt(
#         files_config_nowrite,
#         config=f"{config_dir}/spms-module-config.yaml",
#         wo_mode="of",
#     )

#     outfile = files_config_nowrite["evt"][0]

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


def test_vov(files_config_nowrite):
    evt = build_evt(
        files_config_nowrite,
        config=f"{config_dir}/vov-test-evt-config.json",
    )

    f_tcm = files_config_nowrite["tcm"][0]

    assert len(evt.keys()) == 12

    assert np.all(~np.isnan(evt.timestamp.nda))

    assert isinstance(evt.energy, VectorOfVectors)
    assert isinstance(evt.aoe, VectorOfVectors)
    assert isinstance(evt.multiplicity, Array)
    assert isinstance(evt.energy_times_aoe, VectorOfVectors)
    assert isinstance(evt.energy_times_multiplicity, VectorOfVectors)
    assert isinstance(evt.multiplicity_squared, Array)

    assert evt.energy.dtype == "float32"
    assert evt.aoe.dtype == "float64"
    assert evt.multiplicity.dtype == "int16"

    assert (
        np.diff(evt.energy.cumulative_length.nda, prepend=[0]) == evt.multiplicity.nda
    ).all()

    ids = lh5.read_as("hardware_tcm_1/table_key", f_tcm, library="ak")
    ids = ak.unflatten(
        ak.flatten(ids)[ak.flatten(evt.energy_idx.view_as("ak"))],
        ak.count(evt.energy_idx.view_as("ak"), axis=-1),
    )
    assert ak.all(ids == evt.energy_id.view_as("ak"))

    assert ak.all(
        ak.isclose(
            evt.energy_sum.view_as("ak"),
            ak.nansum(evt.energy.view_as("ak"), axis=-1),
            rtol=1e-3,
        )
    )
    assert ak.all(evt.aoe.view_as("ak") == evt.aoe_idx.view_as("ak"))


def test_graceful_crashing(files_config_nowrite):
    with pytest.raises(TypeError):
        build_evt(files_config_nowrite, None)

    conf = {"operations": {}}
    with pytest.raises(ValueError):
        build_evt(files_config_nowrite, conf)

    conf = {"channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]}}
    with pytest.raises(ValueError):
        build_evt(files_config_nowrite, conf)

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
            files_config_nowrite,
            conf,
        )


def test_query(files_config_nowrite):
    evt = build_evt(
        files_config_nowrite,
        config=f"{config_dir}/query-test-evt-config.json",
    )

    assert len(evt.keys()) == 12


def test_vector_sort(files_config_nowrite):
    conf = {
        "channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]},
        "outputs": ["acend_id", "t0_acend", "decend_id", "t0_decend"],
        "operations": {
            "acend_id": {
                "channels": "geds_on",
                "aggregation_mode": "gather",
                "query": "hit.cuspEmax_ctc_cal>25",
                "expression": "tcm.table_key",
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
                "expression": "tcm.table_key",
                "sort": "descend_by:dsp.tp_0_est",
            },
            "t0_decend": {
                "aggregation_mode": "keep_at_ch:evt.acend_id",
                "expression": "dsp.tp_0_est",
            },
        },
    }

    evt = build_evt(
        files_config_nowrite,
        conf,
    )

    assert len(evt.keys()) == 4
    vov_t0 = evt.t0_acend
    nda_t0 = vov_t0.to_aoesa().view_as("np")
    assert ((np.diff(nda_t0) >= 0) | (np.isnan(np.diff(nda_t0)))).all()
    vov_t0 = evt.t0_decend
    nda_t0 = vov_t0.to_aoesa().view_as("np")
    assert ((np.diff(nda_t0) <= 0) | (np.isnan(np.diff(nda_t0)))).all()


def test_build_evt_write(files_config_write):
    build_evt(
        files_config_write,
        config=f"{config_dir}/basic-evt-config.yaml",
        wo_mode="of",
    )
    outfile = files_config_write["evt"][0]
    assert Path(outfile).exists()
