from __future__ import annotations

from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pytest
from lgdo import Table, WaveformTable

from pygama.pargen import noise_optimization as noise_optimization_module

mpl.use("Agg", force=True)


def test_noise_optimization_does_not_mutate_outputs(monkeypatch):
    outputs_seen = []

    def fake_run_one_dsp(_tb_data, dsp_proc_chain, db_dict=None):  # noqa: ARG001
        outputs_seen.append(dsp_proc_chain["outputs"].copy())
        return {"energy": SimpleNamespace(nda=np.array([1.0, 2.0, 3.0]))}

    def fake_simple_gaussian_fit(_energies, dx):
        return {"fom": dx, "fom_err": 0.1}

    monkeypatch.setattr(noise_optimization_module, "run_one_dsp", fake_run_one_dsp)
    monkeypatch.setattr(
        noise_optimization_module, "simple_gaussian_fit", fake_simple_gaussian_fit
    )

    # extra unread outputs (wf_presum, bl_mean) and two optimisation entries
    # sharing one ene_str: the grid runs must request exactly ["energy"]
    dsp_proc_chain = {"outputs": ["wf_psd", "energy", "wf_presum", "bl_mean"]}
    noise_optimization_module.noise_optimization(
        tb_data=object(),
        dsp_proc_chain=dsp_proc_chain,
        par_dsp={},
        opt_dict={
            "start": 1,
            "stop": 3,
            "step": 1,
            "step_val": 1,
            "optimization": {
                "trap": {
                    "dict_str": "etrap",
                    "filter_par": "rise",
                    "ene_str": "energy",
                },
                "cusp": {
                    "dict_str": "cusp",
                    "filter_par": "sigma",
                    "ene_str": "energy",
                },
            },
            "perform_fit": True,
            "dx": 1,
            "fit_deg": 1,
            "n_bootstrap_samples": 2,
        },
        _lh5_path="",
    )

    assert dsp_proc_chain["outputs"] == ["wf_psd", "energy", "wf_presum", "bl_mean"]
    assert outputs_seen == [["energy"], ["energy"]]


def _waveform_table(n_rows=10, wf_len=8):
    rng = np.random.default_rng(1)
    return Table(
        col_dict={
            "waveform": WaveformTable(
                values=rng.normal(size=(n_rows, wf_len)).astype(np.float32),
                dt=16.0,
                dt_units="ns",
            )
        }
    )


def _fake_psd_run_one_dsp(record, fft_field="wf_psd"):
    """A run_one_dsp stand-in whose 'PSD' is the batch's own waveform values."""

    def fake(tb_data, dsp_proc_chain, db_dict=None):  # noqa: ARG001
        record.append({"outputs": dsp_proc_chain["outputs"].copy(), "n": len(tb_data)})
        if dsp_proc_chain["outputs"] == [fft_field]:
            vals = tb_data["waveform"].values.nda.astype(np.float32)
            return {fft_field: SimpleNamespace(values=SimpleNamespace(nda=vals))}
        return {"energy": SimpleNamespace(nda=np.array([1.0, 2.0, 3.0]))}

    return fake


def test_batched_mean_psd_matches_the_full_mean(monkeypatch):
    tb_data = _waveform_table(n_rows=10)
    calls = []
    monkeypatch.setattr(
        noise_optimization_module, "run_one_dsp", _fake_psd_run_one_dsp(calls)
    )

    psd = noise_optimization_module._batched_mean_psd(
        tb_data, {"outputs": ["wf_psd", "energy"]}, {}, "wf_psd", batch_size=3
    )

    full_mean = tb_data["waveform"].values.nda.mean(axis=0, dtype=np.float64)
    assert np.allclose(psd, full_mean, rtol=1e-12)
    # 10 rows in batches of 3 -> 4 calls of sizes 3,3,3,1
    assert [c["n"] for c in calls] == [3, 3, 3, 1]
    # the plot run must only request the fft field
    assert all(c["outputs"] == ["wf_psd"] for c in calls)


def test_batched_mean_psd_rejects_empty_input(monkeypatch):
    monkeypatch.setattr(
        noise_optimization_module, "run_one_dsp", _fake_psd_run_one_dsp([])
    )
    with pytest.raises(ValueError, match="no waveforms"):
        noise_optimization_module._batched_mean_psd(
            _waveform_table(n_rows=0), {"outputs": ["wf_psd"]}, {}, "wf_psd", 3
        )


def test_display_path_batches_and_does_not_mutate(monkeypatch):
    tb_data = _waveform_table(n_rows=10)
    calls = []
    rng = np.random.default_rng(2)

    def fake_run_one_dsp(tb, dsp_proc_chain, db_dict=None):  # noqa: ARG001
        calls.append({"outputs": dsp_proc_chain["outputs"].copy(), "n": len(tb)})
        if dsp_proc_chain["outputs"] == ["wf_psd"]:
            vals = tb["waveform"].values.nda.astype(np.float32)
            return {"wf_psd": SimpleNamespace(values=SimpleNamespace(nda=vals))}
        return {"energy": SimpleNamespace(nda=rng.normal(5, 1, size=200))}

    monkeypatch.setattr(noise_optimization_module, "run_one_dsp", fake_run_one_dsp)

    dsp_proc_chain = {"outputs": ["wf_psd", "energy"]}
    res_dict, plot_dict = noise_optimization_module.noise_optimization(
        tb_data=tb_data,
        dsp_proc_chain=dsp_proc_chain,
        par_dsp={},
        opt_dict={
            "start": 1,
            "stop": 3,
            "step": 1,
            "step_val": 1,
            "optimization": {
                "trap": {"dict_str": "trap", "filter_par": "rise", "ene_str": "energy"}
            },
            "perform_fit": False,
            "percentile_low": 10,
            "percentile_high": 90,
            "dx": 1,
            "fit_deg": 1,
            "n_bootstrap_samples": 2,
            "plot_range": (0, 10),
            "fft_batch_size": 4,
        },
        _lh5_path="",
        display=1,
    )

    # fft plot ran in batches of <= fft_batch_size, requesting only wf_psd
    fft_calls = [c for c in calls if c["outputs"] == ["wf_psd"]]
    assert [c["n"] for c in fft_calls] == [4, 4, 2]
    # the plotted psd is the mean over all waveforms
    assert np.allclose(
        plot_dict["nopt"]["fft"]["psd"],
        tb_data["waveform"].values.nda.mean(axis=0, dtype=np.float64),
        rtol=1e-12,
    )
    assert plot_dict["nopt"]["fft"]["fig"] is not None
    assert len(plot_dict["nopt"]["fft"]["frequency"]) == len(
        plot_dict["nopt"]["fft"]["psd"]
    )
    # grid-search runs saw outputs without the fft field
    grid_calls = [c for c in calls if c["outputs"] != ["wf_psd"]]
    assert all(c["outputs"] == ["energy"] for c in grid_calls)
    # and the caller's chain is unmutated
    assert dsp_proc_chain["outputs"] == ["wf_psd", "energy"]
    assert "trap" in res_dict


def test_fft_batch_size_must_be_positive(monkeypatch):
    monkeypatch.setattr(
        noise_optimization_module, "run_one_dsp", _fake_psd_run_one_dsp([])
    )
    with pytest.raises(ValueError, match="fft_batch_size must be positive"):
        noise_optimization_module.noise_optimization(
            tb_data=_waveform_table(),
            dsp_proc_chain={"outputs": ["wf_psd", "energy"]},
            par_dsp={},
            opt_dict={
                "start": 1,
                "stop": 3,
                "step": 1,
                "step_val": 1,
                "optimization": {},
                "perform_fit": True,
                "fit_deg": 1,
                "fft_batch_size": 0,
            },
            _lh5_path="",
            display=1,
        )


def test_custom_fft_field_is_respected(monkeypatch):
    tb_data = _waveform_table(n_rows=6)
    calls = []
    monkeypatch.setattr(
        noise_optimization_module,
        "run_one_dsp",
        _fake_psd_run_one_dsp(calls, fft_field="my_psd"),
    )

    psd = noise_optimization_module._batched_mean_psd(
        tb_data, {"outputs": ["my_psd", "energy"]}, {}, "my_psd", batch_size=4
    )
    assert np.allclose(
        psd, tb_data["waveform"].values.nda.mean(axis=0, dtype=np.float64)
    )
    assert all(c["outputs"] == ["my_psd"] for c in calls)
