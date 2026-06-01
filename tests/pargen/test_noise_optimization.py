from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pygama.pargen import noise_optimization as noise_optimization_module


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

    dsp_proc_chain = {"outputs": ["wf_psd", "energy"]}
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
                    "dict_str": "trap",
                    "filter_par": "rise",
                    "ene_str": "energy",
                }
            },
            "perform_fit": True,
            "dx": 1,
            "fit_deg": 1,
            "n_bootstrap_samples": 2,
        },
        _lh5_path="",
    )

    assert dsp_proc_chain["outputs"] == ["wf_psd", "energy"]
    assert outputs_seen == [["energy"], ["energy"]]
