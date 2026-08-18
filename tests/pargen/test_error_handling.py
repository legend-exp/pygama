"""Tests for the error paths of the pargen modules."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import pygama.math.distributions as pgd
import pygama.pargen.data_cleaning as dc
import pygama.pargen.energy_optimisation as eo
import pygama.pargen.lq_cal as lqc
import pygama.pargen.survival_fractions as sf
from pygama.pargen import dsp_optimize, energy_cal
from pygama.pargen.utils import load_data, require_config_keys


def unsupported_func(x):
    return x


class TestRequireConfigKeys:
    def test_passes_when_complete(self):
        require_config_keys({"a": 1, "b": 2}, ["a", "b"])

    def test_raises_listing_missing_keys(self):
        with pytest.raises(
            KeyError, match=r"my_dict is missing required key\(s\): a, c"
        ):
            require_config_keys({"b": 2}, ["a", "b", "c"], name="my_dict")


class TestHPGeCalibration:
    def test_invalid_deg_raises(self):
        with pytest.raises(ValueError, match="invalid deg = -2"):
            energy_cal.HPGeCalibration("energy", [2614.5], 0.5, deg=-2)

    def test_invalid_guess_kev_raises(self):
        with pytest.raises(ValueError, match="invalid guess_kev = 0"):
            energy_cal.HPGeCalibration("energy", [2614.5], 0, deg=0)

    def test_get_hpge_energy_fixed_unsupported_func(self):
        with pytest.raises(NotImplementedError, match="unsupported_func"):
            energy_cal.get_hpge_energy_fixed(unsupported_func)

    def test_fwhm_fit_with_no_peaks_falls_back_to_nan(self, caplog):
        with caplog.at_level(logging.ERROR, logger="pygama.pargen.energy_cal"):
            results = energy_cal.HPGeCalibration.fit_energy_res_curve(
                energy_cal.FWHMLinear, np.array([]), np.array([]), np.array([])
            )
        assert np.isnan(np.array(results["parameters"])).all()
        assert "no valid peaks" in caplog.text


class TestDataCleaning:
    def test_get_keys_wrong_type(self):
        with pytest.raises(TypeError, match="got int"):
            dc.get_keys(42, {"cut": {"cut_parameter": "bl_std"}})

    def test_generate_cuts_wrong_data_type(self):
        with pytest.raises(ValueError, match="got list"):
            dc.generate_cuts(
                [1, 2, 3],
                {
                    "cut": {
                        "cut_parameter": "bl_std",
                        "cut_level": 4,
                        "mode": "inclusive",
                    }
                },
            )

    def test_generate_cuts_unknown_mode(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"bl_std": rng.normal(10, 1, 1000)})
        with pytest.raises(ValueError, match="banana"):
            dc.generate_cuts(
                data,
                {"cut": {"cut_parameter": "bl_std", "cut_level": 4, "mode": "banana"}},
            )

    def test_generate_cuts_bad_cut_level_type(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"bl_std": rng.normal(10, 1, 1000)})
        with pytest.raises(TypeError, match="got str"):
            dc.generate_cuts(
                data,
                {
                    "cut": {
                        "cut_parameter": "bl_std",
                        "cut_level": "4",
                        "mode": "inclusive",
                    }
                },
            )

    def test_generate_cuts_one_sided_cut_level(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"bl_std": rng.normal(10, 1, 1000)})
        cut_dict = dc.generate_cuts(
            data,
            {
                "cut": {
                    "cut_parameter": "bl_std",
                    "cut_level": {"high_side": 4},
                    "mode": "inclusive",
                }
            },
        )
        assert cut_dict["cut"]["expression"] == "bl_std<a"
        assert cut_dict["cut"]["parameters"]["a"] == pytest.approx(14, abs=1)

    def test_generate_cut_classifiers_one_sided_percentile(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"bl_std": rng.normal(10, 1, 1000)})
        cut_dict = dc.generate_cut_classifiers(
            data,
            {
                "cut": {
                    "cut_parameter": "bl_std",
                    "cut_percentile": {"high_side": 97.5},
                    "mode": "inclusive",
                    "method": "percentile",
                }
            },
        )
        assert cut_dict["cut"]["expression"] == "cut_classifier<a"
        # 97.5th percentile of a standard normal is ~1.96 sigma; the classifier
        # std comes from an fwhm-based mode estimate, so allow a loose window
        assert 1.5 < cut_dict["cut"]["parameters"]["a"] < 3.5

    def test_generate_cuts_unknown_parameter(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"bl_std": rng.normal(10, 1, 1000)})
        with pytest.raises(ValueError, match="not_a_column"):
            dc.generate_cuts(
                data,
                {
                    "cut": {
                        "cut_parameter": "not_a_column",
                        "cut_level": 4,
                        "mode": "inclusive",
                    }
                },
            )


class TestSurvivalFractions:
    def test_compton_sf_unknown_mode(self):
        with pytest.raises(ValueError, match="banana"):
            sf.compton_sf(np.array([1.0, 2.0, 3.0]), 1.5, mode="banana")

    def test_energy_guess_unsupported_func(self):
        with pytest.raises(NotImplementedError, match="unsupported_func"):
            sf.energy_guess(np.array([1.0, 2.0, 3.0]), unsupported_func)

    def test_fix_all_but_nevents_unsupported_func(self):
        with pytest.raises(NotImplementedError, match="unsupported_func"):
            sf.fix_all_but_nevents(unsupported_func)

    def test_get_bounds_unsupported_func(self):
        with pytest.raises(NotImplementedError, match="unsupported_func"):
            sf.get_bounds(unsupported_func, {})


class TestDspOptimize:
    def test_bad_sampling_rate_type(self):
        with pytest.raises(TypeError, match="got float"):
            dsp_optimize.BayesianOptimizer(
                acq_func="ei", batch_size=1, sampling_rate=3.0
            )

    def test_bad_acq_func(self):
        with pytest.raises(ValueError, match="banana"):
            dsp_optimize.BayesianOptimizer(acq_func="banana", batch_size=1)


class TestLoadData:
    def test_wrong_files_type(self):
        with pytest.raises(TypeError, match="got int"):
            load_data(42, "dsp", {}, {"energy"})

    def test_empty_files_list(self):
        with pytest.raises(ValueError, match="empty"):
            load_data([], "dsp", {}, {"energy"})

    def test_cal_dict_missing_expression(self):
        with pytest.raises(KeyError, match="'expression'"):
            load_data(
                ["dummy.lh5"], "dsp", {"energy_cal": {"parameters": {}}}, {"energy"}
            )


class TestStageTags:
    """A failing internal stage is named in the fallback log line."""

    def test_get_peak_fwhm_with_dt_corr_tags_staged_fit(self, monkeypatch, caplog):
        def boom(*_args, **_kwargs):
            msg = "synthetic fit blowup"
            raise RuntimeError(msg)

        monkeypatch.setattr(eo.pgc, "unbinned_staged_energy_fit", boom)

        rng = np.random.default_rng(0)
        energies = rng.normal(2039, 5, 5000)
        dt = np.zeros_like(energies)
        with caplog.at_level(
            logging.WARNING, logger="pygama.pargen.energy_optimisation"
        ):
            result = eo.get_peak_fwhm_with_dt_corr(
                energies, 0, dt, pgd.hpge_peak, 2039, (20, 20)
            )
        # fit failed -> nan tuple, and the failing stage is named
        assert np.isnan(result[0])
        assert result[-1] is None
        assert "staged energy fit failed" in caplog.text

    def test_lq_drift_time_correction_tags_binned_fit(self, monkeypatch, caplog):
        def boom(*_args, **_kwargs):
            msg = "synthetic binned fit blowup"
            raise RuntimeError(msg)

        monkeypatch.setattr(lqc, "binned_lq_fit", boom)

        cal = lqc.LQCal(
            cal_dicts={"foo": {}},
            cal_energy_param="e_cal",
            dt_param="dt",
            eres_func=lambda _e: 1.0,
        )
        df = pd.DataFrame(
            {
                "LQ_Timecorr": [0.0, 1.0],
                "e_cal": [1592.5, 1592.5],
                "dt": [0.0, 1.0],
            }
        )
        with caplog.at_level(logging.ERROR, logger="pygama.pargen.lq_cal"):
            cal.drift_time_correction(
                df, lq_param="LQ_Timecorr", cal_energy_param="e_cal"
            )
        assert np.isnan(cal.dt_fit_pars).all()
        assert "binned LQ fit at DEP failed" in caplog.text

    def test_hpge_fit_energy_peaks_tags_fit_window(self, monkeypatch, caplog):
        rng = np.random.default_rng(0)
        energies = np.concatenate(
            [
                rng.uniform(100, 26000, 20000),
                rng.normal(26145, 10, 10000),
            ]
        ).round()
        cal = energy_cal.HPGeCalibration(
            "energy", [2614.5], 2614.5 / 26145, deg=0, uncal_is_int=True
        )
        cal.hpge_get_energy_peaks(energies)

        def boom(*_args, **_kwargs):
            msg = "synthetic binning blowup"
            raise RuntimeError(msg)

        monkeypatch.setattr(energy_cal.pgh, "better_int_binning", boom)

        with caplog.at_level(logging.DEBUG, logger="pygama.pargen.energy_cal"):
            cal.hpge_fit_energy_peaks(
                energies, peak_pars=[(2614.5, (20, 20), pgd.hpge_peak)]
            )

        # the setup failure is contained by the per-peak fallback: the loop
        # completes, the peak is recorded as invalid with nan binning, and the
        # failing stage is named
        pk_dict = cal.results["hpge_fit_energy_peaks"]["peak_parameters"][2614.5]
        assert pk_dict["validity"] is False
        assert np.isnan(pk_dict["bin_width"])
        assert "computing fit window failed" in caplog.text
