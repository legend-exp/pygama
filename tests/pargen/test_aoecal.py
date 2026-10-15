from __future__ import annotations

import logging

import lh5
import numpy as np
import pytest

import pygama.pargen.AoE_cal as Coe


def _make_aoe(cal_dict=None):
    """Helper to build a CalAoE instance with standard test settings."""
    if cal_dict is None:
        cal_dict = {
            "AoE_Uncorr": {
                "expression": "A_max/cuspEmax",
                "parameters": {},
            }
        }
    return Coe.CalAoE(
        cal_dict,
        "cuspEmax_cal",
        lambda x: np.sqrt(1.5 + 0.1 * x),
        selection_string="index==index",
        debug_mode=True,
    )


def _load_test_df(lgnd_test_data):
    """Load the standard LH5 test data and add derived columns."""
    data = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )
    dsp_df = lh5.read_as("ch1104000/dsp", data, "pd")
    dsp_df["AoE_Uncorr"] = dsp_df["A_max"] / dsp_df["cuspEmax"]
    dsp_df["cuspEmax_cal"] = dsp_df["cuspEmax"] * 0.155
    return dsp_df


def test_aoe_cal(lgnd_test_data):
    data_df = _load_test_df(lgnd_test_data)

    aoe = _make_aoe()
    aoe.calibrate(data_df, "AoE_Uncorr")
    assert (
        (aoe.low_cut_val < -1.0) & (aoe.low_cut_val > -3) & (~np.isnan(aoe.low_cut_val))
    )
    assert (
        (~np.isnan(aoe.low_side_sfs.loc[2614.5]["sf"]))
        & (aoe.low_side_sfs.loc[2614.5]["sf"] < 20)
        & (aoe.low_side_sfs.loc[2614.5]["sf"] > 0)
    )


def test_aoe_cal_override_dict(lgnd_test_data):
    """The override_dict paths produce the same columns and cal_dict entries as a normal run."""
    # Reference run: calibrate in-place so we have the cal_dict entries afterwards.
    data_df = _load_test_df(lgnd_test_data)
    aoe_ref = _make_aoe()
    aoe_ref.calibrate(data_df, "AoE_Uncorr")

    # Build an override_dict from the entries stored by the reference run.
    override_dict = {
        key: aoe_ref.cal_dicts[key]
        for key in [
            "AoE_Timecorr",
            "AoE_Corrected",
            "_AoE_Classifier_intermediate",
            "AoE_Classifier",
            "AoE_Low_Cut",
        ]
    }

    # Override run: a fresh DataFrame, skip all fits.
    data_df2 = _load_test_df(lgnd_test_data)
    aoe_ovr = _make_aoe()
    aoe_ovr.calibrate(data_df2, "AoE_Uncorr", override_dict=override_dict)

    # All expected columns must be present.
    for col in (
        "AoE_Timecorr",
        "AoE_Corrected",
        "_AoE_Classifier_intermediate",
        "AoE_Classifier",
        "AoE_Low_Cut",
    ):
        assert col in data_df2.columns, (
            f"column '{col}' missing after override calibrate()"
        )

    # All overridden entries must land in cal_dicts.
    for key, val in override_dict.items():
        assert key in aoe_ovr.cal_dicts, (
            f"cal_dict key '{key}' missing after override calibrate()"
        )
        assert aoe_ovr.cal_dicts[key] == val, (
            f"cal_dict entry for '{key}' was not taken from override_dict"
        )

    # Numeric column values must be finite (no NaNs or Infs introduced by the override path).
    for col in ("AoE_Timecorr", "AoE_Corrected", "AoE_Classifier"):
        assert np.isfinite(data_df2[col].to_numpy()).all(), (
            f"column '{col}' contains non-finite values after override calibrate()"
        )

    # low_cut_val must be set from the override and match the stored parameter.
    assert hasattr(aoe_ovr, "low_cut_val"), (
        "low_cut_val not set after AoE_Low_Cut override"
    )
    assert np.isfinite(aoe_ovr.low_cut_val), (
        "low_cut_val is not finite after AoE_Low_Cut override"
    )
    assert aoe_ovr.low_cut_val == pytest.approx(aoe_ref.low_cut_val), (
        "low_cut_val from override does not match reference"
    )


def test_aoe_cal_override_partial_energy_warns(lgnd_test_data, caplog):
    """A partial energy-correction override (only one key) logs a warning and falls back to normal fitting."""
    data_df = _load_test_df(lgnd_test_data)

    # First run to get AoE_Corrected entry only.
    aoe_ref = _make_aoe()
    aoe_ref.calibrate(data_df.copy(), "AoE_Uncorr")

    # Supply only AoE_Corrected (missing AoE_Classifier) → should warn and run normal fit.
    override_dict_partial = {"AoE_Corrected": aoe_ref.cal_dicts["AoE_Corrected"]}

    data_df2 = _load_test_df(lgnd_test_data)
    aoe_partial = _make_aoe()
    with caplog.at_level(logging.WARNING, logger="pygama.pargen.AoE_cal"):
        aoe_partial.calibrate(
            data_df2, "AoE_Uncorr", override_dict=override_dict_partial
        )

    assert any(
        "AoE_Corrected" in rec.message and "AoE_Classifier" in rec.message
        for rec in caplog.records
        if rec.levelno == logging.WARNING
    ), "Expected a warning about partial energy-correction override"

    # Normal energy correction still ran, so the cut should be valid.
    assert hasattr(aoe_partial, "low_cut_val")
    assert np.isfinite(aoe_partial.low_cut_val)


def test_aoe_plot_data(lgnd_test_data):
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib.figure import Figure

    data_df = _load_test_df(lgnd_test_data)
    aoe = _make_aoe()
    aoe.calibrate(data_df, "AoE_Uncorr", suffix="sfx")
    assert aoe.suffix == "sfx"

    spec = Coe.get_spectra_data(aoe, data_df)
    sel = data_df.query(aoe.selection_string)
    expected, _ = np.histogram(
        sel.query("AoE_Low_Cut_sfx")["cuspEmax_cal"], bins=spec["edges"]
    )
    np.testing.assert_array_equal(spec["low_cut"], expected)  # suffixed column used
    assert (spec["before"] >= spec["double_cut"]).all()
    assert len(spec["inset"]["edges"]) == 200

    sf = Coe.get_sf_vs_energy_data(aoe, data_df)
    assert ((sf["sf"] >= 0) & (sf["sf"] <= 100)).all()

    hist = Coe.get_classifier_data(aoe, data_df)
    assert hist["counts"].shape == (699, 499)

    dt = Coe.get_dt_dep_data(aoe, data_df, eranges=[(1000, 1300), (1300, 1500)])
    assert set(dt) == {"1000-1300", "1300-1500"}
    assert dt["1000-1300"]["counts"].shape == (200, 100)

    bands = Coe.get_compt_bands_data(aoe, data_df, eranges=[(1000, 1020)])
    assert len(bands["1000-1020"]["counts"]) == 49

    corr = Coe.get_energy_corr_data(aoe, data_df)
    np.testing.assert_array_equal(corr["mean"], aoe.energy_corr_fits["mean"])

    cut = Coe.get_cut_fit_data(aoe, data_df)
    assert cut["function"] == "SigmoidFit"
    assert cut["low_cut"] == aoe.low_cut_val

    curves = Coe.get_survival_fraction_curves_data(aoe, data_df)
    assert set(curves["peaks"]) == {str(p) for p in aoe.low_side_peak_dfs}

    for plot in (
        Coe.plot_spectra,
        Coe.plot_sf_vs_energy,
        Coe.plot_classifier,
        Coe.plot_mean_fit,
        Coe.plot_sigma_fit,
        Coe.plot_cut_fit,
        Coe.plot_survival_fraction_curves,
    ):
        assert plot.data_func is not None
        assert isinstance(plot(aoe, data_df), Figure)
    assert isinstance(Coe.plot_dt_dep(aoe, data_df, eranges=[(1000, 1300)]), Figure)
    assert isinstance(
        Coe.plot_compt_bands_overlayed(aoe, data_df, eranges=[(1000, 1020)]), Figure
    )


@pytest.mark.filterwarnings("ignore:No artists with labels:UserWarning")
def test_aoe_plots_without_calibration():
    # plotting must never raise when calibration steps were skipped or failed
    import matplotlib as mpl
    import pandas as pd

    mpl.use("Agg")
    aoe = _make_aoe()
    df = pd.DataFrame(
        {"cuspEmax_cal": np.linspace(1000, 2000, 50), "dt_eff": np.ones(50)}
    )
    for plot in (
        Coe.plot_spectra,
        Coe.plot_sf_vs_energy,
        Coe.plot_classifier,
        Coe.plot_mean_fit,
        Coe.plot_sigma_fit,
        Coe.plot_cut_fit,
        Coe.plot_survival_fraction_curves,
    ):
        assert plot(aoe, df) is not None
        assert isinstance(plot.data_func(aoe, df), dict)
    assert Coe.plot_dt_dep(aoe, df, eranges=[(1000, 1300)]) is not None
    assert Coe.plot_compt_bands_overlayed(aoe, df, eranges=[(1000, 1020)]) is not None
