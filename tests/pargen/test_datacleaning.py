import lgdo.lh5 as lh5
from pytest import approx

import pygama.pargen.data_cleaning as dc


def test_cuts(lgnd_test_data):
    # load test data here
    data = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )
    tbl = lh5.read("ch1104000/dsp", data)

    cut_pars = {
        "bl_std_cut": {"cut_parameter": "bl_std", "cut_level": 4, "mode": "inclusive"},
        "dt_eff_cut": {
            "expression": "(dt_eff>a)&(dt_eff<b)",
            "parameters": {"a": 0, "b": 4000},
        },
        "bl_pileup_cut": {
            "cut_parameter": "baselineEmax",
            "cut_level": 4,
            "mode": "exclusive",
        },
    }

    cut_dict = dc.generate_cuts(tbl, cut_pars)

    assert cut_dict["dt_eff_cut"] == cut_pars["dt_eff_cut"]
    assert cut_dict["bl_std_cut"]["expression"] == "(bl_std>a) & (bl_std<b)"
    assert approx(cut_dict["bl_std_cut"]["parameters"]["a"], 0.1) == 10
    assert approx(cut_dict["bl_std_cut"]["parameters"]["b"], 0.1) == 15
    assert (
        cut_dict["bl_pileup_cut"]["expression"] == "(baselineEmax<a) | (baselineEmax>b)"
    )
    assert approx(cut_dict["bl_pileup_cut"]["parameters"]["a"], 0.1) == 7
    assert approx(cut_dict["bl_pileup_cut"]["parameters"]["b"], 0.1) == 24

    # check also works for df
    df = lh5.read_as("ch1104000/dsp", data, "pd")

    df["baselineEmax"].to_numpy()

    cut_dict_df = dc.generate_cuts(df, cut_pars)

    assert cut_dict_df == cut_dict

    ids = dc.get_cut_indexes(df, cut_pars)
    ids_tbl = dc.get_cut_indexes(tbl, cut_pars)

    assert (ids == ids_tbl).all()
