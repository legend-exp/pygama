import lgdo.lh5 as lh5
import numpy as np

import pygama.pargen.lq_cal as lq
from pygama.math.distributions import gaussian


def test_lq_cal(lgnd_test_data):
    # test the HPGe calibration function
    # the function should return a calibration polynomial
    # that maps ADC channel to energy in keV

    # load test data here
    data = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )

    df = lh5.read_as("ch1104000/dsp", data, "pd")

    df["cuspEmax_cal"] = df["cuspEmax"] * 0.155

    cal_dict = {
        "LQ_Ecorr": {
            "expression": "lq80/cuspEmax",
            "parameters": {},
        }
    }

    lqcal = lq.LQCal(
        cal_dict,
        "cuspEmax_cal",
        "dt_eff",
        lambda x: np.sqrt(1.5 + 0.1 * x),
        selection_string="index==index",
        cdf=gaussian,
        debug_mode=True,
    )

    df["LQ_Ecorr"] = np.divide(df["lq80"], df["cuspEmax"])

    lqcal.calibrate(df, "LQ_Ecorr")
    assert (lqcal.cut_val > 0) & (~np.isnan(lqcal.cut_val))
    assert (~np.isnan(lqcal.low_side_sf.loc[1592.50]["sf"])) & (
        lqcal.low_side_sf.loc[1592.50]["sf"] > 95
    )
