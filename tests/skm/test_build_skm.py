import os
from pathlib import Path

import numpy as np
import pandas as pd

from pygama.evt import build_evt
from pygama.skm import build_skm

config_dir = Path(__file__).parent / "configs"
evt_config_dir = Path(__file__).parent.parent / "evt" / "configs"


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
        evt_config=f"{evt_config_dir}/vov-test-evt-config.json",
        wo_mode="o",
        group="/evt/",
        tcm_group="hardware_tcm_1",
    )

    skm_conf = f"{config_dir}/basic-skm-config.json"
    skm_out = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_skm.parquet"
    build_skm(outfile, skm_out, skm_conf, wo_mode="o", skim_format="hdf")

    assert os.path.exists(skm_out)
    df = pd.read_hdf(skm_out)
    assert df.index.name == "timestamp"
    assert "energy_0" in df.keys()
    assert "energy_1" in df.keys()
    assert "energy_2" in df.keys()
    assert "multiplicity" in df.keys()
    assert (df.multiplicity.to_numpy() <= 3).all()
    assert (
        np.nan_to_num(df.energy_0.to_numpy())
        + np.nan_to_num(df.energy_1.to_numpy())
        + np.nan_to_num(df.energy_2.to_numpy())
        == df.energy_sum.to_numpy()
    ).all()
    assert (np.nan_to_num(df.energy_0.to_numpy()) <= df.max_energy.to_numpy()).all()
    assert (np.nan_to_num(df.energy_1.to_numpy()) <= df.max_energy.to_numpy()).all()
    assert (np.nan_to_num(df.energy_2.to_numpy()) <= df.max_energy.to_numpy()).all()
