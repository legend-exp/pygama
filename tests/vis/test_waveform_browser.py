from pathlib import Path

from pygama.vis import WaveformBrowser

config_dir = Path(__file__).parent / "configs"


def test_basics(lgnd_test_data):
    wb = WaveformBrowser(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        "/geds/raw",
        dsp_config=f"{config_dir}/hpge-dsp-config.json",
        lines=["wf_blsub", "wf_trap", "trapEmax"],
        legend=["waveform", "trapezoidal", "energy = {trapEmax:0.1f}"],
        styles="seaborn-v0.8",
        n_drawn=2,
        x_lim=("20*us", "60*us"),
        x_unit="us",
    )

    wb.draw_next()
    wb.draw_entry(24)
    wb.draw_entry((2, 24))


def test_entry_mask(lgnd_test_data):
    wb = WaveformBrowser(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        "/geds/raw",
        entry_list=[
            7,
            9,
            25,
            27,
            33,
            38,
            46,
            52,
            57,
            59,
            67,
            71,
            72,
            82,
            90,
            92,
            93,
            94,
            97,
        ],
    )

    wb.draw_next()
