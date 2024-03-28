from pygama.evt import utils


def test_tier_data_tuple():
    files = utils.make_files_config(
        {
            "tcm": ("f1", "g1"),
            "dsp": ("f2", "g2"),
            "hit": ("f3", "g3"),
            "evt": ("f4", "g4"),
        }
    )

    assert files.raw == utils.H5DataLoc()
    assert files.tcm.file == "f1"
    assert files.tcm.group == "g1"
    assert files.dsp.file == "f2"
    assert files.dsp.group == "g2"
    assert files.hit.file == "f3"
    assert files.hit.group == "g3"
    assert files.evt.file == "f4"
    assert files.evt.group == "g4"
