import numpy as np

import pygama.math.utils as pgu


def test_sizeof_fmt():
    size = pgu.sizeof_fmt(4012)
    assert size == "3.918 KB"


def test_get_par_names():
    names = pgu.get_par_names(pgu.sizeof_fmt)
    assert names == ["suffix"]
    names = pgu.get_par_names(pgu.print_fit_results)
    assert names == ["cov", "func", "title", "pad"]


def test_get_formatted_stats():
    stats = pgu.get_formatted_stats(10.009, 10.009, ndigs=4)
    assert stats == ("10.01", "10.01")


def test_print_fit_results(caplog):
    pgu.print_fit_results(
        np.ones((3,)), np.ones((3, 3)), func=None, title="Test", pad=True
    )
    assert [
        "Test:",
        "p0 = 1.0 +/- 1.0",
        "p1 = 1.0 +/- 1.0",
        "p2 = 1.0 +/- 1.0",
        "",
    ] == [rec.message for rec in caplog.records]
