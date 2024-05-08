import numpy as np
import pytest

from pygama.evt.modules import larveto


def test_tc_time_pdf():
    assert isinstance(larveto.l200_tc_time_pdf(0), float)
    assert isinstance(
        larveto.l200_tc_time_pdf(np.array([0, -0.5, 3]) * 1e3), np.ndarray
    )
    assert len(larveto.l200_tc_time_pdf(np.array([]))) == 0

    with pytest.raises(ValueError):
        larveto.l200_tc_time_pdf([-10_000, 42])
    with pytest.raises(ValueError):
        larveto.l200_tc_time_pdf([0, 50_000])
