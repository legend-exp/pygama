import numpy as np
import pytest

from pygama.evt.modules import larveto


def test_tc_time_pdf():
    assert isinstance(larveto.l200_tc_time_pdf(0), float)
    assert isinstance(
        larveto.l200_tc_time_pdf(np.array([0, -0.5, 3]) * 1e3), np.ndarray
    )

    with pytest.raises(ValueError):
        assert isinstance(larveto.l200_tc_time_pdf(-10000), float)
