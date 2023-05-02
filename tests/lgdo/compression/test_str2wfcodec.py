import pytest

from pygama.lgdo.compression import RadwareSigcompress
from pygama.lgdo.compression.utils import str2wfcodec


def test_str2wfcodec():
    assert str2wfcodec("RadwareSigcompress()") == RadwareSigcompress()
    assert str2wfcodec("RadwareSigcompress(codec_shift=-32768)") == RadwareSigcompress(
        codec_shift=-32768
    )
    assert str2wfcodec(
        " RadwareSigcompress( codec_shift = -32768 ) "
    ) == RadwareSigcompress(codec_shift=-32768)

    with pytest.raises(ValueError):
        assert str2wfcodec("RadwareSigcompress") == RadwareSigcompress(
            codec_shift=-32768
        )
        assert str2wfcodec("RadwareSigcompress(blabla)") == RadwareSigcompress(
            codec_shift=-32768
        )
