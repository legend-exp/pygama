from pygama.lgdo import VectorOfEncodedVectors, compression


def test_encode_array(wftable):
    result = compression.encode_array(wftable.values, encoder="radware_sigcompress")
    assert isinstance(result, VectorOfEncodedVectors)
    assert len(result) == len(wftable)
    assert result.attrs["codec"] == "radware_sigcompress"
    assert result.attrs["codec_shift"] == -32768
