from pygama.lgdo import VectorOfEncodedVectors, compression
from pygama.lgdo.compression import RadwareSigcompress


def test_encode_decode_array(wftable):
    result = compression.encode_array(
        wftable.values, codec=RadwareSigcompress(codec_shift=-32768)
    )
    assert isinstance(result, VectorOfEncodedVectors)
    assert len(result) == len(wftable)
    assert result.attrs["codec"] == "radware_sigcompress"
    assert result.attrs["codec_shift"] == -32768

    compression.decode_array(result)
