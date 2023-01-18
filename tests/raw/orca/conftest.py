import pytest

from pygama.raw.orca.orca_streamer import OrcaStreamer


@pytest.fixture(scope="module")
def orca_stream(lgnd_test_data):
    orstr = OrcaStreamer()
    orstr.open_stream(
        lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca")
    )
    return orstr
