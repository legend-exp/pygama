import pytest
import fcutils
from pygama.raw.fc.fc_config_decoder import FCConfigDecoder


@pytest.fixture(scope='module')
def fcio_obj(lgnd_test_data):
    return fcutils.fcio(lgnd_test_data.get_path('fcio/th228.fcio'))


@pytest.fixture(scope='module')
def fcio_config(fcio_obj):
    decoder = FCConfigDecoder()
    decoder.decode_config(fcio_obj)
    return decoder.config
