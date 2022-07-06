import os
from pathlib import Path

import pytest

from pygama.dsp import build_dsp
from pygama.raw import build_raw

config_dir = Path(__file__).parent/'configs'


@pytest.fixture(scope="module")
def multich_raw_file(lgnd_test_data):
    out_file = '/tmp/L200-comm-20211130-phy-spms.lh5'
    out_spec = {
      'FCEventDecoder': {
        'ch{key}': {
          'key_list': [[0, 6]],
          'out_stream': out_file + ":{name}/raw"
        }
      }
    }

    build_raw(in_stream=lgnd_test_data.get_path('fcio/L200-comm-20211130-phy-spms.fcio'),
              out_spec=out_spec, overwrite=True)

    return out_file


def test_build_dsp_basics(lgnd_test_data):
    build_dsp(lgnd_test_data.get_path('lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5'),
              '/tmp/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5',
              dsp_config=f'{config_dir}/icpc-dsp-config.json')

    assert os.path.exists('/tmp/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5')

    with pytest.raises(FileNotFoundError):
        build_dsp('non-existent-file.lh5',
                  '/tmp/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5',
                  dsp_config=f'{config_dir}/icpc-dsp-config.json')
