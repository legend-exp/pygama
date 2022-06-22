import os.path
from pathlib import Path

import pytest

import pygama.raw as raw
from pygama.lgdo.lh5_store import LH5Store, ls

config_dir = Path(__file__).parent/'configs'


def test_build_raw_basics(lgnd_test_data):
    with pytest.raises(FileNotFoundError):
        raw.build_raw(in_stream='non-existent-file')

    with pytest.raises(FileNotFoundError):
        raw.build_raw(in_stream=lgnd_test_data.get_path('fcio/L200-comm-20211130-phy-spms.fcio'),
                      out_spec='non-existent-file.json')


def test_build_raw_fc(lgnd_test_data):
    raw.build_raw(in_stream=lgnd_test_data.get_path('fcio/L200-comm-20211130-phy-spms.fcio'))

    assert lgnd_test_data.get_path('fcio/L200-comm-20211130-phy-spms.lh5') != ''

    out_file = '/tmp/L200-comm-20211130-phy-spms.lh5'

    raw.build_raw(in_stream=lgnd_test_data.get_path('fcio/L200-comm-20211130-phy-spms.fcio'),
                  out_spec=out_file)

    assert os.path.exists('/tmp/L200-comm-20211130-phy-spms.lh5')


def test_build_raw_fc_out_spec(lgnd_test_data):
    out_file = '/tmp/L200-comm-20211130-phy-spms.lh5'
    out_spec = {
      'FCEventDecoder': {
        'spms': {
          'key_list': [[2, 4]],
          'out_stream': out_file
        }
      }
    }

    raw.build_raw(in_stream=lgnd_test_data.get_path('fcio/L200-comm-20211130-phy-spms.fcio'),
                  out_spec=out_spec, n_max=10)

    store = LH5Store()
    lh5_obj, n_rows = store.read_object('/spms', out_file)
    assert n_rows == 10
    assert (lh5_obj['channel'].nda == [2, 3, 4, 2, 3, 4, 2, 3, 4, 2]).all()

    raw.build_raw(in_stream=lgnd_test_data.get_path('fcio/L200-comm-20211130-phy-spms.fcio'),
                  out_spec=f'{config_dir}/fc-out-spec.json', n_max=10)


def test_build_raw_fc_channelwise_out_spec(lgnd_test_data):
    out_file = '/tmp/L200-comm-20211130-phy-spms.lh5'
    out_spec = {
      'FCEventDecoder': {
        's{key}': {
          'key_list': [[0, 6]],
          'out_stream': out_file
        }
      }
    }

    raw.build_raw(in_stream=lgnd_test_data.get_path('fcio/L200-comm-20211130-phy-spms.fcio'),
                  out_spec=out_spec)

    assert ls(out_file) == ['s0', 's1', 's2', 's3', 's4', 's5']


def test_build_raw_orca(lgnd_test_data):
    raw.build_raw(in_stream=lgnd_test_data.get_path('orca/fc/L200-comm-20220519-phy-geds.orca'))

    assert lgnd_test_data.get_path('orca/fc/L200-comm-20220519-phy-geds.lh5') != ''

    out_file = '/tmp/L200-comm-20220519-phy-geds.lh5'

    raw.build_raw(in_stream=lgnd_test_data.get_path('orca/fc/L200-comm-20220519-phy-geds.orca'),
                  out_spec=out_file)

    assert os.path.exists('/tmp/L200-comm-20220519-phy-geds.lh5')


def test_build_raw_orca_out_spec(lgnd_test_data):
    out_file = '/tmp/L200-comm-20220519-phy-geds.lh5'
    out_spec = {
      'ORFlashCamADCWaveformDecoder': {
        'geds': {
          'key_list': [[0, 5]],
          'out_stream': out_file
        }
      }
    }

    raw.build_raw(in_stream=lgnd_test_data.get_path('orca/fc/L200-comm-20220519-phy-geds.orca'),
                  out_spec=out_spec, n_max=10)

    store = LH5Store()
    lh5_obj, n_rows = store.read_object('/geds', out_file)
    assert n_rows == 10
    # assert (lh5_obj['channel'].nda == [2, 3, 4, 2, 3, 4, 2, 3, 4, 2]).all()

    raw.build_raw(in_stream=lgnd_test_data.get_path('orca/fc/L200-comm-20220519-phy-geds.orca'),
                  out_spec=f'{config_dir}/orca-out-spec.json', n_max=10)


def test_build_raw_overwrite(lgnd_test_data):
    with pytest.raises(RuntimeError):
        raw.build_raw(in_stream=lgnd_test_data.get_path('fcio/L200-comm-20211130-phy-spms.fcio'),
                      overwrite=False)
