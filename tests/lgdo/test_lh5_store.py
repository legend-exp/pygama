import h5py
import numpy as np
import pandas as pd
import pytest

import pygama.lgdo as lgdo
import pygama.lgdo.lh5_store as lh5
from pygama.lgdo.lh5_store import LH5Iterator, LH5Store


@pytest.fixture(scope='module')
def lgnd_file(lgnd_test_data):
    return lgnd_test_data.get_path('lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5')


def test_init():
    LH5Store()


def test_gimme_file(lgnd_file):
    store = LH5Store(keep_open=True)

    f = store.gimme_file(lgnd_file)
    assert isinstance(f, h5py.File)
    assert store.files[lgnd_file] == f

    with pytest.raises(FileNotFoundError):
        store.gimme_file('non-existent-file')


def test_gimme_group(lgnd_file):
    f = h5py.File(lgnd_file)
    store = LH5Store()
    g = store.gimme_group('/geds', f)
    assert isinstance(g, h5py.Group)

    f = h5py.File('/tmp/testfile.lh5', mode='w')
    g = store.gimme_group('/geds', f, grp_attrs={'attr1': 1}, overwrite=True)
    assert isinstance(g, h5py.Group)


def test_ls(lgnd_file):
    assert lh5.ls(lgnd_file) == ['geds']
    assert lh5.ls(lgnd_file, '/*/raw') == ['geds/raw']
    assert lh5.ls(lgnd_file, 'geds/raw/') == [
        'geds/raw/baseline',
        'geds/raw/channel',
        'geds/raw/energy',
        'geds/raw/ievt',
        'geds/raw/numtraces',
        'geds/raw/packet_id',
        'geds/raw/timestamp',
        'geds/raw/tracelist',
        'geds/raw/waveform',
        'geds/raw/wf_max',
        'geds/raw/wf_std'
    ]


def test_load_nda(lgnd_file):
    nda = lh5.load_nda([lgnd_file, lgnd_file],
                       ['baseline', 'waveform/values'],
                       lh5_group='/geds/raw',
                       idx_list=[[1, 3, 5], [2, 6, 7]])

    assert isinstance(nda, dict)
    assert isinstance(nda['baseline'], np.ndarray)
    assert nda['baseline'].shape == (6,)
    assert isinstance(nda['waveform/values'], np.ndarray)
    assert nda['waveform/values'].shape == (6, 5592)


def test_load_dfs(lgnd_file):
    dfs = lh5.load_dfs([lgnd_file, lgnd_file],
                       ['baseline', 'waveform/t0'],
                       lh5_group='/geds/raw',
                       idx_list=[[1, 3, 5], [2, 6, 7]])

    assert isinstance(dfs, pd.DataFrame)


@pytest.fixture(scope='module')
def lh5_file():
    store = LH5Store()

    struct = lgdo.Struct()
    struct.add_field('scalar', lgdo.Scalar(value=10))
    struct.add_field('array', lgdo.Array(nda=np.array([1, 2, 3, 4, 5])))
    struct.add_field('aoesa', lgdo.ArrayOfEqualSizedArrays(shape=(5, 5), dtype=np.float32, fill_val=42))
    struct.add_field('vov', lgdo.VectorOfVectors(
        flattened_data=lgdo.Array(nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])),
        cumulative_length=lgdo.Array(nda=np.array([2, 5, 6, 10, 13]))
    ))

    col_dict = {
        'a': lgdo.Array(nda=np.array([1, 2, 3, 4])),
        'b': lgdo.Array(nda=np.array([5, 6, 7, 8]))
    }

    struct.add_field('table', lgdo.Table(col_dict=col_dict))

    store.write_object(struct, 'struct',
                       '/tmp/tmp-pygama-lgdo-types.lh5',
                       group='/data',
                       start_row=1, n_rows=3,
                       wo_mode='overwrite_file')

    return '/tmp/tmp-pygama-lgdo-types.lh5'


def test_write_objects(lh5_file):
    pass


def test_read_scalar(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object('/data/struct/scalar', lh5_file)
    assert isinstance(lh5_obj, lgdo.Scalar)
    assert lh5_obj.value == 10
    assert n_rows == 1


def test_read_array(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object('/data/struct/array', lh5_file)
    assert isinstance(lh5_obj, lgdo.Array)
    assert (lh5_obj.nda == np.array([2, 3, 4])).all()
    assert n_rows == 3


def test_read_vov(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object('/data/struct/vov', lh5_file)
    assert isinstance(lh5_obj, lgdo.VectorOfVectors)

    desired = [np.array([3, 4, 5, 2]),
               np.array([]),
               np.array([])]

    for i in range(len(desired)):
        assert (desired[i] == list(lh5_obj)[i]).all()

    assert n_rows == 3


def test_read_aoesa(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object('/data/struct/aoesa', lh5_file)
    assert isinstance(lh5_obj, lgdo.ArrayOfEqualSizedArrays)
    assert (lh5_obj.nda == np.full((3, 5), fill_value=42)).all()


def test_read_table(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object('/data/struct/table', lh5_file)
    assert isinstance(lh5_obj, lgdo.Table)
    assert n_rows == 3

    lh5_obj, n_rows = store.read_object('/data/struct/table', [lh5_file, lh5_file])
    assert n_rows == 6


def test_read_lgnd_array(lgnd_file):
    store = LH5Store()

    lh5_obj, n_rows = store.read_object('/geds/raw/baseline', lgnd_file)
    assert isinstance(lh5_obj, lgdo.Array)
    assert n_rows == 100
    assert len(lh5_obj) == 100

    lh5_obj, n_rows = store.read_object('/geds/raw/waveform/values', lgnd_file)
    assert isinstance(lh5_obj, lgdo.ArrayOfEqualSizedArrays)


def test_read_array_concatenation(lgnd_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object('/geds/raw/baseline', [lgnd_file, lgnd_file])
    assert isinstance(lh5_obj, lgdo.Array)
    assert n_rows == 200
    assert len(lh5_obj) == 200


def test_read_lgnd_waveform_table(lgnd_file):
    store = LH5Store()

    lh5_obj, n_rows = store.read_object('/geds/raw/waveform', lgnd_file)
    assert isinstance(lh5_obj, lgdo.WaveformTable)

    lh5_obj, n_rows = store.read_object('/geds/raw/waveform', lgnd_file,
                                        start_row=10, n_rows=10,
                                        field_mask=['t0', 'dt'])

    assert isinstance(lh5_obj, lgdo.Table)
    assert list(lh5_obj.keys()) == ['t0', 'dt']
    assert len(lh5_obj) == 10


def test_lh5_iterator(lgnd_file):
    lh5_it = LH5Iterator(lgnd_file, '/geds/raw',
                         entry_list=range(100),
                         field_mask=['baseline'],
                         buffer_len=5)

    lh5_obj, n_rows = lh5_it.read(4)
    assert n_rows == 5
    assert isinstance(lh5_obj, lgdo.Table)
    assert list(lh5_obj.keys()) == ['baseline']
    assert (lh5_obj['baseline'].nda == np.array([14353, 14254, 14525, 11656, 13576])).all()

    for lh5_obj, entry, n_rows in lh5_it:
        assert len(lh5_obj) == 5
        assert n_rows == 5
        assert entry % 5 == 0
