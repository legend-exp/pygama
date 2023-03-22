import logging
import os

import h5py
import numpy as np
import pandas as pd
import pytest

import pygama.lgdo as lgdo
import pygama.lgdo.lh5_store as lh5
from pygama.lgdo import compression
from pygama.lgdo.compression import RadwareSigcompress
from pygama.lgdo.lh5_store import LH5Store


@pytest.fixture(scope="module")
def lgnd_file(lgnd_test_data):
    return lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5")


def test_init():
    LH5Store()


def test_gimme_file(lgnd_file):
    store = LH5Store(keep_open=True)

    f = store.gimme_file(lgnd_file)
    assert isinstance(f, h5py.File)
    assert store.files[lgnd_file] == f

    with pytest.raises(FileNotFoundError):
        store.gimme_file("non-existent-file")


def test_gimme_group(lgnd_file):
    f = h5py.File(lgnd_file)
    store = LH5Store()
    g = store.gimme_group("/geds", f)
    assert isinstance(g, h5py.Group)

    f = h5py.File("/tmp/testfile.lh5", mode="w")
    g = store.gimme_group("/geds", f, grp_attrs={"attr1": 1}, overwrite=True)
    assert isinstance(g, h5py.Group)


def test_show(lgnd_file):
    lh5.show(lgnd_file)
    lh5.show(lgnd_file, "/geds/raw")
    lh5.show(lgnd_file, "geds/raw")


def test_ls(lgnd_file):
    assert lh5.ls(lgnd_file) == ["geds"]
    assert lh5.ls(lgnd_file, "/*/raw") == ["geds/raw"]
    assert lh5.ls(lgnd_file, "geds/raw/") == [
        "geds/raw/baseline",
        "geds/raw/channel",
        "geds/raw/energy",
        "geds/raw/ievt",
        "geds/raw/numtraces",
        "geds/raw/packet_id",
        "geds/raw/timestamp",
        "geds/raw/tracelist",
        "geds/raw/waveform",
        "geds/raw/wf_max",
        "geds/raw/wf_std",
    ]


def test_load_nda(lgnd_file):
    nda = lh5.load_nda(
        [lgnd_file, lgnd_file],
        ["baseline", "waveform/values"],
        lh5_group="/geds/raw",
        idx_list=[[1, 3, 5], [2, 6, 7]],
    )

    assert isinstance(nda, dict)
    assert isinstance(nda["baseline"], np.ndarray)
    assert nda["baseline"].shape == (6,)
    assert isinstance(nda["waveform/values"], np.ndarray)
    assert nda["waveform/values"].shape == (6, 5592)


def test_load_dfs(lgnd_file):
    dfs = lh5.load_dfs(
        [lgnd_file, lgnd_file],
        ["baseline", "waveform/t0"],
        lh5_group="/geds/raw",
        idx_list=[[1, 3, 5], [2, 6, 7]],
    )

    assert isinstance(dfs, pd.DataFrame)


@pytest.fixture(scope="module")
def lh5_file():
    store = LH5Store()

    struct = lgdo.Struct()
    struct.add_field("scalar", lgdo.Scalar(value=10, attrs={"sth": 1}))
    struct.add_field("array", lgdo.Array(nda=np.array([1, 2, 3, 4, 5])))
    struct.add_field(
        "aoesa",
        lgdo.ArrayOfEqualSizedArrays(shape=(5, 5), dtype=np.float32, fill_val=42),
    )
    struct.add_field(
        "vov",
        lgdo.VectorOfVectors(
            flattened_data=lgdo.Array(
                nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])
            ),
            cumulative_length=lgdo.Array(nda=np.array([2, 5, 6, 10, 13])),
            attrs={"myattr": 2},
        ),
    )

    struct.add_field(
        "voev",
        lgdo.VectorOfEncodedVectors(
            encoded_data=lgdo.VectorOfVectors(
                flattened_data=lgdo.Array(
                    nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])
                ),
                cumulative_length=lgdo.Array(nda=np.array([2, 5, 6, 10, 13])),
            ),
            decoded_size=lgdo.Array(shape=5, fill_val=6),
        ),
    )

    col_dict = {
        "a": lgdo.Array(nda=np.array([1, 2, 3, 4]), attrs={"attr": 9}),
        "b": lgdo.Array(nda=np.array([5, 6, 7, 8]), attrs={"compression": "gzip"}),
    }

    struct.add_field("table", lgdo.Table(col_dict=col_dict, attrs={"stuff": 5}))

    struct.add_field(
        "wftable",
        lgdo.WaveformTable(
            t0=lgdo.Array(np.zeros(10)),
            dt=lgdo.Array(np.full(10, fill_value=1)),
            values=lgdo.ArrayOfEqualSizedArrays(
                shape=(10, 1000), dtype=np.uint16, fill_val=100, attrs={"custom": 8}
            ),
        ),
    )

    struct.add_field(
        "wftable_enc",
        lgdo.WaveformTable(
            t0=lgdo.Array(np.zeros(10)),
            dt=lgdo.Array(np.full(10, fill_value=1)),
            values=compression.encode(
                struct["wftable"].values, codec=RadwareSigcompress(codec_shift=-32768)
            ),
        ),
    )

    store.write_object(
        struct,
        "struct",
        "/tmp/tmp-pygama-lgdo-types.lh5",
        group="/data",
        start_row=1,
        n_rows=3,
        wo_mode="overwrite_file",
    )

    store.write_object(
        struct,
        "struct_full",
        "/tmp/tmp-pygama-lgdo-types.lh5",
        group="/data",
        wo_mode="append",
    )

    return "/tmp/tmp-pygama-lgdo-types.lh5"


def test_write_objects(lh5_file):
    pass


def test_read_n_rows(lh5_file):
    store = LH5Store()
    assert store.read_n_rows("/data/struct_full/aoesa", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/array", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/scalar", lh5_file) is None
    assert store.read_n_rows("/data/struct_full/table", lh5_file) == 4
    assert store.read_n_rows("/data/struct_full/voev", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/vov", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/wftable", lh5_file) == 10
    assert store.read_n_rows("/data/struct_full/wftable_enc/values", lh5_file) == 10


def test_get_buffer(lh5_file):
    store = LH5Store()
    buf = store.get_buffer("/data/struct_full/wftable_enc", lh5_file)
    assert isinstance(buf.values, lgdo.ArrayOfEqualSizedArrays)


def test_read_scalar(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/data/struct/scalar", lh5_file)
    assert isinstance(lh5_obj, lgdo.Scalar)
    assert lh5_obj.value == 10
    assert n_rows == 1
    assert lh5_obj.attrs["sth"] == 1


def test_read_array(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/data/struct/array", lh5_file)
    assert isinstance(lh5_obj, lgdo.Array)
    assert (lh5_obj.nda == np.array([2, 3, 4])).all()
    assert n_rows == 3


def test_read_array_fancy_idx(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object(
        "/data/struct_full/array", lh5_file, idx=[0, 3, 4]
    )
    assert isinstance(lh5_obj, lgdo.Array)
    assert (lh5_obj.nda == np.array([1, 4, 5])).all()
    assert n_rows == 3


def test_read_vov(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/data/struct/vov", lh5_file)
    assert isinstance(lh5_obj, lgdo.VectorOfVectors)

    desired = [np.array([3, 4, 5]), np.array([2]), np.array([4, 8, 9, 7])]

    for i in range(len(desired)):
        assert (desired[i] == list(lh5_obj)[i]).all()

    assert n_rows == 3
    assert lh5_obj.attrs["myattr"] == 2


def test_read_vov_fancy_idx(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/data/struct_full/vov", lh5_file, idx=[0, 2])
    assert isinstance(lh5_obj, lgdo.VectorOfVectors)

    desired = [np.array([1, 2]), np.array([2])]

    for i in range(len(desired)):
        assert (desired[i] == list(lh5_obj)[i]).all()

    assert n_rows == 2


def test_read_voev(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/data/struct/voev", lh5_file, decompress=False)
    assert isinstance(lh5_obj, lgdo.VectorOfEncodedVectors)

    desired = [np.array([3, 4, 5]), np.array([2]), np.array([4, 8, 9, 7])]

    for i in range(len(desired)):
        assert (desired[i] == lh5_obj[i][0]).all()

    assert n_rows == 3

    lh5_obj, n_rows = store.read_object(
        "/data/struct/voev", [lh5_file, lh5_file], decompress=False
    )
    assert isinstance(lh5_obj, lgdo.VectorOfEncodedVectors)
    assert n_rows == 6


def test_read_voev_fancy_idx(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object(
        "/data/struct_full/voev", lh5_file, idx=[0, 2], decompress=False
    )
    assert isinstance(lh5_obj, lgdo.VectorOfEncodedVectors)

    desired = [np.array([1, 2]), np.array([2])]

    for i in range(len(desired)):
        assert (desired[i] == lh5_obj[i][0]).all()

    assert n_rows == 2


def test_read_aoesa(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/data/struct/aoesa", lh5_file)
    assert isinstance(lh5_obj, lgdo.ArrayOfEqualSizedArrays)
    assert (lh5_obj.nda == np.full((3, 5), fill_value=42)).all()


def test_read_table(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/data/struct/table", lh5_file)
    assert isinstance(lh5_obj, lgdo.Table)
    assert n_rows == 3

    lh5_obj, n_rows = store.read_object("/data/struct/table", [lh5_file, lh5_file])
    assert n_rows == 6
    assert lh5_obj.attrs["stuff"] == 5
    assert lh5_obj["a"].attrs["attr"] == 9


def test_read_hdf5_compressed_data(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/data/struct/table", lh5_file)

    assert "compression" not in lh5_obj["b"].attrs
    with h5py.File(lh5_file) as h5f:
        assert h5f["/data/struct/table/b"].compression == "gzip"
        assert h5f["/data/struct/table/a"].compression is None


def test_read_wftable(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/data/struct/wftable", lh5_file)
    assert isinstance(lh5_obj, lgdo.WaveformTable)
    assert n_rows == 3

    lh5_obj, n_rows = store.read_object("/data/struct/wftable", [lh5_file, lh5_file])
    assert n_rows == 6
    assert lh5_obj.values.attrs["custom"] == 8


def test_read_wftable_encoded(lh5_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object(
        "/data/struct/wftable_enc", lh5_file, decompress=False
    )
    assert isinstance(lh5_obj, lgdo.WaveformTable)
    assert isinstance(lh5_obj.values, lgdo.ArrayOfEncodedEqualSizedArrays)
    assert n_rows == 3
    assert lh5_obj.values.attrs["codec"] == "radware_sigcompress"
    assert "codec_shift" in lh5_obj.values.attrs

    lh5_obj, n_rows = store.read_object("/data/struct/wftable_enc", lh5_file)
    assert isinstance(lh5_obj, lgdo.WaveformTable)
    assert isinstance(lh5_obj.values, lgdo.ArrayOfEqualSizedArrays)
    assert n_rows == 3

    lh5_obj, n_rows = store.read_object("/data/struct/wftable_enc/values", lh5_file)
    assert isinstance(lh5_obj, lgdo.ArrayOfEqualSizedArrays)
    assert n_rows == 3

    lh5_obj, n_rows = store.read_object(
        "/data/struct/wftable_enc", [lh5_file, lh5_file], decompress=False
    )
    assert n_rows == 6

    lh5_obj, n_rows = store.read_object(
        "/data/struct/wftable_enc", [lh5_file, lh5_file], decompress=True
    )
    assert n_rows == 6


def test_read_with_field_mask(lh5_file):
    store = LH5Store()

    lh5_obj, n_rows = store.read_object(
        "/data/struct_full", lh5_file, field_mask=["array"]
    )
    assert list(lh5_obj.keys()) == ["array"]

    lh5_obj, n_rows = store.read_object(
        "/data/struct_full", lh5_file, field_mask=("array", "table")
    )
    assert list(lh5_obj.keys()) == ["array", "table"]

    lh5_obj, n_rows = store.read_object(
        "/data/struct_full", lh5_file, field_mask={"array": True}
    )
    assert list(lh5_obj.keys()) == ["array"]

    lh5_obj, n_rows = store.read_object(
        "/data/struct_full", lh5_file, field_mask={"vov": False, "voev": False}
    )
    assert list(lh5_obj.keys()) == [
        "scalar",
        "array",
        "aoesa",
        "table",
        "wftable",
        "wftable_enc",
    ]


def test_read_lgnd_array(lgnd_file):
    store = LH5Store()

    lh5_obj, n_rows = store.read_object("/geds/raw/baseline", lgnd_file)
    assert isinstance(lh5_obj, lgdo.Array)
    assert n_rows == 100
    assert len(lh5_obj) == 100

    lh5_obj, n_rows = store.read_object("/geds/raw/waveform/values", lgnd_file)
    assert isinstance(lh5_obj, lgdo.ArrayOfEqualSizedArrays)


def test_read_lgnd_array_fancy_idx(lgnd_file):
    store = LH5Store()

    lh5_obj, n_rows = store.read_object(
        "/geds/raw/baseline", lgnd_file, idx=[2, 4, 6, 9, 11, 16, 68]
    )
    assert isinstance(lh5_obj, lgdo.Array)
    assert n_rows == 7
    assert len(lh5_obj) == 7
    assert (lh5_obj.nda == [13508, 14353, 14525, 14341, 15079, 11675, 13995]).all()


def test_read_lgnd_vov(lgnd_file):
    store = LH5Store()

    lh5_obj, n_rows = store.read_object("/geds/raw/tracelist", lgnd_file)
    assert isinstance(lh5_obj, lgdo.VectorOfVectors)
    assert n_rows == 100
    assert len(lh5_obj) == 100


def test_read_lgnd_vov_fancy_idx(lgnd_file):
    store = LH5Store()

    lh5_obj, n_rows = store.read_object(
        "/geds/raw/tracelist", lgnd_file, idx=[2, 4, 6, 9, 11, 16, 68]
    )
    assert isinstance(lh5_obj, lgdo.VectorOfVectors)
    assert n_rows == 7
    assert len(lh5_obj) == 7
    assert (lh5_obj.cumulative_length.nda == [1, 2, 3, 4, 5, 6, 7]).all()
    assert (lh5_obj.flattened_data.nda == [40, 60, 64, 60, 64, 28, 60]).all()


def test_read_array_concatenation(lgnd_file):
    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/geds/raw/baseline", [lgnd_file, lgnd_file])
    assert isinstance(lh5_obj, lgdo.Array)
    assert n_rows == 200
    assert len(lh5_obj) == 200


def test_read_lgnd_waveform_table(lgnd_file):
    store = LH5Store()

    lh5_obj, n_rows = store.read_object("/geds/raw/waveform", lgnd_file)
    assert isinstance(lh5_obj, lgdo.WaveformTable)

    lh5_obj, n_rows = store.read_object(
        "/geds/raw/waveform",
        lgnd_file,
        start_row=10,
        n_rows=10,
        field_mask=["t0", "dt"],
    )

    assert isinstance(lh5_obj, lgdo.Table)
    assert list(lh5_obj.keys()) == ["t0", "dt"]
    assert len(lh5_obj) == 10


def test_read_lgnd_waveform_table_fancy_idx(lgnd_file):
    store = LH5Store()

    lh5_obj, n_rows = store.read_object(
        "/geds/raw/waveform",
        lgnd_file,
        idx=[7, 9, 25, 27, 33, 38, 46, 52, 57, 59, 67, 71, 72, 82, 90, 92, 93, 94, 97],
    )
    assert isinstance(lh5_obj, lgdo.WaveformTable)
    assert len(lh5_obj) == 19


@pytest.fixture(scope="module")
def enc_lgnd_file(lgnd_file):
    store = LH5Store()
    wft, n_rows = store.read_object("/geds/raw/waveform", lgnd_file)
    wft.values.attrs["compression"] = RadwareSigcompress(codec_shift=-32768)
    store.write_object(
        wft,
        "/geds/raw/waveform",
        "/tmp/tmp-pygama-compressed-wfs.lh5",
        wo_mode="overwrite_file",
    )
    return "/tmp/tmp-pygama-compressed-wfs.lh5"


def test_write_compressed_lgnd_waveform_table(enc_lgnd_file):
    pass


def test_read_compressed_lgnd_waveform_table(lgnd_file, enc_lgnd_file):
    store = LH5Store()
    wft, _ = store.read_object("/geds/raw/waveform", enc_lgnd_file)
    assert isinstance(wft.values, lgdo.ArrayOfEqualSizedArrays)
    assert "compression" not in wft.values.attrs


# First test that we can overwrite a table with the same name without deleting the original field
def test_write_object_overwrite_table_no_deletion(caplog):
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    if os.path.exists("/tmp/write_object_overwrite_test.lh5"):
        os.remove("/tmp/write_object_overwrite_test.lh5")

    tb1 = lh5.Table(col_dict={"dset1": lh5.Array(np.zeros(10))})
    tb2 = lh5.Table(
        col_dict={"dset1": lh5.Array(np.ones(10))}
    )  # Same field name, different values
    store = LH5Store()
    store.write_object(tb1, "my_group", "/tmp/write_object_overwrite_test.lh5")
    store.write_object(
        tb2, "my_group", "/tmp/write_object_overwrite_test.lh5", wo_mode="overwrite"
    )  # Now, try to overwrite the same field

    # If the old field is deleted from the file before writing the new field, then we would get an extra debug statement
    assert "dset1 is not present in new table, deleting field" not in [
        rec.message for rec in caplog.records
    ]

    # Now, check that the data were overwritten
    tb_dat, _ = store.read_object("my_group", "/tmp/write_object_overwrite_test.lh5")
    assert np.array_equal(tb_dat["dset1"].nda, np.ones(10))


# Second: test that when we overwrite a table with fields with a different name, we delete the original field
def test_write_object_overwrite_table_with_deletion(caplog):
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    if os.path.exists("/tmp/write_object_overwrite_test.lh5"):
        os.remove("/tmp/write_object_overwrite_test.lh5")

    tb1 = lh5.Table(col_dict={"dset1": lh5.Array(np.zeros(10))})
    tb2 = lh5.Table(
        col_dict={"dset2": lh5.Array(np.ones(10))}
    )  # Same field name, different values
    store = LH5Store()
    store.write_object(tb1, "my_group", "/tmp/write_object_overwrite_test.lh5")
    store.write_object(
        tb2, "my_group", "/tmp/write_object_overwrite_test.lh5", wo_mode="overwrite"
    )  # Now, try to overwrite with a different field

    # Now, check that the data were overwritten
    tb_dat, _ = store.read_object("my_group", "/tmp/write_object_overwrite_test.lh5")
    assert np.array_equal(tb_dat["dset2"].nda, np.ones(10))

    # Also make sure that the first table's fields aren't lurking around the lh5 file!
    with h5py.File("/tmp/write_object_overwrite_test.lh5", "r") as lh5file:
        assert "dset1" not in list(lh5file["my_group"].keys())

    # Make sure the same behavior happens when we nest the table in a group
    if os.path.exists("/tmp/write_object_overwrite_test.lh5"):
        os.remove("/tmp/write_object_overwrite_test.lh5")

    tb1 = lh5.Table(col_dict={"dset1": lh5.Array(np.zeros(10))})
    tb2 = lh5.Table(
        col_dict={"dset2": lh5.Array(np.ones(10))}
    )  # Same field name, different values
    store = LH5Store()
    store.write_object(
        tb1, "my_table", "/tmp/write_object_overwrite_test.lh5", group="my_group"
    )
    store.write_object(
        tb2,
        "my_table",
        "/tmp/write_object_overwrite_test.lh5",
        group="my_group",
        wo_mode="overwrite",
    )  # Now, try to overwrite with a different field

    # Now, check that the data were overwritten
    tb_dat, _ = store.read_object(
        "my_group/my_table", "/tmp/write_object_overwrite_test.lh5"
    )
    assert np.array_equal(tb_dat["dset2"].nda, np.ones(10))

    # Also make sure that the first table's fields aren't lurking around the lh5 file!
    with h5py.File("/tmp/write_object_overwrite_test.lh5", "r") as lh5file:
        assert "dset1" not in list(lh5file["my_group/my_table"].keys())


# Third: test that when we overwrite other LGDO classes
def test_write_object_overwrite_lgdo(caplog):
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    # Start with an lgdo.WaveformTable
    if os.path.exists("/tmp/write_object_overwrite_test.lh5"):
        os.remove("/tmp/write_object_overwrite_test.lh5")

    tb1 = lh5.WaveformTable(
        t0=np.zeros(10),
        t0_units="ns",
        dt=np.zeros(10),
        dt_units="ns",
        values=np.zeros((10, 10)),
        values_units="ADC",
    )
    tb2 = lh5.WaveformTable(
        t0=np.ones(10),
        t0_units="ns",
        dt=np.ones(10),
        dt_units="ns",
        values=np.ones((10, 10)),
        values_units="ADC",
    )  # Same field name, different values
    store = LH5Store()
    store.write_object(
        tb1, "my_table", "/tmp/write_object_overwrite_test.lh5", group="my_group"
    )
    store.write_object(
        tb2,
        "my_table",
        "/tmp/write_object_overwrite_test.lh5",
        wo_mode="overwrite",
        group="my_group",
    )

    # If the old field is deleted from the file before writing the new field, then we would get a debug statement
    assert "my_table is not present in new table, deleting field" not in [
        rec.message for rec in caplog.records
    ]

    # Now, check that the data were overwritten
    tb_dat, _ = store.read_object(
        "my_group/my_table", "/tmp/write_object_overwrite_test.lh5"
    )
    assert np.array_equal(tb_dat["values"].nda, np.ones((10, 10)))

    # Now try overwriting an array, and test the write_start argument
    array1 = lh5.Array(nda=np.zeros(10))
    array2 = lh5.Array(nda=np.ones(20))
    store = LH5Store()
    store.write_object(array1, "my_array", "/tmp/write_object_overwrite_test.lh5")
    store.write_object(
        array2,
        "my_array",
        "/tmp/write_object_overwrite_test.lh5",
        wo_mode="overwrite",
        write_start=5,
    )

    # Now, check that the data were overwritten
    array_dat, _ = store.read_object("my_array", "/tmp/write_object_overwrite_test.lh5")
    expected_out_array = np.append(np.zeros(5), np.ones(20))

    assert np.array_equal(array_dat.nda, expected_out_array)

    # Now try overwriting a scalar
    scalar1 = lh5.Scalar(0)
    scalar2 = lh5.Scalar(1)
    store = LH5Store()
    store.write_object(scalar1, "my_scalar", "/tmp/write_object_overwrite_test.lh5")
    store.write_object(
        scalar2,
        "my_scalar",
        "/tmp/write_object_overwrite_test.lh5",
        wo_mode="overwrite",
    )

    # Now, check that the data were overwritten
    scalar_dat, _ = store.read_object(
        "my_scalar", "/tmp/write_object_overwrite_test.lh5"
    )

    assert scalar_dat.value == 1

    # Finally, try overwriting a vector of vectors
    vov1 = lh5.VectorOfVectors(listoflists=[np.zeros(1), np.ones(2), np.zeros(3)])
    vov2 = lh5.VectorOfVectors(listoflists=[np.ones(1), np.zeros(2), np.ones(3)])
    store = LH5Store()
    store.write_object(vov1, "my_vector", "/tmp/write_object_overwrite_test.lh5")
    store.write_object(
        vov2,
        "my_vector",
        "/tmp/write_object_overwrite_test.lh5",
        wo_mode="overwrite",
        write_start=1,
    )  # start overwriting the second list of lists

    vector_dat, _ = store.read_object(
        "my_vector", "/tmp/write_object_overwrite_test.lh5"
    )

    assert np.array_equal(vector_dat.cumulative_length.nda, [1, 2, 4, 7])
    assert np.array_equal(vector_dat.flattened_data.nda, [0, 1, 0, 0, 1, 1, 1])


# Test that when we try to overwrite an existing column in a table we fail
def test_write_object_append_column():
    # Try to append an array to a table
    if os.path.exists("/tmp/write_object_append_column_test.lh5"):
        os.remove("/tmp/write_object_append_column_test.lh5")

    array1 = lh5.Array(np.zeros(10))
    tb1 = lh5.Table(col_dict={"dset1`": lh5.Array(np.ones(10))})
    store = LH5Store()
    store.write_object(array1, "my_table", "/tmp/write_object_append_column_test.lh5")
    with pytest.raises(RuntimeError) as exc_info:
        store.write_object(
            tb1,
            "my_table",
            "/tmp/write_object_append_column_test.lh5",
            wo_mode="append_column",
        )  # Now, try to append a column to an array

    assert exc_info.type is RuntimeError
    assert (
        exc_info.value.args[0] == "Trying to append columns to an object of type array"
    )

    # Try to append a table that has a same key as the old table
    if os.path.exists("/tmp/write_object_append_column_test.lh5"):
        os.remove("/tmp/write_object_append_column_test.lh5")

    tb1 = lh5.Table(
        col_dict={"dset1": lh5.Array(np.zeros(10)), "dset2": lh5.Array(np.zeros(10))}
    )
    tb2 = lh5.Table(
        col_dict={"dset2": lh5.Array(np.ones(10))}
    )  # Same field name, different values
    store = LH5Store()
    store.write_object(tb1, "my_table", "/tmp/write_object_append_column_test.lh5")
    with pytest.raises(ValueError) as exc_info:
        store.write_object(
            tb2,
            "my_table",
            "/tmp/write_object_append_column_test.lh5",
            wo_mode="append_column",
        )  # Now, try to append a column with a same field

    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0]
        == "Can't append ['dset2'] column(s) to a table with the same field(s)"
    )

    # try appending a column that is larger than one that exists
    if os.path.exists("/tmp/write_object_append_column_test.lh5"):
        os.remove("/tmp/write_object_append_column_test.lh5")

    tb1 = lh5.Table(col_dict={"dset1": lh5.Array(np.zeros(10))})
    tb2 = lh5.Table(
        col_dict={"dset2": lh5.Array(np.ones(20))}
    )  # different field name, different size
    store = LH5Store()
    store.write_object(tb1, "my_table", "/tmp/write_object_append_column_test.lh5")
    with pytest.raises(ValueError) as exc_info:
        store.write_object(
            tb2,
            "my_table",
            "/tmp/write_object_append_column_test.lh5",
            wo_mode="append_column",
        )  # Now, try to append a column with a different field size

    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0]
        == "Table sizes don't match. Trying to append column of size 20 to a table of size 10."
    )

    # Finally successfully append a column
    if os.path.exists("/tmp/write_object_append_column_test.lh5"):
        os.remove("/tmp/write_object_append_column_test.lh5")

    tb1 = lh5.Table(col_dict={"dset1": lh5.Array(np.zeros(10))})
    tb2 = lh5.Table(
        col_dict={"dset2": lh5.Array(np.ones(10))}
    )  # different field name, different size
    store = LH5Store()
    store.write_object(
        tb1, "my_table", "/tmp/write_object_append_column_test.lh5", group="my_group"
    )
    store.write_object(
        tb2,
        "my_table",
        "/tmp/write_object_append_column_test.lh5",
        group="my_group",
        wo_mode="append_column",
    )

    # Now, check that the data were appended
    tb_dat, _ = store.read_object(
        "my_group/my_table", "/tmp/write_object_append_column_test.lh5"
    )
    assert isinstance(tb_dat, lgdo.table.Table)
    assert np.array_equal(tb_dat["dset1"].nda, np.zeros(10))
    assert np.array_equal(tb_dat["dset2"].nda, np.ones(10))
