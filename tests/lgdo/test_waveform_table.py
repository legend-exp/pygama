import numpy as np

import pygama.lgdo as lgdo
from pygama.lgdo.waveform_table import WaveformTable


# TODO: add comparison operators for lgdo's?
def test_init():
    wft = WaveformTable()
    assert len(wft.t0) == 1024
    assert (wft.t0.nda == lgdo.Array(shape=(1024,), fill_val=0).nda).all()
    assert (wft.dt.nda == lgdo.Array(shape=(1024,), fill_val=1).nda).all()
    assert isinstance(wft.values, lgdo.VectorOfVectors)

    # TODO: add assertions and more initializers
    wft = WaveformTable(size=3, wf_len=1024)
    wft = WaveformTable(values=lgdo.ArrayOfEqualSizedArrays(shape=(3, 1024), fill_val=69))
    wft = WaveformTable(t0=[0, 0, 0], dt=[1, 1, 1], values=lgdo.ArrayOfEqualSizedArrays(shape=(3, 1024)))
    wft = WaveformTable(t0=[0, 0, 0], dt=[1, 1, 1], wf_len=1024)
    wft = WaveformTable(t0=[0, 0, 0], dt=[1, 1, 1], wf_len=1024, dtype=np.float32)
