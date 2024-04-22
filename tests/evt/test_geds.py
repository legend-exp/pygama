import awkward as ak
from pygama.evt.modules import geds


import pytest
import numpy as np


def test_manipulate_ctx_matrix():

    test_matrix={
        "V02160A": {"V02160B": 0.1},
        "V02160B": {"V02160A": 0.1}
    }    
    test_positive={
        "V02160A": {"V02160B": 0.3},
        "V02160B": {"V02160A": 0.3}
    }    
    test_small={
        "V02160A": {"V02160B": 0.01},
        "V02160B": {"V02160A": 0.01}
    }    

    test_bad_channel={
        "AAAAAAA": {"BBBBBBB": 0.1},
        "BBBBBBB": {"AAAAAAA": 0.1}
    }   
    matrix_rawids = geds.manipulate_ctx_matrix(test_matrix,None,True)

    ## check names are converted to rawid ok
    print(matrix_rawids)
    
    assert matrix_rawids["ch1104000"]["ch1104001"]==test_matrix["V02160A"]["V02160B"]

    matrix_comb = geds.manipulate_ctx_matrix(test_matrix,test_positive,True)
    assert matrix_comb["ch1104000"]["ch1104001"]==-0.3

    matrix_comb_small = geds.manipulate_ctx_matrix(test_matrix,test_small,True)
    assert matrix_comb_small["ch1104000"]["ch1104001"]==0.1

    ## check if the matrix contains some non-existing channel the right exception is raised
    with pytest.raises(ValueError):
        geds.manipulate_ctx_matrix(test_bad_channel,None,True)

