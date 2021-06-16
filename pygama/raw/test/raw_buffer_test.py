import pytest, json
import pygama.raw.raw_buffer as prb
import numpy as np
from numpy.testing import assert_


rb_json = '''
{
  "FlashCamEventDecoder" : {
    "g{key:0>3d}" : {
      "key_list" : [ [24,64] ],
      "out_stream" : "$DATADIR/{file_key}_geds",
      "out_name" : "geds/{name}"
    },
    "spms" : {
      "key_list" : [ [6,23] ],
      "out_stream" : "$DATADIR/{file_key}_spms",
      "out_name" : "spms/{name}"
    },
    "puls" : {
      "key_list" : [ 0 ],
      "out_stream" : "$DATADIR/{file_key}_auxs",
      "out_name" : "auxs/{name}"
    },
    "muvt" : {
      "key_list" : [ 1, 5 ],
      "out_stream" : "$DATADIR/{file_key}_auxs",
      "out_name" : "auxs/{name}"
    }
  }
}
'''

def test_rb_json_load():
    json_dict = json.loads(rb_json)
    kw_dict = { 'file_key' : 'run0' } 
    rblib = prb.RawBufferLibrary(json_dict=json_dict, kw_dict=kw_dict)
    rb_keyed = rblib['FlashCamEventDecoder'].get_keyed_dict()
    name = rb_keyed[41].out_name
    assert_(name == 'geds/g041', f'got {name}')


def test_rb_init():
    rblist = prb.RawBufferList()
    rblist.append(prb.RawBuffer())
    rblist.append(prb.RawBuffer())
    rblist.init_lgdos(size=888)
    length = len(rblist[0].lgdo) 
    assert_(length == 888, f'got {length}')
