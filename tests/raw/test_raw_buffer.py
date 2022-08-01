import json

import pygama.raw.raw_buffer as prb


def test_raw_buffer_list():
    rbl = prb.RawBufferList()
    rbl.set_from_json_dict(
        {
            "g{key:0>3d}": {
                "key_list": [[3, 6]],
                "out_stream": "$DATADIR/{file_key}_geds.lh5",
            }
        },
        kw_dict={"file_key": "run0"},
    )
    assert len(rbl) == 4
    assert rbl[1].out_stream == "$DATADIR/run0_geds.lh5"
    assert rbl.get_list_of("out_name") == ["g003", "g004", "g005", "g006"]
    assert rbl.get_list_of("key_list") == [[3], [4], [5], [6]]

    rbl.clear()
    rbl.set_from_json_dict(
        {
            "spms": {
                "key_list": [[3, 6]],
                "out_stream": "$DATADIR/{file_key}_geds.lh5",
                "out_name": "testspms",
            }
        },
        kw_dict={"file_key": "run0"},
    )
    assert rbl.get_list_of("out_name") == ["testspms"]
    assert rbl.get_list_of("key_list") == [[3, 4, 5, 6]]


def test_raw_buffer_lib_json_load():

    rb_json = """
    {
      "FCEventDecoder" : {
        "g{key:0>3d}" : {
          "key_list" : [[24, 64]],
          "out_stream" : "$DATADIR/{file_key}_geds.lh5:/geds"
        },
        "spms" : {
          "key_list" : [[6, 23]],
          "out_stream" : "$DATADIR/{file_key}_spms.lh5"
        },
        "puls" : {
          "key_list" : [0],
          "out_stream" : "$DATADIR/{file_key}_auxs.lh5:/auxs"
        },
        "muvt" : {
          "key_list" : [1, 5],
          "out_stream" : "$DATADIR/{file_key}_auxs.lh5:/auxs"
        }
      },
      "*" : {
        "{name}" : {
          "key_list" : ["*"],
          "out_stream" : "$DATADIR/{file_key}_others.lh5"
        }
      }
    }
    """
    json_dict = json.loads(rb_json)
    rblib = prb.RawBufferLibrary(json_dict=json_dict, kw_dict={"file_key": "run0"})
    rb_keyed = rblib["FCEventDecoder"].get_keyed_dict()
    name = rb_keyed[41].out_name
    assert name == "g041"
