from __future__ import annotations

import os

import pytest

from pygama.datatools.utils import _read_dataflow_config, parse_query_paths


def test_read_dataflow_config(test_refprod):
    # test with explicit path
    df_config1, tier_paths1, query_config1 = _read_dataflow_config(
        f"{test_refprod}/dataflow-config.yaml"
    )
    assert df_config1 is not None
    assert tier_paths1 is not None
    assert query_config1 is not None

    # test with environment variable
    assert os.environ["REFPROD"] == test_refprod
    df_config2, tier_paths2, query_config2 = _read_dataflow_config()
    assert df_config1 == df_config2
    assert tier_paths1 == tier_paths2
    assert query_config1 == query_config2

    # test with already-read dict
    df_config3, tier_paths3, query_config3 = _read_dataflow_config(df_config1)
    assert df_config1 == df_config3
    assert tier_paths1 == tier_paths3
    assert query_config1 == query_config3

    with pytest.raises(ValueError):
        _read_dataflow_config(5)


def test_parse_query_paths():
    # test an expression with a variety of patterns
    assert parse_query_paths(
        "abc + x@par.xyz[3:5] - {asdf} + np.add(123, x1) * @db/:var-y:var2[x] and \"abc@xyz\" or 'xyz:abc' "
    ) == [
        ("abc", None, "abc"),
        ("x@par.xyz", "x", "@par.xyz"),
        ("x1", None, "x1"),
        ("@db", None, "@db"),
        (":var", None, "var"),
        ("y:var2", "y", "var2"),
    ]

    # test fullmatch
    assert parse_query_paths("abc:def.ghi", fullmatch=True) == (
        "abc:def.ghi",
        "abc",
        "def.ghi",
    )

    # variable must not start with a digit
    with pytest.raises(NameError):
        parse_query_paths("1abc")
    # alias cannot have attributes
    with pytest.raises(NameError):
        parse_query_paths("ab.cd:ef")
    # alias cannot be reserved name
    with pytest.raises(NameError):
        parse_query_paths("and:nope")
    # alias cannot be reserved name
    with pytest.raises(NameError):
        parse_query_paths("and", fullmatch=True)
    # fullmatch variable cannot be number
    with pytest.raises(NameError):
        parse_query_paths("123", fullmatch=True)
    # cannot have multiple @ or : separators
    with pytest.raises(NameError):
        parse_query_paths("first:second@third")
    # full match can't have multiple variables
    with pytest.raises(NameError):
        parse_query_paths("abc + def", fullmatch=True)
