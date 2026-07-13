from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pygama.flow.utils import parse_query_paths, to_datetime, to_unixtime


def test_key_to_datetime():
    assert to_datetime("20220716T105236Z") == datetime(
        2022, 7, 16, 10, 52, 36, tzinfo=timezone.utc
    )


def test_key_to_unixtime():
    assert to_unixtime("20220716T105236Z") == 1657968756


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
