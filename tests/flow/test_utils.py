from datetime import datetime, timezone

from pygama.flow.utils import to_datetime, to_unixtime


def test_key_to_datetime():
    assert to_datetime("20220716T105236Z") == datetime(
        2022, 7, 16, 10, 52, 36, tzinfo=timezone.utc
    )


def test_key_to_unixtime():
    assert to_unixtime("20220716T105236Z") == 1657968756
