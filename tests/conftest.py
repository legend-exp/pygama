import pytest
from legend_testdata import LegendTestData


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout('5de1d80')
    return ldata
