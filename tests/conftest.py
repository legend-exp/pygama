import pytest
from legend_testdata import LegendTestData


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout('3581957')
    return ldata
