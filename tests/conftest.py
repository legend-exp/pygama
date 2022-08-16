import pytest
from legend_testdata import LegendTestData


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("9c248f1")
    return ldata
