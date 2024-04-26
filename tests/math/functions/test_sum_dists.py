import pytest
from scipy.stats import norm

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.sum_dists import SumDists


def test_only_two_distributions():
    with pytest.raises(ValueError) as exc_info:
        SumDists([(gaussian, [0, 1])], [0], "fracs")

    exception_raised = exc_info.value
    assert str(exception_raised) == "Can only sum two distributions at once."


def test_two_distributions_no_par_array():
    with pytest.raises(ValueError) as exc_info:
        SumDists([(gaussian, [0, 1]), (gaussian, [], [])], [0], "fracs")

    exception_raised = exc_info.value
    assert (
        str(exception_raised)
        == "Each tuple needs a distribution and a parameter index array."
    )


def test_pygama_continuous_check():
    with pytest.raises(ValueError) as exc_info:
        SumDists([(gaussian, [0, 1]), (norm, [1, 0])], [0], "fracs")

    exception_raised = exc_info.value
    assert str(exception_raised)[-12:] == "distribution"


def test_fracs_flag_with_wrong_args():
    with pytest.raises(ValueError) as exc_info:
        SumDists([(gaussian, [0, 1]), (gaussian, [1, 0])], [0, 2], "fracs")

    exception_raised = exc_info.value
    assert (
        str(exception_raised)
        == "SumDists only accepts the parameter position of one fraction."
    )


def test_areas_flag_with_wrong_args():
    with pytest.raises(ValueError) as exc_info:
        SumDists([(gaussian, [0, 1]), (gaussian, [1, 0])], [0], "areas")

    exception_raised = exc_info.value
    assert str(exception_raised) == "SumDists needs two parameter indices of areas."


def test_onea_area_flag_with_wrong_args():
    with pytest.raises(ValueError) as exc_info:
        SumDists([(gaussian, [0, 1]), (gaussian, [1, 0])], [0, 2], "one_area")

    exception_raised = exc_info.value
    assert str(exception_raised) == "SumDists needs one parameter index of an area."
