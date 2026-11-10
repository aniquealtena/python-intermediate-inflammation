"""Tests for statistics functions within the Model layer."""
import pytest
import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean, daily_min, daily_max


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [0, 3], [3, 5], [-2, 8] ], [-2, 3]),
    ])
def test_daily_min(test, expected):
    """Test min function works for array of zeroes and positive and negative integers."""
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [-5, 3], [-3, 5], [-2, 8] ], [-2, 8]),
    ])
def test_daily_max(test, expected):
    """Test max function works for array of zeroes and positive amd negative integers."""
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])