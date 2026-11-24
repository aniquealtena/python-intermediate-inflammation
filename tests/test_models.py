"""Tests for statistics functions within the Model layer."""

import os
import pytest
import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean
from unittest.mock import Mock
from inflammation.models import daily_mean, daily_min, daily_max, patient_normalise, CSVDataSource, analyse_data, compute_standard_deviation_by_day


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


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


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None,
        ),
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[1, 2, 3], [4, 5, float('nan')], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.8, 1, 0], [0.78, 0.89, 1]],
            None,
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        )
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""

    if expect_raises is not None:
        with pytest.raises(expect_raises):
            patient_normalise(np.array(test))
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
def test_analyse_data_mock_source():
    data_source = Mock()
    mock_data = [[[0, 2, 0]],
                [[0, 1, 0]]]
    data_source.load_inflammation_data.return_value = mock_data

    analyse_data(data_source)

def test_analyse_data():
    path = os.path.join(os.getcwd(), "./data/")
    data_source = CSVDataSource(path)
    result = analyse_data(data_source)
    expected_result = [0.,0.22510286,0.18157299,0.1264423,0.9495481,0.27118211,
                        0.25104719,0.22330897,0.89680503,0.21573875,1.24235548,0.63042094,
                        1.57511696,2.18850242,0.3729574,0.69395538,2.52365162,0.3179312,
                        1.22850657,1.63149639,2.45861227,1.55556052,2.8214853,0.92117578,
                        0.76176979,2.18346188,0.55368435,1.78441632,0.26549221,1.43938417,
                        0.78959769,0.64913879,1.16078544,0.42417995,0.36019114,0.80801707,
                        0.50323031,0.47574665,0.45197398,0.22070227]

    npt.assert_array_almost_equal(result, expected_result)


@pytest.mark.parametrize('data, expected_standard_deviation', [
    ([0, 0, 0], 0.0),
    ([1.0, 1.0, 1.0], 0),
    ([0.0, 2.0], 1.0),
    ([1.0, 2.0, 3.0], 0.81649658092773)
])
def test_daily_standard_deviation(data, expected_standard_deviation):
    from inflammation.models import compute_standard_deviation
    result_data = compute_standard_deviation(data)
    npt.assert_approx_equal(result_data, expected_standard_deviation)
    
    
@pytest.mark.parametrize(
    "data, expected_output",
	[
		([[ [0, 1, 0],
     [0, 2, 0] ]], 
     [0, 0, 0]), 
    ([ [ [0, 1, 0] ],
     [ [0, 2, 0] ] ], 
     [0, np.sqrt(0.25), 0])
	])
def test_compute_standard_deviation_by_day(data, expected_output):
    result = compute_standard_deviation_by_day(data)
    npt.assert_array_almost_equal(result, expected_output)
