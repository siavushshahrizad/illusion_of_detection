import pytest
import pandas as pd
import numpy as np
from processing import (
    compute_mean_and_std, 
    normalise_df,
    compress_left, 
    compress_right,
    compute_class_weights
)


TOL = 1e-3

class TestClass:
    @pytest.fixture
    def setup_test_df(self):
        person1 =  np.array([[1, 2], [1, 2], [3, 4]])
        person2 =  np.array([[4, 3], [1, 2], [0, 4]])
        df = pd.DataFrame({"mfcc": [person1, person2]})
        return df
       
    def test_compute_mean_and_std(self, setup_test_df):
        df = setup_test_df
        mean, std = compute_mean_and_std(df)
        assert len(mean) == len(df["mfcc"].iloc[0][0])

        assert mean[0] == 5 / 3
        assert mean[1] == 17 / 6

        assert len(mean) == len(std)
        assert std[0] == pytest.approx(1.506, abs=TOL) 
        assert std[1] == pytest.approx(0.983, abs=TOL) 

    def test_normalise(self, setup_test_df):
        df = setup_test_df
        mean, std = compute_mean_and_std(df)
        normalise_df(df, mean, std)

        # Checking first respondent
        assert "mfcc" in df.columns
        expected = np.array([
            [-0.443, -0.848],
            [-0.443, -0.848],
            [ 0.886,  1.187]
        ])
        result = df["mfcc"].iloc[0]
        assert np.allclose(expected, result, atol=TOL)

    def test_compress_left(self):
        arr = np.array([
            [2, 5],
            [6, 15],
            [2, 2],
            [5, 15],
        ])
        result = compress_left(arr)
        expected = np.array([4.0, 10.0])
        np.testing.assert_array_equal(result, expected)

    def test_compress_right(self):
        arr = np.array([
            [2, 5],
            [6, 15],
            [2, 3],
            [5, 15],
        ])
        result = compress_right(arr)
        expected = np.array([3.5, 9.0])
        np.testing.assert_array_equal(result, expected)

    def test_compute_class_weights(self):
        outcome = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        df = pd.DataFrame({"outcome": outcome})
        weights = compute_class_weights(df, "outcome")
        expected = [.625, 2.5]
        assert np.array_equal(weights, expected)
