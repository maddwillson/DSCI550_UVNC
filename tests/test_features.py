"""Checks for window features."""

import sys
from pathlib import Path
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.features import window_features



def make_test_window():
    window = np.full((1,15,25), 60.0)
    window[0,7,:] = 70.0 # all bands of min 8

    return window


def test_peak_minus_median():
    result = window_features(make_test_window())
    
    #check shape
    assert result.shape[0] == 1, "expected one row per window"
    assert np.allclose(result[:, 0:25], 10.0), "peak minus median should be 10"



def test_nan_handling():
    window = make_test_window()
    window[0,11,:] = np.nan # NaN all bands of min 10

    result = window_features(window)
    
    assert np.allclose(result[:, 0:25], 10.0), "missing values should be skipped"


def test_broadband_and_nan_cols():
    window = make_test_window()
    window[0, 2, :] = np.nan
    
    result = window_features(window)

    assert result.shape == (1, 78), "expected 78 feature columns"
    assert np.isclose(result[0, 75], 10.0), "broadband peak minus median should be 10"
    assert result[0, 76] == 8, "ship should peak at minute 8"
    assert result[0, 77] == 1, "one minute was blank"





if __name__ == "__main__":
    #call each test
    test_peak_minus_median()
    test_nan_handling()
    test_broadband_and_nan_cols()
    
    print("features checks were passed")
