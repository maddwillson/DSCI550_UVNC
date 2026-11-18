"""Checks for window features."""

import sys
from pathlib import Path
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.features import window_features, FittedFeatures, BackgroundNeighbors



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
    window[0,9,:] = np.nan # NaN all bands of min 10

    result = window_features(window)
    
    assert np.allclose(result[:, 0:25], 10.0), "missing values should be skipped"


def test_broadband_and_nan_cols():
    window = make_test_window()
    window[0, 2, :] = np.nan
    
    result = window_features(window)

    assert result.shape == (1, 79), "expected 79 feature columns"
    assert np.isclose(result[0, 75], 10.0), "broadband peak minus median should be 10"
    assert result[0, 76] == 8, "ship should peak at minute 8"
    assert result[0, 77] == 1, "one minute was blank"

def test_above_floor():
    result = window_features(make_test_window())
    assert np.isclose(result[0, 78], .375), "expected the above floor share to be about 0.375"



def make_rand_windows(n=200, seed=0):
    """Random 60-80 dB windows from four fake sites."""
    rng = np.random.default_rng(seed)
    windows = rng.uniform(60, 80, size=(n, 15, 25))
    site_ids = rng.choice([1, 2, 3, 4], size=n)
    return windows, site_ids


def test_fitted_feats_shape():
    windows, site_ids = make_rand_windows()
    fitted = FittedFeatures(n_components=3, n_clusters=4).fit(windows, site_ids)
    result = fitted.transform(windows)
    assert result.shape == (200, 7), "3 PCA scores plus 4 cluster distances"
    assert len(fitted.feature_names_) == result.shape[1], "one name per column"


def test_fitted_feats_ignore():
    windows, site_ids = make_rand_windows()
    fitted = FittedFeatures(n_components=3, n_clusters=4).fit(windows, site_ids)
    assert np.allclose(fitted.transform(windows), fitted.transform(windows + 3)), "a loud recorder should not change the features"


def test_neighbors_flat():
    backgrounds = np.full((40, 15, 25), 60.0)
    site_ids = np.repeat([1, 2], 20)
    model = BackgroundNeighbors(n_neighbors=3).fit(backgrounds, site_ids)
    result = model.transform(make_test_window(), np.array([1]))
    assert result.shape == (1, 2), "expected and excess"
    assert np.allclose(result, [[0.0, 10.0]]), "flat backgrounds should be 0, so excess is the ship's 10"


def test_neighbors_other_site():
    backgrounds = np.full((40, 15, 25), 60.0)
    backgrounds[20:, 2, :] = 65.0
    site_ids = np.repeat([1, 2], 20)
    model = BackgroundNeighbors(n_neighbors=3).fit(backgrounds, site_ids)
    result = model.transform(make_test_window(), np.array([1]))
    assert np.isclose(result[0, 0], 5.0), "a site 1 window must use site 2 backgrounds"




if __name__ == "__main__":
    #call each test
    test_peak_minus_median()
    test_nan_handling()
    test_broadband_and_nan_cols()
    test_above_floor()

    test_fitted_feats_shape()
    test_fitted_feats_ignore()

    test_neighbors_flat()
    test_neighbors_other_site()

    print("features checks were passed")
