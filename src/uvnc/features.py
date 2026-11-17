"""Band statistics, PCA, k-means, and the background estimate.

Owner: Madeleine Willson.
Input: training windows of shape ``(n, 15, 25)`` from ``windows_from_frame``.
Output: one feature row per window. Fit PCA, k-means, and scalers on
training windows only.
"""

import numpy as np

from uvnc.cleaning import BAND_CENTRES
from uvnc.units import db_to_linear, linear_to_db


FEATURE_NAMES = (
    [f"peak_minus_median_TOL_{c}" for c in BAND_CENTRES]
    + [f"range_TOL_{c}" for c in BAND_CENTRES]
    + [f"spread_TOL_{c}" for c in BAND_CENTRES]
    + ["broadband_peak_minus_median", "broadband_peak_minute", "missing_minutes"]
)



def _peak_minus_median(windows):
    """Loudest minute/band minus the median minute/band, per band/window."""

    return np.nanmax(windows, axis=1) - np.nanmedian(windows, axis=1)
  


def _band_range(windows):
    """Loudest minute minus the quietest minute, per band. 
    Returns ``(n, 25)`` in dB."""
    
    return np.nanmax(windows, axis=1) - np.nanmin(windows, axis=1)


def _band_spread(windows):
    """Standard deviation of minutes, per band.
    Returns ``(n, 25)`` in dB.
    """
    return  np.nanstd(windows, axis=1)  



def _broadband_db(windows):
    """Total level of each minute across all 25 bands.
    Bands are added in linear power. A missing minute stays NaN.
    Returns ``(n, 15)`` in dB.
    """
    linear = db_to_linear(windows)
    total = np.sum(linear, axis= 2)

    return linear_to_db(total)


def _missing_minutes(windows):
    """Number of whole minutes that are blank.
    Returns ``(n,)``.
    """
    return np.isnan(windows).all(axis=2).sum(axis=1)







def window_features(windows):
    """Turn minute-by-band windows into one feature row each.

    Parameters
    ----------
    windows : numpy.ndarray
        Shape ``(n, 15, 25)``. NaN marks a missing cell.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n_features)``.
    """


    broadband = _broadband_db(windows)
    broadband_peak = _peak_minus_median(broadband)

    peak_minute = np.nanargmax(broadband, axis=1) + 1 
    
    return np.column_stack([
        _peak_minus_median(windows),
        _band_range(windows),
        _band_spread(windows),
        broadband_peak,
        peak_minute,
        _missing_minutes(windows),
    ])