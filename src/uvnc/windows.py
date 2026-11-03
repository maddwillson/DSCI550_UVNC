"""Reshape test-style rows into minute-by-band windows.

Owner: Joshua Lee.
A row in ``Test.csv`` is 15 minute positions by 25 third-octave bands.
This module restores that matrix so later steps can index minutes and bands.
"""

import re

import numpy as np
import pandas as pd

_BAND_COLUMN = re.compile(r"^m(\d{2})_TOL_([0-9.]+)$")
_N_MINUTES = 15
_N_BANDS = 25


def windows_from_frame(frame):
    """Reshape test-style rows into minute-by-band windows.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per 15-minute window. Acoustic columns are named
        ``m01_TOL_31.6`` through ``m15_TOL_7943.3``. An ``ID`` column is
        ignored. Blank cells stay NaN.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_windows, 15, 25)``, dtype float64. Axis 1 is minute
        position 1 through 15. Axis 2 runs from the low band to the high band.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Pass a pandas DataFrame of test-style rows.")
    ordered = _ordered_feature_columns(frame.columns)
    block = frame.loc[:, ordered].apply(pd.to_numeric, errors="coerce")
    values = block.to_numpy(dtype=np.float64)
    return values.reshape(-1, _N_MINUTES, _N_BANDS)


def _ordered_feature_columns(columns):
    """Return the 375 band columns, minute then rising centre frequency."""
    parsed = []
    for name in columns:
        match = _BAND_COLUMN.fullmatch(str(name))
        if match is None:
            continue
        minute = int(match.group(1))
        centre = float(match.group(2))
        parsed.append((minute, centre, name))
    counts = {}
    for minute, _, _ in parsed:
        counts[minute] = counts.get(minute, 0) + 1
    expected = {minute: _N_BANDS for minute in range(1, _N_MINUTES + 1)}
    if counts != expected:
        raise ValueError(
            "Expected 15 minutes with 25 bands each "
            f"(375 acoustic columns); found {len(parsed)}."
        )
    parsed.sort()
    return [name for _, _, name in parsed]
