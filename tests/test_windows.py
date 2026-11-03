"""Checks for the minute-by-band window shape."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.windows import windows_from_frame

ROOT = Path(__file__).resolve().parents[1]
CENTRES = [
    "31.6", "39.8", "50.1", "63.1", "79.4", "100", "125.9", "158.5",
    "199.5", "251.2", "316.2", "398.1", "501.2", "631", "794.3", "1000",
    "1258.9", "1584.9", "1995.3", "2511.9", "3162.3", "3981.1", "5011.9",
    "6309.6", "7943.3",
]


def _column_names():
    names = []
    for minute in range(1, 16):
        for centre in CENTRES:
            names.append(f"m{minute:02d}_TOL_{centre}")
    return names


def test_shape_and_minute_band_index():
    columns = _column_names()
    frame = pd.DataFrame([[0.0] * 375, [1.0] * 375], columns=columns)
    frame.insert(0, "ID", ["a", "b"])
    frame.loc[0, "m01_TOL_31.6"] = 11.0
    frame.loc[0, "m15_TOL_7943.3"] = 22.0
    frame.loc[1, "m02_TOL_100"] = np.nan
    windows = windows_from_frame(frame)
    assert windows.shape == (2, 15, 25)
    assert windows[0, 0, 0] == 11.0
    assert windows[0, 14, 24] == 22.0
    assert np.isnan(windows[1, 1, 5])


def test_column_order_does_not_matter():
    columns = _column_names()
    frame = pd.DataFrame([[float(i) for i in range(375)]], columns=columns)
    shuffled = frame[list(reversed(frame.columns))]
    assert np.allclose(windows_from_frame(frame), windows_from_frame(shuffled))


def test_rejects_a_short_table():
    frame = pd.DataFrame([[1.0, 2.0]], columns=["m01_TOL_31.6", "m01_TOL_39.8"])
    try:
        windows_from_frame(frame)
    except ValueError:
        return
    raise AssertionError("a short table should fail the 375-column check")


def test_first_test_rows_match_the_file():
    sample = pd.read_csv(ROOT / "Test.csv", nrows=2)
    windows = windows_from_frame(sample)
    assert windows.shape == (2, 15, 25)
    assert windows[0, 0, 0] == sample.loc[0, "m01_TOL_31.6"]
    assert windows[0, 14, 24] == sample.loc[0, "m15_TOL_7943.3"]


if __name__ == "__main__":
    test_shape_and_minute_band_index()
    test_column_order_does_not_matter()
    test_rejects_a_short_table()
    test_first_test_rows_match_the_file()
    print("window checks passed")
