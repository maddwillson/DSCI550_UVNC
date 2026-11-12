"""Checks that labeled windows keep each site id on its own row."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_labeled_windows import label_windows
from uvnc.cleaning import WINDOW_COLUMNS
from uvnc.pipeline import build_labeled_windows
from uvnc.windows import windows_from_frame


def _frame(levels, site_ids, index=None):
    rows = [[level] * len(WINDOW_COLUMNS) for level in levels]
    frame = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    frame.insert(0, "site_id", site_ids)
    if index is not None:
        frame.index = index
    return frame


def test_site_ids_follow_row_order():
    frame = _frame([60.0, 90.0, 70.0], [4, 13, 4], index=[10, 2, 7])
    mixtures, shares, site_ids = label_windows(
        frame, closest_minute=8, added_energy=1e10, seed=0, n_drop=2
    )
    expected_mixtures, expected_shares, _metadata = build_labeled_windows(
        windows_from_frame(frame),
        closest_minutes=8,
        added_energies=1e10,
        seeds=0,
        n_drops=2,
    )
    assert mixtures.shape == (3, 15, 25)
    assert list(site_ids) == [4, 13, 4]
    assert np.allclose(mixtures, expected_mixtures, equal_nan=True)
    assert np.allclose(shares, expected_shares)
    assert shares[0] > shares[2] > shares[1]


def test_missing_site_id_is_rejected():
    frame = _frame([70.0], [1]).drop(columns=["site_id"])
    try:
        label_windows(frame)
    except ValueError:
        return
    raise AssertionError("a table without site_id should fail")


def test_empty_frame_is_rejected():
    frame = _frame([], [])
    try:
        label_windows(frame)
    except ValueError:
        return
    raise AssertionError("an empty table should fail")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("labeled-window checks passed")
