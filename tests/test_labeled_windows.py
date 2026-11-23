"""Checks that labeled windows keep each site id on its own row."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_labeled_windows import assign_window_settings, label_windows
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


def test_each_window_gets_its_own_settings():
    closest, energy, seeds, drops = assign_window_settings(40, settings_seed=0)
    assert closest.shape == energy.shape == seeds.shape == drops.shape == (40,)
    assert set(np.unique(closest)).issubset(set(range(1, 16)))
    assert len(np.unique(closest)) > 1
    assert np.all((energy >= 1e9) & (energy < 1e11))
    assert len(np.unique(energy)) > 1
    assert len(np.unique(seeds)) == 40
    assert set(np.unique(drops)) == {0, 1}

    frame = _frame(np.full(40, 70.0), np.arange(40))
    mixtures, shares, site_ids = label_windows(
        frame,
        closest_minute=closest,
        added_energy=energy,
        seed=seeds,
        n_drop=drops,
    )
    hidden = np.isnan(mixtures).all(axis=2).sum(axis=1)
    assert np.array_equal(hidden, drops)
    dropped_minutes = [tuple(np.flatnonzero(row) + 1) for row in np.isnan(mixtures).all(axis=2)]
    assert len(set(dropped_minutes)) > 2
    assert list(site_ids) == list(range(40))
    assert len(np.unique(np.round(shares, 6))) > 1


def test_a_fixed_setting_stays_fixed():
    closest, energy, seeds, drops = assign_window_settings(
        5, closest_minute=8, added_energy=1e10, seed=0, n_drop=2, settings_seed=3
    )
    assert np.array_equal(closest, np.full(5, 8))
    assert np.allclose(energy, np.full(5, 1e10))
    assert np.array_equal(seeds, np.zeros(5, dtype=np.int64))
    assert np.array_equal(drops, np.full(5, 2))


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
