"""Checks for site cleaning and 15-minute background windows."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.cleaning import (
    RAW_BANDS,
    WINDOW_COLUMNS,
    clean_background_windows,
    qc_row_flags,
    site_id_from_name,
)
from uvnc.windows import windows_from_frame


def _site(minutes, recorder="rec_A", start="2024-01-01 00:00:00", site="4 Bird Island"):
    rng = np.random.default_rng(0)
    times = pd.date_range(start, periods=minutes, freq="60s")
    levels = 60 + rng.normal(0, 5, size=(minutes, 25))
    frame = pd.DataFrame(levels, columns=RAW_BANDS)
    frame.insert(0, "dur", 60.0)
    frame.insert(0, "t", times)
    frame.insert(0, "recorder", recorder)
    frame.insert(0, "site", site)
    return frame


def test_site_id_from_name():
    assert site_id_from_name("13 Saldhana Bay") == 13
    assert site_id_from_name("3 Cape_Recife") == 3


def test_columns_match_test_layout_and_shape():
    out = clean_background_windows(_site(30))
    assert len(out) == 2
    assert list(out.columns[5:]) == WINDOW_COLUMNS
    assert out["site_id"].tolist() == [4, 4]
    assert windows_from_frame(out).shape == (2, 15, 25)


def test_minutes_are_in_time_order():
    frame = _site(15)
    out = clean_background_windows(frame.sample(frac=1, random_state=1))
    expected = frame[RAW_BANDS].to_numpy()
    assert np.allclose(windows_from_frame(out)[0], expected)


def test_gap_blocks_a_window():
    frame = _site(30)
    frame = frame.drop(index=14)  # a missing minute in the first window
    out = clean_background_windows(frame)
    assert len(out) == 1
    assert out["t_start"].iloc[0] == frame["t"].iloc[14 + 0]


def test_windows_never_cross_recorders():
    both = pd.concat([_site(10, "rec_A"), _site(10, "rec_B", "2024-01-01 00:10:00")])
    assert len(clean_background_windows(both)) == 0


def test_bad_rows_are_removed():
    frame = _site(15)
    frame.loc[3, "dur"] = 20.0          # short minute
    assert len(clean_background_windows(frame)) == 0
    frame = _site(15)
    frame.loc[5, RAW_BANDS] = 12.0      # flat spectrum
    flags = qc_row_flags(frame)
    assert not flags.loc[5, "spread_ok"]
    assert len(clean_background_windows(frame)) == 0


def test_frozen_repeat_is_removed():
    frame = _site(16)
    frame.loc[8, RAW_BANDS] = frame.loc[7, RAW_BANDS].to_numpy()
    assert not qc_row_flags(frame).loc[8, "not_frozen"]


def test_output_has_no_nan_and_unique_ids():
    out = clean_background_windows(_site(45))
    assert out["ID"].is_unique
    assert not out[WINDOW_COLUMNS].isna().any().any()


def test_stride_overlap():
    assert len(clean_background_windows(_site(20), stride=1)) == 6


def test_level_floor_alone_removes_a_row():
    frame = _site(15)
    # One band far below the 20 dB floor, but a wide spread, full minute,
    # finite values and no repeat: only the floor rule should fail.
    frame.loc[6, RAW_BANDS[0]] = 15.0
    flags = qc_row_flags(frame)
    assert not flags.loc[6, "level_ok"]
    assert flags.loc[6, ["duration_ok", "finite_25_bands", "spread_ok",
                         "not_frozen"]].all()
    assert not flags.loc[6, "row_ok"]
    assert flags.drop(index=6)["row_ok"].all()
    assert len(clean_background_windows(frame)) == 0


def test_frozen_minute_inside_good_run_breaks_the_window():
    frame = _site(15)
    # Minute 9 repeats minute 8 exactly: a frozen reading in an otherwise
    # good run. It is dropped, which leaves 14 minutes and a time gap.
    frame.loc[8, RAW_BANDS] = frame.loc[7, RAW_BANDS].to_numpy()
    flags = qc_row_flags(frame)
    assert not flags.loc[8, "not_frozen"]
    assert flags.drop(index=8)["row_ok"].all()
    assert len(clean_background_windows(frame)) == 0


def test_recorders_sharing_a_prefix_get_distinct_ids():
    name_a = "URN_8321_A.1.24000"
    name_b = "URN_8321_B.1.24000"
    assert name_a[:8] == name_b[:8]
    both = pd.concat([_site(15, name_a), _site(15, name_b)])  # repeated index
    both = both.sample(frac=1, random_state=2)                # shuffled
    out = clean_background_windows(both)
    assert len(out) == 2
    assert out["ID"].is_unique
    assert set(out["recorder"]) == {name_a, name_b}
    assert out["t_start"].nunique() == 1  # same start time, ID still differs


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_cleaning: all passed")