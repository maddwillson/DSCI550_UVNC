"""Checks for decibel and linear-power conversion."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.units import db_to_linear, linear_to_db


def test_missing_levels_stay_missing():
    converted = db_to_linear([np.nan, 10.0])
    assert np.isnan(converted[0])
    assert converted[1] == np.power(10.0, 1.0)


def test_round_trip_keeps_finite_levels():
    levels = np.array([0.0, 10.0, -20.0, 63.5])
    restored = linear_to_db(db_to_linear(levels))
    assert np.allclose(restored, levels)


def test_equal_powers_add_to_about_three_decibels():
    # Two equal linear powers are +10*log10(2) dB, about 3.01 dB, above either one.
    combined = linear_to_db(db_to_linear(10.0) + db_to_linear(10.0))
    assert abs(combined - (10.0 + 10.0 * np.log10(2.0))) < 1e-9


def test_zero_and_negative_power():
    converted = linear_to_db([0.0, -1.0, np.nan])
    assert np.isneginf(converted[0])
    assert np.isnan(converted[1])
    assert np.isnan(converted[2])


if __name__ == "__main__":
    test_missing_levels_stay_missing()
    test_round_trip_keeps_finite_levels()
    test_equal_powers_add_to_about_three_decibels()
    test_zero_and_negative_power()
    print("units checks passed")
