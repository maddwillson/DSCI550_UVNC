"""Checks for minute-by-band power on a synthetic tone."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.audio import _pattern_around_loudest, minute_band_power, passage_pattern
from uvnc.passage import third_octave_centres


def _tone(frequency_hz, sample_rate_hz, n_minutes):
    n_samples = n_minutes * sample_rate_hz * 60
    time = np.arange(n_samples) / sample_rate_hz
    return np.sin(2 * np.pi * frequency_hz * time)


def test_one_minute_tone_peaks_in_its_band():
    sample_rate = 8000
    power = minute_band_power(_tone(100.0, sample_rate, 1), sample_rate)
    assert power.shape == (1, 25)
    finite = np.isfinite(power[0])
    assert finite.any()
    assert np.all(power[0, finite] >= 0)
    centres = third_octave_centres()
    peak = int(np.nanargmax(power[0]))
    assert centres[peak] == 100.0
    assert np.isnan(power[0, -1])


def test_a_partial_minute_is_dropped():
    sample_rate = 8000
    samples = _tone(100.0, sample_rate, 1)
    extra = np.zeros(sample_rate * 10)
    power = minute_band_power(np.concatenate([samples, extra]), sample_rate)
    assert power.shape == (1, 25)


def test_a_short_recording_is_rejected():
    try:
        minute_band_power(np.zeros(8000), 8000)
    except ValueError:
        return
    raise AssertionError("a recording under one minute should be rejected")


def _rising_passage(sample_rate_hz):
    """Fifteen minutes of a 100 Hz tone, loudest at minute 8."""
    chunks = []
    for minute in range(1, 16):
        gain = np.exp(-0.5 * ((minute - 8) / 2.0) ** 2)
        time = np.arange(sample_rate_hz * 60) / sample_rate_hz
        chunks.append(gain * np.sin(2 * np.pi * 100.0 * time))
    return np.concatenate(chunks)


def test_passage_pattern_centers_on_the_loudest_minute():
    sample_rate = 2000
    pattern, closest_minute = passage_pattern(_rising_passage(sample_rate), sample_rate)
    assert pattern.shape == (15, 25)
    assert closest_minute == 8
    finite = np.isfinite(pattern)
    assert abs(pattern[finite].sum() - 1.0) < 1e-12
    assert np.all(pattern[finite] >= 0)
    assert np.isnan(pattern[:, -1]).all()
    centres = third_octave_centres()
    assert centres[int(np.nanargmax(pattern[closest_minute - 1]))] == 100.0
    minute_energy = np.nansum(pattern, axis=1)
    assert int(np.nanargmax(minute_energy)) == closest_minute - 1


def test_passage_window_slides_at_the_edges():
    power = np.zeros((20, 25))
    power[0, 0] = 5.0
    pattern, closest = _pattern_around_loudest(power)
    assert pattern.shape == (15, 25)
    assert closest == 1
    power[-1, 0] = 9.0
    _, closest = _pattern_around_loudest(power)
    assert closest == 15


def test_a_recording_under_15_minutes_is_rejected():
    try:
        passage_pattern(_tone(100.0, 2000, 1), 2000)
    except ValueError:
        return
    raise AssertionError("a passage needs 15 complete minutes")


if __name__ == "__main__":
    test_one_minute_tone_peaks_in_its_band()
    test_a_partial_minute_is_dropped()
    test_a_short_recording_is_rejected()
    test_passage_pattern_centers_on_the_loudest_minute()
    test_passage_window_slides_at_the_edges()
    test_a_recording_under_15_minutes_is_rejected()
    print("audio checks passed")
