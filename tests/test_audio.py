"""Checks for minute-by-band power on a synthetic tone."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.audio import minute_band_power
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


if __name__ == "__main__":
    test_one_minute_tone_peaks_in_its_band()
    test_a_partial_minute_is_dropped()
    test_a_short_recording_is_rejected()
    print("audio checks passed")
