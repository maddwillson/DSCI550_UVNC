"""Checks for the passage envelope and the fallback ship spectrum."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.passage import passage_envelope, ship_spectrum, third_octave_centres


def test_centres_match_the_starter_grid():
    centres = third_octave_centres()
    expected = 10.0 ** (1.5 + 0.1 * np.arange(25))
    assert centres.shape == (25,)
    assert np.allclose(centres, expected)
    assert np.all(np.diff(centres) > 0)


def test_envelope_peaks_at_closest_approach_and_sums_to_one():
    envelope = passage_envelope(15, 4)
    assert envelope.shape == (15,)
    assert abs(envelope.sum() - 1.0) < 1e-12
    assert np.argmax(envelope) == 3
    assert np.all(envelope > 0)
    centred = passage_envelope(15, 8)
    assert np.allclose(centred, centred[::-1])


def test_envelope_rejects_a_minute_outside_the_window():
    try:
        passage_envelope(15, 16)
    except ValueError:
        return
    raise AssertionError("closest approach must sit inside the window")


def test_ship_spectrum_is_smooth_and_louder_at_low_frequency():
    spectrum = ship_spectrum()
    assert spectrum.shape == (25,)
    assert abs(spectrum.sum() - 1.0) < 1e-12
    assert np.all(spectrum > 0)
    assert np.all(np.diff(spectrum) < 0)


if __name__ == "__main__":
    test_centres_match_the_starter_grid()
    test_envelope_peaks_at_closest_approach_and_sums_to_one()
    test_envelope_rejects_a_minute_outside_the_window()
    test_ship_spectrum_is_smooth_and_louder_at_low_frequency()
    print("passage checks passed")
