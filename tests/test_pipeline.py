"""Checks for the labeled-window entry point."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.mixing import mask_random_minutes, mix_passage, vessel_share
from uvnc.passage import passage_envelope, ship_spectrum
from uvnc.pipeline import build_labeled_window, build_labeled_windows


def test_entry_point_matches_the_separate_steps():
    background = np.zeros((15, 25))
    background[1, 0] = np.nan
    original = background.copy()
    mixture, share, metadata = build_labeled_window(
        background, closest_minute=6, added_energy=40.0, seed=7, n_drop=2
    )
    expected_mixture = mix_passage(
        background, passage_envelope(15, 6), ship_spectrum(), 40.0
    )
    expected_masked, dropped = mask_random_minutes(
        expected_mixture, np.random.default_rng(7), 2
    )
    expected_background = background.copy()
    expected_background[dropped - 1, :] = np.nan
    assert np.allclose(mixture, expected_masked, equal_nan=True)
    assert abs(share - vessel_share(expected_background, expected_masked)) < 1e-12
    assert metadata == {
        "seed": 7,
        "closest_minute": 6,
        "dropped_minutes": [int(minute) for minute in dropped],
    }
    assert np.array_equal(background, original, equal_nan=True)
    assert 0.0 <= share <= 1.0


def test_a_separable_pattern_matches_the_fallback():
    background = np.full((15, 25), 55.0)
    pattern = np.outer(passage_envelope(15, 8), ship_spectrum())
    fallback, fallback_share, fallback_meta = build_labeled_window(
        background, closest_minute=8, added_energy=25.0, seed=4, n_drop=1
    )
    mixed, mixed_share, mixed_meta = build_labeled_window(
        background,
        closest_minute=8,
        added_energy=25.0,
        seed=4,
        n_drop=1,
        pattern=pattern,
    )
    assert np.allclose(mixed, fallback, equal_nan=True)
    assert abs(mixed_share - fallback_share) < 1e-12
    assert mixed_meta == fallback_meta


def test_pattern_keeps_the_closest_minute_the_caller_passed():
    background = np.zeros((15, 25))
    pattern = np.zeros(background.shape)
    pattern[0, 0] = 1.0
    _, share, metadata = build_labeled_window(
        background,
        closest_minute=8,
        added_energy=20.0,
        seed=1,
        n_drop=0,
        pattern=pattern,
    )
    assert metadata["closest_minute"] == 8
    assert metadata["dropped_minutes"] == []
    assert 0.0 <= share <= 1.0


def test_same_seed_blanks_the_same_minutes():
    background = np.full((15, 25), 30.0)
    first, share_first, meta_first = build_labeled_window(
        background, closest_minute=8, added_energy=10.0, seed=3, n_drop=3
    )
    second, share_second, meta_second = build_labeled_window(
        background, closest_minute=8, added_energy=10.0, seed=3, n_drop=3
    )
    assert np.allclose(first, second, equal_nan=True)
    assert share_first == share_second
    assert meta_first == meta_second
    for minute in meta_first["dropped_minutes"]:
        assert np.isnan(first[minute - 1]).all()


def test_a_stack_matches_one_window_at_a_time():
    backgrounds = np.stack([
        np.full((15, 25), 40.0),
        np.full((15, 25), 55.0),
        np.zeros((15, 25)),
    ])
    mixtures, shares, metadata = build_labeled_windows(
        backgrounds,
        closest_minutes=[4, 8, 12],
        added_energies=30.0,
        seeds=[1, 2, 3],
        n_drops=2,
    )
    assert mixtures.shape == (3, 15, 25)
    assert shares.shape == (3,)
    assert len(metadata) == 3
    for index in range(3):
        mixture, share, info = build_labeled_window(
            backgrounds[index],
            closest_minute=int([4, 8, 12][index]),
            added_energy=30.0,
            seed=index + 1,
            n_drop=2,
        )
        assert np.allclose(mixtures[index], mixture, equal_nan=True)
        assert shares[index] == share
        assert metadata[index] == info


def test_a_stack_can_mix_one_recording_pattern():
    backgrounds = np.zeros((2, 15, 25))
    patterns = np.stack([
        np.outer(passage_envelope(15, 8), ship_spectrum()),
        np.outer(passage_envelope(15, 3), ship_spectrum()),
    ])
    mixtures, shares, metadata = build_labeled_windows(
        backgrounds,
        closest_minutes=[8, 3],
        added_energies=[10.0, 20.0],
        seeds=5,
        n_drops=0,
        patterns=patterns,
    )
    for index, energy in enumerate([10.0, 20.0]):
        mixture, share, info = build_labeled_window(
            backgrounds[index],
            closest_minute=int([8, 3][index]),
            added_energy=energy,
            seed=5,
            n_drop=0,
            pattern=patterns[index],
        )
        assert np.allclose(mixtures[index], mixture, equal_nan=True)
        assert shares[index] == share
        assert metadata[index] == info


if __name__ == "__main__":
    test_entry_point_matches_the_separate_steps()
    test_a_separable_pattern_matches_the_fallback()
    test_pattern_keeps_the_closest_minute_the_caller_passed()
    test_same_seed_blanks_the_same_minutes()
    test_a_stack_matches_one_window_at_a_time()
    test_a_stack_can_mix_one_recording_pattern()
    print("pipeline checks passed")
