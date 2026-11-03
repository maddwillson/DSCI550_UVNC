"""Checks for mixing, the energy-fraction label, and minute gaps."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.mixing import mask_random_minutes, mix_passage, vessel_share
from uvnc.passage import passage_envelope, ship_spectrum
from uvnc.units import db_to_linear


def test_share_matches_the_competition_ratio_on_a_full_window():
    background = np.zeros((15, 25))
    envelope = passage_envelope(15, 8)
    spectrum = ship_spectrum()
    added_energy = 125.0
    mixture = mix_passage(background, envelope, spectrum, added_energy)
    share = vessel_share(background, mixture)
    background_energy = db_to_linear(background).sum()
    expected = added_energy / (background_energy + added_energy)
    assert abs(share - expected) < 1e-9
    assert 0.0 <= share <= 1.0


def test_missing_cells_stay_missing_and_drop_out_of_both_sums():
    background = np.zeros((15, 25))
    background[0, :] = np.nan
    envelope = passage_envelope(15, 8)
    spectrum = ship_spectrum()
    added_energy = 10.0
    mixture = mix_passage(background, envelope, spectrum, added_energy)
    assert np.isnan(mixture[0]).all()
    pattern = added_energy * np.outer(envelope, spectrum)
    observed = np.isfinite(background)
    expected_added = pattern[observed].sum()
    expected_total = db_to_linear(background)[observed].sum() + expected_added
    assert abs(vessel_share(background, mixture) - expected_added / expected_total) < 1e-9


def test_masking_removes_whole_minutes_and_changes_the_share():
    background = np.zeros((15, 25))
    mixture = mix_passage(
        background, passage_envelope(15, 8), ship_spectrum(), added_energy=50.0
    )
    before = vessel_share(background, mixture)
    masked, dropped = mask_random_minutes(mixture, np.random.default_rng(0), 2)
    assert dropped.shape == (2,)
    assert list(dropped) == sorted(dropped)
    for minute in dropped:
        assert np.isnan(masked[minute - 1]).all()
    kept = np.ones(15, dtype=bool)
    kept[dropped - 1] = False
    assert np.allclose(masked[kept], mixture[kept])
    masked_background = background.copy()
    masked_background[dropped - 1, :] = np.nan
    after = vessel_share(masked_background, masked)
    assert after != before


def test_zero_added_energy_has_share_zero():
    background = np.full((15, 25), 40.0)
    mixture = mix_passage(
        background, passage_envelope(15, 3), ship_spectrum(), added_energy=0.0
    )
    assert abs(vessel_share(background, mixture)) < 1e-12


if __name__ == "__main__":
    test_share_matches_the_competition_ratio_on_a_full_window()
    test_missing_cells_stay_missing_and_drop_out_of_both_sums()
    test_masking_removes_whole_minutes_and_changes_the_share()
    test_zero_added_energy_has_share_zero()
    print("mixing checks passed")
