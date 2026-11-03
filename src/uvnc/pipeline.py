"""One entry point for a labeled training window.

Owner: Joshua Lee.
``build_labeled_windows`` is the training table Madeleine and Levi should
read. Each row is one ``build_labeled_window`` call. Pass a ``pattern``
from ``passage_pattern`` to mix a recording. Leave it out to use the
fallback envelope and ship spectrum.
"""

import numpy as np

from uvnc.mixing import mask_random_minutes, mix_passage, mix_pattern, vessel_share
from uvnc.passage import passage_envelope, ship_spectrum

_N_BANDS = 25


def build_labeled_window(
    background_db, closest_minute, added_energy, seed, n_drop, pattern=None
):
    """Mix a passage into a background and label the observed energy share.

    The background must come from the South African site files, not from
    ``Test.csv``. Minutes are 1-based. The same ``seed`` and ``n_drop``
    blank the same minutes on every run. Leave ``pattern`` unset to build
    the fallback passage from ``closest_minute``. Pass the array from
    ``passage_pattern`` to mix that recording instead. ``closest_minute``
    is stored as given. It is not read off the pattern.

    Parameters
    ----------
    background_db : numpy.ndarray
        Shape ``(n_minutes, 25)``. Decibel band levels. NaN is missing.
    closest_minute : int
        Minute of closest approach, from 1 through ``n_minutes``.
    added_energy : float
        Total linear passage power assigned across the full window.
    seed : int
        Seed for the minute mask.
    n_drop : int
        How many whole minutes to blank after mixing.
    pattern : numpy.ndarray, optional
        Shape ``(n_minutes, 25)``. Finite weights sum to 1. ``None`` uses
        the fallback envelope and ship spectrum.

    Returns
    -------
    tuple
        ``(mixture_db, share, metadata)``. ``mixture_db`` is the masked
        window in decibels. ``share`` is ``share_added_vessel`` on the
        cells that remain. ``metadata`` holds ``seed``, ``closest_minute``,
        and ``dropped_minutes``.
    """
    background = np.asarray(background_db, dtype=np.float64)
    if background.ndim != 2 or background.shape[1] != _N_BANDS:
        raise ValueError("background_db must have shape (n_minutes, 25).")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    n_minutes = background.shape[0]
    if pattern is None:
        envelope = passage_envelope(n_minutes, closest_minute)
        spectrum = ship_spectrum()
        mixture = mix_passage(background, envelope, spectrum, added_energy)
    else:
        mixture = mix_pattern(background, pattern, added_energy)
    masked, dropped = mask_random_minutes(
        mixture, np.random.default_rng(int(seed)), n_drop
    )
    masked_background = np.array(background, copy=True)
    if dropped.size:
        masked_background[dropped - 1, :] = np.nan
    share = vessel_share(masked_background, masked)
    metadata = {
        "seed": int(seed),
        "closest_minute": int(closest_minute),
        "dropped_minutes": [int(minute) for minute in dropped],
    }
    return masked, share, metadata


def build_labeled_windows(
    backgrounds_db,
    closest_minutes,
    added_energies,
    seeds,
    n_drops,
    patterns=None,
):
    """Label many background windows.

    Each numeric setting may be one value for every window or one value
    per window. Leave ``patterns`` out for the fallback passage on every
    row, or pass one pattern per window.

    Parameters
    ----------
    backgrounds_db : numpy.ndarray
        Shape ``(n_windows, n_minutes, 25)``. Decibel band levels.
    closest_minutes, seeds, n_drops : int or array-like
        One integer, or one integer per window.
    added_energies : float or array-like
        One linear-power total, or one total per window.
    patterns : numpy.ndarray, optional
        Shape ``(n_windows, n_minutes, 25)``. Finite weights in each
        pattern sum to 1.

    Returns
    -------
    tuple
        ``(mixtures_db, shares, metadata)``. ``mixtures_db`` has shape
        ``(n_windows, n_minutes, 25)``. ``shares`` has shape
        ``(n_windows,)``. ``metadata`` is one dict per window, with
        ``seed``, ``closest_minute``, and ``dropped_minutes``.
    """
    backgrounds = np.asarray(backgrounds_db, dtype=np.float64)
    if backgrounds.ndim != 3 or backgrounds.shape[2] != _N_BANDS:
        raise ValueError(
            "backgrounds_db must have shape (n_windows, n_minutes, 25)."
        )
    n_windows, n_minutes, _ = backgrounds.shape
    closest = _per_window(closest_minutes, n_windows, "closest_minutes", integer=True)
    energy = _per_window(added_energies, n_windows, "added_energies", integer=False)
    seed = _per_window(seeds, n_windows, "seeds", integer=True)
    drops = _per_window(n_drops, n_windows, "n_drops", integer=True)
    pattern_stack = None
    if patterns is not None:
        pattern_stack = np.asarray(patterns, dtype=np.float64)
        if pattern_stack.shape != backgrounds.shape:
            raise ValueError("patterns must have the same shape as backgrounds_db.")
    mixtures = np.empty((n_windows, n_minutes, _N_BANDS), dtype=np.float64)
    shares = np.empty(n_windows, dtype=np.float64)
    metadata = []
    for index in range(n_windows):
        pattern = None if pattern_stack is None else pattern_stack[index]
        mixture, share, info = build_labeled_window(
            backgrounds[index],
            int(closest[index]),
            float(energy[index]),
            int(seed[index]),
            int(drops[index]),
            pattern,
        )
        mixtures[index] = mixture
        shares[index] = share
        metadata.append(info)
    return mixtures, shares, metadata


def _per_window(values, n_windows, name, integer):
    """Expand one value, or check a per-window column."""
    array = np.asarray(values)
    if array.dtype == bool:
        raise TypeError(f"{name} must be numeric.")
    if array.ndim == 0:
        array = np.full(n_windows, array)
    if array.shape != (n_windows,):
        raise ValueError(f"{name} must be one value or length {n_windows}.")
    if integer and not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be integers.")
    return array
