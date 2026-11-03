"""One entry point for a labeled training window.

Owner: Joshua Lee.
``build_labeled_window`` is the function Madeleine and Levi should call.
It keeps the seed, the closest-approach minute, and the dropped minutes
with the returned window.
"""

import numpy as np

from uvnc.mixing import mask_random_minutes, mix_passage, vessel_share
from uvnc.passage import passage_envelope, ship_spectrum

_N_BANDS = 25


def build_labeled_window(background_db, closest_minute, added_energy, seed, n_drop):
    """Mix a passage into a background and label the observed energy share.

    The background must come from the South African site files, not from
    ``Test.csv``. Minutes are 1-based. The same ``seed`` and ``n_drop``
    blank the same minutes on every run.

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
    envelope = passage_envelope(n_minutes, closest_minute)
    spectrum = ship_spectrum()
    mixture = mix_passage(background, envelope, spectrum, added_energy)
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
