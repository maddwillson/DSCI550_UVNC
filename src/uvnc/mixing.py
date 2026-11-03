"""Mix a vessel passage into a background window and label the share.

Owner: Joshua Lee.
Addition happens in linear power from ``uvnc.units``. The label matches
the competition definition: added passage energy divided by mixture energy,
summed over the same observed minute-band cells.

Build backgrounds from the South African site files. ``Test.csv`` is for
inference and for checking the column layout.
"""

import numpy as np

from uvnc.units import db_to_linear, linear_to_db

_N_BANDS = 25


def mix_passage(background_db, envelope, spectrum, added_energy):
    """Add a vessel passage to a background window.

    ``added_energy`` is the total linear power of the passage before any
    cell is dropped. It is spread by ``outer(envelope, spectrum)``. A cell
    that is missing in the background stays missing. Passage energy that
    lands on a missing cell is left out of the window rather than moved
    onto the observed minutes.

    Parameters
    ----------
    background_db : numpy.ndarray
        Shape ``(n_minutes, 25)``. Decibel band levels. NaN is missing.
    envelope : numpy.ndarray
        Shape ``(n_minutes,)``. Minute weights from ``passage_envelope``.
        Must sum to 1.
    spectrum : numpy.ndarray
        Shape ``(25,)``. Band weights from ``ship_spectrum`` or a converted
        VTUAD clip. Must sum to 1.
    added_energy : float
        Total linear passage power assigned across the full window.

    Returns
    -------
    numpy.ndarray
        Mixture in decibels, same shape as ``background_db``. Missing cells
        stay NaN.
    """
    background = np.asarray(background_db, dtype=np.float64)
    if background.ndim != 2 or background.shape[1] != _N_BANDS:
        raise ValueError("background_db must have shape (n_minutes, 25).")
    n_minutes = background.shape[0]
    minute_weights = _require_distribution("envelope", envelope, n_minutes)
    band_weights = _require_distribution("spectrum", spectrum, _N_BANDS)
    energy = _require_energy(added_energy)
    pattern = energy * np.outer(minute_weights, band_weights)
    observed = np.isfinite(background)
    background_linear = db_to_linear(background)
    mixture_linear = np.where(observed, background_linear + pattern, np.nan)
    return linear_to_db(mixture_linear)


def vessel_share(background_db, mixture_db):
    """Return the added passage's fraction of mixture energy.

    The fraction is the sum of added linear energy divided by the sum of
    mixture linear energy. Both sums use the cells that are present in the
    background and in the mixture. A target of 0 means no passage energy
    landed on an observed cell.

    Parameters
    ----------
    background_db, mixture_db : numpy.ndarray
        Decibel windows of the same shape.

    Returns
    -------
    float
        Energy fraction in ``[0, 1]`` for a passage that was added in
        linear power.
    """
    background = np.asarray(background_db, dtype=np.float64)
    mixture = np.asarray(mixture_db, dtype=np.float64)
    if background.shape != mixture.shape:
        raise ValueError("background_db and mixture_db must have the same shape.")
    observed = np.isfinite(background) & np.isfinite(mixture)
    if not observed.any():
        raise ValueError("The window has no observed minute-band cells.")
    background_linear = db_to_linear(background)
    mixture_linear = db_to_linear(mixture)
    added = mixture_linear[observed] - background_linear[observed]
    total_mixture = mixture_linear[observed].sum()
    if total_mixture <= 0:
        raise ValueError("Mixture energy on observed cells must be positive.")
    return float(added.sum() / total_mixture)


def mask_random_minutes(window_db, generator, n_drop):
    """Drop whole minutes from a window.

    The challenge files leave some minute positions blank. Those blanks are
    missing observations. This copies ``window_db`` and sets the chosen
    minutes to NaN. Recompute ``vessel_share`` after masking, because the
    label uses only the cells that remain.

    Parameters
    ----------
    window_db : numpy.ndarray
        Shape ``(n_minutes, n_bands)``.
    generator : numpy.random.Generator
        Source of the minute choices. Pass one in so a run can be repeated.
    n_drop : int
        How many distinct minutes to blank.

    Returns
    -------
    tuple
        ``(masked_window, dropped_minutes)``. Dropped minutes are 1-based
        and sorted.
    """
    window = np.array(window_db, dtype=np.float64, copy=True)
    if window.ndim != 2:
        raise ValueError("window_db must be a minute-by-band array.")
    n_drop = _require_drop_count(n_drop, window.shape[0])
    if n_drop == 0:
        return window, np.array([], dtype=int)
    chosen = np.sort(generator.choice(window.shape[0], size=n_drop, replace=False))
    window[chosen, :] = np.nan
    return window, chosen + 1


def _require_distribution(name, values, length):
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},).")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    if abs(weights.sum() - 1.0) > 1e-8:
        raise ValueError(f"{name} must sum to 1.")
    return weights


def _require_energy(added_energy):
    energy = float(added_energy)
    if not np.isfinite(energy) or energy < 0:
        raise ValueError("added_energy must be a finite number of at least 0.")
    return energy


def _require_drop_count(n_drop, n_minutes):
    if isinstance(n_drop, bool) or not isinstance(n_drop, (int, np.integer)):
        raise TypeError("n_drop must be an integer.")
    count = int(n_drop)
    if count < 0 or count > n_minutes:
        raise ValueError("n_drop must be between 0 and the number of minutes.")
    return count
