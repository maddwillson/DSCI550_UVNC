"""Vessel passage shape and the third-octave spectrum.

Owner: Joshua Lee.
These functions describe a controlled vessel addition in linear power.
``third_octave_centres`` matches the starter notebook. ``passage_envelope``
and ``ship_spectrum`` are the from-scratch fallback used until a VTUAD clip
is converted with ``minute_band_power``. Mixing stays in ``mixing.py``.
"""

import numpy as np

_N_BANDS = 25


def third_octave_centres():
    """Return the 25 third-octave centre frequencies in Hz.

    This is the same grid as ``minute_band_power`` in the starter notebook:
    ``10 ** (1.5 + 0.1 * np.arange(25))``. The first centre is 31.62 Hz and
    the last is 7943.28 Hz.

    Returns
    -------
    numpy.ndarray
        Shape ``(25,)``, dtype float64, in increasing order.
    """
    return 10.0 ** (1.5 + 0.1 * np.arange(_N_BANDS, dtype=np.float64))


def passage_envelope(n_minutes, closest_minute):
    """Return the fraction of vessel energy in each minute.

    The weight is a Gaussian in minute position. It peaks at
    ``closest_minute`` and falls off on either side. Minute numbers are
    1-based, matching ``m01`` through ``m15``.

    Parameters
    ----------
    n_minutes : int
        Length of the window. The challenge windows use 15.
    closest_minute : int
        Minute of closest approach, from 1 through ``n_minutes``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_minutes,)``. Nonnegative weights that sum to 1.
    """
    n_minutes = _require_count("n_minutes", n_minutes)
    closest = _require_count("closest_minute", closest_minute)
    if closest > n_minutes:
        raise ValueError("closest_minute must fall inside the window.")
    minutes = np.arange(1, n_minutes + 1, dtype=np.float64)
    sigma = max(n_minutes / 6.0, 1.0)
    weights = np.exp(-0.5 * ((minutes - closest) / sigma) ** 2)
    return weights / weights.sum()


def ship_spectrum():
    """Return a smooth low-frequency ship spectrum across the 25 bands.

    Weights are proportional to ``1 / centre_frequency``, so low bands carry
    more of the vessel energy than high bands. This is a fallback shape for
    when a VTUAD passage is not available. It is not a fitted recording.

    Returns
    -------
    numpy.ndarray
        Shape ``(25,)``. Nonnegative weights that sum to 1, in the same
        band order as ``third_octave_centres``.
    """
    weights = 1.0 / third_octave_centres()
    return weights / weights.sum()


def _require_count(name, value):
    """Return ``value`` as an int of at least 1."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    count = int(value)
    if count < 1:
        raise ValueError(f"{name} must be at least 1.")
    return count
