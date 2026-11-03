"""Convert between decibel band levels and linear power.

Owner: Joshua Lee.
The challenge files store band levels in decibels. Energy is added in
linear power, using the power ratio ``10 ** (dB / 10)``.
"""

import numpy as np


def db_to_linear(db):
    """Convert dB band levels to linear power.

    Parameters
    ----------
    db : array-like
        Band levels. A missing cell is NaN and stays NaN. Negative levels
        are quiet bands, and they stay valid.

    Returns
    -------
    numpy.ndarray
        Linear power with the same shape as ``db``.
    """
    levels = np.asarray(db, dtype=np.float64)
    return np.power(10.0, levels / 10.0)


def linear_to_db(power):
    """Convert linear power to dB band levels.

    Parameters
    ----------
    power : array-like
        Linear power. NaN stays NaN. Zero power becomes negative infinity.
        A negative value is not a physical band power and becomes NaN.

    Returns
    -------
    numpy.ndarray
        Decibel band levels with the same shape as ``power``.
    """
    linear = np.asarray(power, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(linear)
