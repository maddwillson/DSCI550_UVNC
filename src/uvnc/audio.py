"""Convert one audio channel into minute-by-band linear power.

Owner: Joshua Lee.
The procedure matches ``minute_band_power`` in the starter notebook.
Use it on a permitted recording such as VTUAD. Do not commit the audio.
"""

import numpy as np
from scipy.signal import periodogram

from uvnc.passage import third_octave_centres


def minute_band_power(samples, sample_rate_hz):
    """Convert one channel to complete-minute mean-square band power.

    Each minute is thirty non-overlapping 2-second segments. The segment
    mean is removed, a one-sided Hann periodogram estimates the power
    spectral density, and each third-octave band is integrated. The thirty
    segment powers are then averaged in linear units. A band that reaches
    past the Nyquist rate is NaN, not zero. A trailing partial minute is
    omitted.

    Parameters
    ----------
    samples : array-like
        One finite channel. Shape ``(n_samples,)``.
    sample_rate_hz : int or float
        The recording's sample rate in hertz. It must be a positive integer.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_complete_minutes, 25)``. Linear mean-square power, in
        the same band order as ``third_octave_centres``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Use one channel with finite samples.")
    sample_rate = float(sample_rate_hz)
    if not np.isfinite(sample_rate) or sample_rate <= 0 or not sample_rate.is_integer():
        raise ValueError("Use the actual integer sample rate from the recording.")
    segment_size = int(2 * sample_rate)
    n_minutes = samples.size // (30 * segment_size)
    if n_minutes == 0:
        raise ValueError("At least one complete minute is needed.")
    segments = samples[: n_minutes * 30 * segment_size].reshape(-1, segment_size)
    frequency, psd = periodogram(
        segments,
        fs=sample_rate,
        window="hann",
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=1,
    )
    centres = third_octave_centres()
    power = np.full((len(segments), centres.size), np.nan)
    bin_width = frequency[1] - frequency[0]
    for band, centre in enumerate(centres):
        low = centre / 10 ** 0.05
        high = centre * 10 ** 0.05
        bins = (frequency >= low) & (frequency < high)
        if high <= sample_rate / 2 and bins.any():
            power[:, band] = psd[:, bins].sum(axis=1) * bin_width
    return power.reshape(n_minutes, 30, centres.size).mean(axis=1)
