"""Band statistics, PCA, k-means, and the background estimate.

Owner: Madeleine Willson.
Input: training windows of shape ``(n, 15, 25)`` from ``windows_from_frame``.
Output: one feature row per window. Fit PCA, k-means, and scalers on
training windows only.
"""


def window_features(windows):
    """Turn minute-by-band windows into one feature row each.

    Parameters
    ----------
    windows : numpy.ndarray
        Shape ``(n, 15, 25)``. NaN marks a missing cell.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n_features)``.
    """
    raise NotImplementedError(
        "Madeleine: band statistics, PCA, k-means, nearest-neighbor background."
    )
