"""Site-held-out models and leaderboard tracking.

Owner: Levi Glaser.
Input: Madeleine's feature matrix and Joshua's ``share_added_vessel`` labels.
Fit every model on training rows only. Hold out whole recording sites when
measuring error. ``score_added_vessel`` is the diagnostic column and does
not affect the leaderboard.
"""


def fit_site_held_out(features, share, site_ids):
    """Fit the models and return site-held-out scores.

    Parameters
    ----------
    features : numpy.ndarray
        Shape ``(n, n_features)`` from ``window_features``.
    share : numpy.ndarray
        Shape ``(n,)``. Vessel share of total linear energy.
    site_ids : array-like
        Shape ``(n,)``. Cross-validation keeps every row from a site
        on the same side of the split.

    Returns
    -------
    dict
        Fitted models and the held-out error for each one.
    """
    raise NotImplementedError("Levi: models, site-held-out CV, leaderboard log.")
