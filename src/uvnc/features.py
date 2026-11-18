"""Band statistics, PCA, k-means, and the background estimate.

Owner: Madeleine Willson.
Input: training windows of shape ``(n, 15, 25)`` from ``windows_from_frame``.
Output: one feature row per window. Fit PCA, k-means, and scalers on
training windows only.
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from uvnc.cleaning import BAND_CENTRES
from uvnc.units import db_to_linear, linear_to_db


FEATURE_NAMES = (
    [f"peak_minus_median_TOL_{c}" for c in BAND_CENTRES]
    + [f"range_TOL_{c}" for c in BAND_CENTRES]
    + [f"spread_TOL_{c}" for c in BAND_CENTRES]
    + ["broadband_peak_minus_median", "broadband_peak_minute", 
       "missing_minutes", "above_floor_share"]
)



def _peak_minus_median(windows):
    """Loudest minute/band minus the median minute/band, per band/window."""

    return np.nanmax(windows, axis=1) - np.nanmedian(windows, axis=1)
  


def _band_range(windows):
    """Loudest minute minus the quietest minute, per band. 
    Returns ``(n, 25)`` in dB."""
    
    return np.nanmax(windows, axis=1) - np.nanmin(windows, axis=1)


def _band_spread(windows):
    """Standard deviation of minutes, per band.
    Returns ``(n, 25)`` in dB.
    """
    return  np.nanstd(windows, axis=1)  



def _broadband_db(windows):
    """Total level of each minute across all 25 bands.
    Bands are added in linear power. A missing minute stays NaN.
    Returns ``(n, 15)`` in dB.
    """
    linear = db_to_linear(windows)
    total = np.sum(linear, axis= 2)

    return linear_to_db(total)


def _missing_minutes(windows):
    """Number of whole minutes that are blank.
    Returns ``(n,)``.
    """
    return np.isnan(windows).all(axis=2).sum(axis=1)


def _above_floor(windows, percentile  = 20):
    """The window energy that is above the band's low percentile over minutes, per band. 

    Returns ``(n,)``, between 0 and 1.
    """
    floor_db = np.nanpercentile(windows, percentile, axis=1)
    linear = db_to_linear(windows)
    floor_linear = db_to_linear(floor_db)[:, None, :]

    mins_nonneg = np.clip(linear - floor_linear, 0, None) # get rid of negatives

    return np.nansum(mins_nonneg, axis=(1,2)) / np.nansum(linear, axis = (1,2))


def _spectrum_shape(windows):
    """Median spectrum minus its own mean level, per window.

    Removes absolute level, so recorder offsets drop out.
    Returns ``(n, 25)`` in dB.
    """
    spectrum = np.nanmedian(windows, axis=1)

    return spectrum - spectrum.mean(axis=1, keepdims=True)


def _cap_per_site(site_ids, site_max, seed):
    """Row positions with at most ``site_max`` rows from each site.

    Returns a 1-D array of row positions.
    """
    rng = np.random.default_rng(seed)

    keep = []
    for site in np.unique(site_ids):
        rows = np.flatnonzero(site_ids == site)

        if len(rows) > site_max:
            rows = rng.choice(rows, site_max, replace=False)
        keep.append(rows)

    return np.concatenate(keep)





def window_features(windows):
    """Turn minute-by-band windows into one feature row each.

    Parameters
    ----------
    windows : numpy.ndarray
        Shape ``(n, 15, 25)``. NaN marks a missing cell.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 79)``.
    """


    broadband = _broadband_db(windows)
    broadband_peak = _peak_minus_median(broadband)

    peak_minute = np.nanargmax(broadband, axis=1) + 1 
    
    return np.column_stack([
        _peak_minus_median(windows),
        _band_range(windows),
        _band_spread(windows),
        broadband_peak,
        peak_minute,
        _missing_minutes(windows),
        _above_floor(windows),
    ])




class FittedFeatures:
    """PCA and k-means features, fit on training windows inside each cross validation fold.
    """

    def __init__(self, n_components=5, n_clusters=4, site_max=3000, seed=550): #found clusters=4 from elbow curve
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.site_max = site_max
        self.seed = seed

    def fit(self, windows, site_ids):
        """Learn PCA and k-means from training windows.
        """
        keep = _cap_per_site(site_ids, self.site_max, self.seed)

        shapes = _spectrum_shape(windows[keep])
        self.pca_ = PCA(n_components=self.n_components, random_state=self.seed).fit(shapes)
        scores = self.pca_.transform(shapes)

        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed).fit(scores)
        self.feature_names_ = (
            [f"pca_{i + 1}" for i in range(self.n_components)]      
            + [f"cluster_distance_{i + 1}" for i in range(self.n_clusters)]
        )

        return self

    def transform(self, windows):
        """Return ``(n, n_components + n_clusters)`` features for any windows."""

        scores = self.pca_.transform(_spectrum_shape(windows))
        return np.column_stack([scores, self.kmeans_.transform(scores)])
    




class BackgroundNeighbors:
    """Expected background peak from similar clean windows at other sites.
    Fit on backgrounds with no vessel from the training rows.
    """

    def __init__(self, n_neighbors=10, site_max=3000, seed=550):
        self.n_neighbors = n_neighbors
        self.site_max = site_max
        self.seed = seed

    def fit(self, backgrounds, site_ids):
        """Store the shape and natural broadband peak of each background."""

        keep = _cap_per_site(site_ids, self.site_max, self.seed)

        self.shapes_ = _spectrum_shape(backgrounds[keep])
        self.peaks_ = _peak_minus_median(_broadband_db(backgrounds[keep]))
        self.sites_ = site_ids[keep] 
        self.feature_names_ = ["expected_peak", "excess_peak"]
        
        return self

    def transform(self, windows, site_ids=None):
        """Return ``(n, 2)``: expected background peak, and observed - expected.

        Pass ``site_ids`` for training or held out rows so neighbors come from other sites but leave it out for ``Test.csv``.
        """

        shapes = _spectrum_shape(windows)
        observed = _peak_minus_median(_broadband_db(windows))
        expected = np.empty(len(windows))

        groups = [None] if site_ids is None else np.unique(site_ids)
        for site in groups:
            if site is None:
                rows = np.arange(len(windows))
                pool = np.ones(len(self.sites_), dtype=bool)
            else:
                rows = np.flatnonzero(site_ids == site)
                pool = self.sites_ != site
            nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.shapes_[pool])
            _, idx = nn.kneighbors(shapes[rows])
            expected[rows] = self.peaks_[pool][idx].mean(axis=1)

        return np.column_stack([expected, observed - expected])