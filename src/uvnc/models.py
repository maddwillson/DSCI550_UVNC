"""Site-held-out models and leaderboard tracking.

Owner: Levi Glaser.
Input: Madeleine's feature matrix and Joshua's ``share_added_vessel`` labels.
Fit every model on training rows only. Hold out whole recording sites when
measuring error. ``score_added_vessel`` is the diagnostic column and does
not affect the leaderboard.

``fit_site_held_out`` does the cross-validation and the final fit.
``predict_share`` scores new windows such as ``Test.csv``.
``make_submission`` and ``log_submission`` write the Zindi file and the
leaderboard log.
"""

import csv
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from uvnc.features import FEATURE_NAMES, BackgroundNeighbors, FittedFeatures

BASELINE_NAMES = ("constant_zero", "train_mean", "above_floor")
MODEL_NAMES = ("ridge", "tree", "forest", "boosting")
LOG_COLUMNS = (
    "date",
    "file",
    "model",
    "cv_score",
    "cv_site_mean_score",
    "public_score",
    "notes",
)


def make_models(seed=550):
    """Return the four unfitted models, keyed by name.

    Ridge gets a scaler because its penalty depends on feature scale. The
    tree models do not need one.
    """
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "tree": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=50, random_state=seed
        ),
        "forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            max_features=0.33,
            n_jobs=-1,
            random_state=seed,
        ),
        "boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def site_folds(site_ids, max_per_site=None, seed=550):
    """Yield ``(site, train_rows, held_out_rows)`` with one site held out.

    Every row of the held-out site is scored. ``max_per_site`` caps how many
    rows each training site contributes, so one large site cannot dominate.
    """
    site_ids = np.asarray(site_ids)
    sites = np.unique(site_ids)
    if len(sites) < 2:
        raise ValueError("Site-held-out folds need at least two sites.")
    capped = _cap_rows(site_ids, max_per_site, seed)
    for site in sites:
        yield site, capped[site_ids[capped] != site], np.flatnonzero(site_ids == site)


def _cap_rows(site_ids, max_per_site, seed):
    """Sorted row positions with at most ``max_per_site`` rows per site."""
    rng = np.random.default_rng(seed)
    keep = []
    for site in np.unique(site_ids):
        rows = np.flatnonzero(site_ids == site)
        if max_per_site is not None and len(rows) > max_per_site:
            rows = rng.choice(rows, max_per_site, replace=False)
        keep.append(rows)
    return np.sort(np.concatenate(keep))


def _fit_transformers(windows, backgrounds, site_ids, rows, seed):
    """Fit Madeleine's PCA, k-means, and neighbor features on training rows."""
    fitted = {}
    if windows is not None:
        fitted["shape"] = FittedFeatures(seed=seed).fit(windows[rows], site_ids[rows])
    if backgrounds is not None:
        if windows is None:
            raise ValueError("backgrounds needs windows as well.")
        fitted["neighbors"] = BackgroundNeighbors(seed=seed).fit(
            backgrounds[rows], site_ids[rows]
        )
    return fitted


def _design(fitted, features, windows, site_ids=None):
    """Stack the stateless features with the fold-fitted ones."""
    blocks = [features]
    if "shape" in fitted:
        blocks.append(fitted["shape"].transform(windows))
    if "neighbors" in fitted:
        blocks.append(fitted["neighbors"].transform(windows, site_ids))
    return np.column_stack(blocks)


def _design_names(fitted, feature_names):
    names = list(feature_names)
    for key in ("shape", "neighbors"):
        if key in fitted:
            names += list(fitted[key].feature_names_)
    return names


def _baseline_predictions(features, train_share, feature_names, n_rows):
    """Predictions from the three no-learning baselines for ``n_rows`` rows."""
    out = {
        "constant_zero": np.zeros(n_rows),
        "train_mean": np.full(n_rows, float(np.mean(train_share))),
    }
    if "above_floor_share" in feature_names:
        column = list(feature_names).index("above_floor_share")
        out["above_floor"] = np.clip(features[:, column], 0.0, 1.0)
    return out


def error_summary(share, predicted):
    """RMSE, MAE, and the Zindi score ``1 - RMSE`` for one set of rows."""
    share = np.asarray(share, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((predicted - share) ** 2)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(predicted - share))),
        "score": 1.0 - rmse,
    }


def error_breakdown(predictions, by, names=None):
    """Error of each model within each group of held-out rows.

    Parameters
    ----------
    predictions : pandas.DataFrame
        The ``predictions`` table from ``fit_site_held_out``.
    by : str or array-like
        A column of ``predictions``, or one group label per row.
    names : list of str, optional
        Models to include. Defaults to every prediction column.

    Returns
    -------
    pandas.DataFrame
        One row per group and model: ``n``, ``rmse``, ``mae``, ``score``.
    """
    if names is None:
        names = [c for c in predictions.columns if c.startswith("pred_")]
    else:
        names = [f"pred_{n}" for n in names]
    groups = predictions[by] if isinstance(by, str) else pd.Series(by, name="group")
    rows = []
    for group, part in predictions.groupby(groups.to_numpy(), observed=True):
        for column in names:
            row = {"group": group, "model": column[5:], "n": len(part)}
            row.update(error_summary(part["share"], part[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def fit_site_held_out(
    features,
    share,
    site_ids,
    windows=None,
    backgrounds=None,
    background_features=None,
    models=None,
    max_per_site=None,
    feature_names=None,
    seed=550,
    final_fit=True,
):
    """Fit the models and return site-held-out scores.

    Each site is held out once. The models, and the PCA, k-means, and
    neighbor features when ``windows`` is given, are fit on the other sites
    and scored on every row of the held-out site. Predictions are clipped to
    0 through 1.

    Parameters
    ----------
    features : numpy.ndarray
        Shape ``(n, n_features)`` from ``window_features``.
    share : numpy.ndarray
        Shape ``(n,)``. Vessel share of total linear energy.
    site_ids : array-like
        Shape ``(n,)``. Cross-validation keeps every row from a site
        on the same side of the split.
    windows : numpy.ndarray, optional
        Shape ``(n, 15, 25)``. The mixtures behind ``features``. When given,
        ``FittedFeatures`` is fit inside each fold and its columns are added.
    backgrounds : numpy.ndarray, optional
        Shape ``(n, 15, 25)``. The clean background behind each mixture, in
        the same row order. When given, ``BackgroundNeighbors`` is fit
        inside each fold and its two columns are added.
    background_features : numpy.ndarray, optional
        Shape ``(n, n_features)``. ``window_features`` of each background
        with the same hidden minutes as its mixture. When given with
        ``backgrounds``, the held-out site's backgrounds are scored as
        windows with no added vessel, and the ROC-AUC of the predicted
        share against "vessel added or not" is reported. Nothing is fit on
        these rows.
    models : dict, optional
        Unfitted estimators by name. Defaults to ``make_models(seed)``.
    max_per_site : int or None
        Cap on training rows per site. ``None`` uses every row, which
        scored best in ``notebooks/models.ipynb``.
    feature_names : list of str, optional
        Names for the columns of ``features``. Defaults to ``FEATURE_NAMES``
        when the width matches.
    seed : int
        Seed for the row cap and the models.
    final_fit : bool
        Also fit every model on all sites, for predicting ``Test.csv``.

    Returns
    -------
    dict
        ``scores``: one row per model and baseline, with the pooled and
        site-averaged held-out error, timings, and model size.
        ``by_site``: the same error for each held-out site.
        ``predictions``: one row per training window with its site, share,
        and held-out prediction from every model.
        ``models``, ``transformers``, ``feature_names``: the final fit on
        all sites, for ``predict_share``. Empty when ``final_fit`` is off.
        ``best``: the model with the lowest site-averaged RMSE.
    """
    features = np.asarray(features, dtype=np.float64)
    share = np.asarray(share, dtype=np.float64)
    site_ids = np.asarray(site_ids)
    if features.ndim != 2:
        raise ValueError("features must have shape (n, n_features).")
    if not (len(features) == len(share) == len(site_ids)):
        raise ValueError("features, share, and site_ids must have the same length.")
    if windows is not None and len(windows) != len(features):
        raise ValueError("windows must have one window per feature row.")
    if backgrounds is not None and len(backgrounds) != len(features):
        raise ValueError("backgrounds must have one window per feature row.")
    if feature_names is None:
        if features.shape[1] == len(FEATURE_NAMES):
            feature_names = list(FEATURE_NAMES)
        else:
            feature_names = [f"f{i}" for i in range(features.shape[1])]
    if models is None:
        models = make_models(seed)
    score_negatives = backgrounds is not None and background_features is not None

    names = list(BASELINE_NAMES) + list(models)
    if "above_floor_share" not in feature_names:
        names.remove("above_floor")
    held_out = {name: np.full(len(share), np.nan) for name in names}
    negative = {name: np.full(len(share), np.nan) for name in models}
    fit_seconds = dict.fromkeys(models, 0.0)
    predict_seconds = dict.fromkeys(models, 0.0)

    for site, train, test in site_folds(site_ids, max_per_site, seed):
        fitted = _fit_transformers(windows, backgrounds, site_ids, train, seed)
        x_train = _design(fitted, features[train], _rows(windows, train), site_ids[train])
        x_test = _design(fitted, features[test], _rows(windows, test), site_ids[test])
        if score_negatives:
            hidden = np.isnan(windows[test])
            quiet = np.where(hidden, np.nan, backgrounds[test])
            x_negative = _design(fitted, background_features[test], quiet, site_ids[test])
        baseline = _baseline_predictions(
            features[test], share[train], feature_names, len(test)
        )
        for name, values in baseline.items():
            held_out[name][test] = values
        for name, model in models.items():
            fold_model = clone(model)
            start = time.perf_counter()
            fold_model.fit(x_train, share[train])
            fit_seconds[name] += time.perf_counter() - start
            start = time.perf_counter()
            held_out[name][test] = np.clip(fold_model.predict(x_test), 0.0, 1.0)
            predict_seconds[name] += time.perf_counter() - start
            if score_negatives:
                negative[name][test] = np.clip(fold_model.predict(x_negative), 0.0, 1.0)

    predictions = pd.DataFrame({"site_id": site_ids, "share": share})
    if "missing_minutes" in feature_names:
        column = list(feature_names).index("missing_minutes")
        predictions["missing_minutes"] = features[:, column].astype(int)
    for name in names:
        predictions[f"pred_{name}"] = held_out[name]
    if score_negatives:
        for name in models:
            predictions[f"background_{name}"] = negative[name]

    by_site = error_breakdown(predictions, "site_id").rename(columns={"group": "site_id"})
    if score_negatives:
        by_site["auc"] = [
            _site_auc(predictions, row.site_id, row.model) for row in by_site.itertuples()
        ]

    result = {
        "predictions": predictions,
        "by_site": by_site,
        "models": {},
        "transformers": {},
        "feature_names": list(feature_names),
    }

    final_seconds = {}
    if final_fit:
        rows = _cap_rows(site_ids, max_per_site, seed)
        fitted = _fit_transformers(windows, backgrounds, site_ids, rows, seed)
        x_all = _design(fitted, features[rows], _rows(windows, rows), site_ids[rows])
        for name, model in models.items():
            start = time.perf_counter()
            result["models"][name] = clone(model).fit(x_all, share[rows])
            final_seconds[name] = time.perf_counter() - start
        result["transformers"] = fitted
        result["feature_names"] = _design_names(fitted, feature_names)
        result["train_mean"] = float(np.mean(share[rows]))

    rows = []
    for name in names:
        row = {"model": name}
        row.update(error_summary(share, held_out[name]))
        per_site = by_site[by_site["model"] == name]
        row["site_mean_rmse"] = float(per_site["rmse"].mean())
        row["site_mean_score"] = 1.0 - row["site_mean_rmse"]
        row["worst_site_rmse"] = float(per_site["rmse"].max())
        if score_negatives:
            row["site_mean_auc"] = float(per_site["auc"].mean())
        if name in models:
            row["cv_fit_seconds"] = fit_seconds[name]
            row["predict_ms_per_window"] = 1000.0 * predict_seconds[name] / len(share)
            if final_fit:
                row["final_fit_seconds"] = final_seconds[name]
                row["model_megabytes"] = len(pickle.dumps(result["models"][name])) / 1e6
        rows.append(row)
    scores = pd.DataFrame(rows).sort_values("site_mean_rmse").reset_index(drop=True)
    result["scores"] = scores
    learned = scores[scores["model"].isin(list(models))]
    result["best"] = str(learned.iloc[0]["model"])
    return result


def _rows(windows, rows):
    return None if windows is None else windows[rows]


def _site_auc(predictions, site, name):
    """ROC-AUC of predicted share: the site's mixtures against its backgrounds."""
    column = f"background_{name}"
    if column not in predictions.columns:
        return np.nan
    part = predictions[predictions["site_id"] == site]
    scores = np.concatenate([part[f"pred_{name}"], part[column]])
    labels = np.concatenate([np.ones(len(part)), np.zeros(len(part))])
    return float(roc_auc_score(labels, scores))


def predict_share(result, features, windows=None, model=None):
    """Predict ``share_added_vessel`` for new windows with the final fit.

    Parameters
    ----------
    result : dict
        The return value of ``fit_site_held_out`` with ``final_fit`` on.
    features : numpy.ndarray
        Shape ``(n, n_features)`` from ``window_features``.
    windows : numpy.ndarray, optional
        Shape ``(n, 15, 25)``. Needed when the fit used ``windows``.
    model : str, optional
        Which model to use. Defaults to ``result["best"]``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``, clipped to 0 through 1. Site ids are left out, so
        neighbor backgrounds come from every training site.
    """
    name = result["best"] if model is None else model
    if name == "constant_zero":
        return np.zeros(len(features))
    if name == "train_mean":
        return np.full(len(features), result["train_mean"])
    if name not in result["models"]:
        raise KeyError(f"No final fit for {name!r}. Run with final_fit=True.")
    if result["transformers"] and windows is None:
        raise ValueError("This fit used windows, so pass windows here as well.")
    x_new = _design(result["transformers"], np.asarray(features, dtype=np.float64), windows)
    return np.clip(result["models"][name].predict(x_new), 0.0, 1.0)


def make_submission(ids, share, score=None, path=None):
    """Build the Zindi submission table, and write it when ``path`` is given.

    ``score_added_vessel`` defaults to the predicted share. A larger share
    means a vessel is more likely to have been added, and the column only
    needs to rank windows.
    """
    share = np.asarray(share, dtype=np.float64)
    score = share if score is None else np.asarray(score, dtype=np.float64)
    if len(ids) != len(share) or len(share) != len(score):
        raise ValueError("ids, share, and score must have the same length.")
    if not (np.isfinite(share).all() and np.isfinite(score).all()):
        raise ValueError("Predictions must be finite.")
    if share.min() < 0.0 or share.max() > 1.0 or score.min() < 0.0 or score.max() > 1.0:
        raise ValueError("Predictions must lie between 0 and 1.")
    table = pd.DataFrame(
        {"ID": list(ids), "share_added_vessel": share, "score_added_vessel": score}
    )
    if table["ID"].duplicated().any():
        raise ValueError("Each ID may appear once.")
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    return table


def log_submission(
    log_path,
    file,
    model,
    cv_score,
    cv_site_mean_score=None,
    public_score=None,
    notes="",
    date=None,
):
    """Append one submission to the leaderboard log and return the whole log.

    Leave ``public_score`` empty until Zindi shows it, then fill that cell
    in. ``cv_score`` is ``1 - RMSE`` over all site-held-out rows, the same
    formula as the leaderboard.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d") if date is None else date,
        "file": str(file),
        "model": model,
        "cv_score": _rounded(cv_score),
        "cv_site_mean_score": _rounded(cv_site_mean_score),
        "public_score": _rounded(public_score),
        "notes": notes,
    }
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return pd.read_csv(log_path)


def _rounded(value):
    return "" if value is None else f"{float(value):.4f}"
