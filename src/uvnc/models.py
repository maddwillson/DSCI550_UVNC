"""Site-held-out models and leaderboard tracking."""

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

# Baselines need no fitting beyond the mean training share.
BASELINE_NAMES = ("constant_zero", "train_mean", "above_floor")
MODEL_NAMES = ("ridge", "tree", "forest", "boosting")
# Column order of submissions/leaderboard_log.csv.
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
    """Return the four unfitted models, keyed by name."""
    return {
        # The ridge penalty depends on feature scale, so scale first.
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        # Tree models split on thresholds and need no scaling.
        "tree": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=50, random_state=seed
        ),
        "forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            # A third of the features per split is the usual regression choice.
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
    """Yield ``(site, train_rows, held_out_rows)`` with one site held out."""
    site_ids = np.asarray(site_ids)
    sites = np.unique(site_ids)
    if len(sites) < 2:
        raise ValueError("Site-held-out folds need at least two sites.")
    # The cap applies to training rows only, so one large site cannot dominate.
    capped = _cap_rows(site_ids, max_per_site, seed)
    for site in sites:
        train = capped[site_ids[capped] != site]
        # Every row of the held-out site is scored, capped or not.
        held_out = np.flatnonzero(site_ids == site)
        yield site, train, held_out


def _cap_rows(site_ids, max_per_site, seed):
    """Sorted row positions with at most ``max_per_site`` rows per site."""
    rng = np.random.default_rng(seed)
    keep = []
    for site in np.unique(site_ids):
        rows = np.flatnonzero(site_ids == site)
        # None means no cap.
        if max_per_site is not None and len(rows) > max_per_site:
            rows = rng.choice(rows, max_per_site, replace=False)
        keep.append(rows)
    return np.sort(np.concatenate(keep))


def _fit_transformers(windows, backgrounds, site_ids, rows, seed):
    """Fit the PCA, k-means, and neighbor features on training rows."""
    fitted = {}
    if windows is not None:
        # PCA and k-means on the spectrum shape of the training mixtures.
        fitted["shape"] = FittedFeatures(seed=seed).fit(windows[rows], site_ids[rows])
    if backgrounds is not None:
        if windows is None:
            raise ValueError("backgrounds needs windows as well.")
        # The neighbor pool is the vessel-free background of each training row.
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
        # With site ids, neighbors come from other sites. Without, from all.
        blocks.append(fitted["neighbors"].transform(windows, site_ids))
    return np.column_stack(blocks)


def _design_names(fitted, feature_names):
    """Column names in the same order as ``_design``."""
    names = list(feature_names)
    for key in ("shape", "neighbors"):
        if key in fitted:
            names += list(fitted[key].feature_names_)
    return names


def _baseline_predictions(features, train_share, feature_names, n_rows):
    """Predictions from the no-learning baselines for ``n_rows`` rows."""
    out = {
        # The starter notebook submits a constant.
        "constant_zero": np.zeros(n_rows),
        "train_mean": np.full(n_rows, float(np.mean(train_share))),
    }
    # The physics heuristic: energy above the band floor over total energy.
    if "above_floor_share" in feature_names:
        column = list(feature_names).index("above_floor_share")
        out["above_floor"] = np.clip(features[:, column], 0.0, 1.0)
    return out


def error_summary(share, predicted):
    """RMSE, MAE, and the Zindi score for one set of rows."""
    share = np.asarray(share, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((predicted - share) ** 2)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(predicted - share))),
        # The leaderboard formula, higher is better.
        "score": 1.0 - rmse,
    }


def error_breakdown(predictions, by, names=None):
    """Error of each model within each group of held-out rows."""
    # Prediction columns are named pred_<model>.
    if names is None:
        names = [c for c in predictions.columns if c.startswith("pred_")]
    else:
        names = [f"pred_{n}" for n in names]
    # ``by`` is a column of ``predictions`` or one group label per row.
    groups = predictions[by] if isinstance(by, str) else pd.Series(by, name="group")
    rows = []
    for group, part in predictions.groupby(groups.to_numpy(), observed=True):
        for column in names:
            # One output row per group and model.
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
    """Fit the models and return site-held-out scores."""
    # features is (n, n_features) from window_features, share is (n,).
    features = np.asarray(features, dtype=np.float64)
    share = np.asarray(share, dtype=np.float64)
    site_ids = np.asarray(site_ids)
    if features.ndim != 2:
        raise ValueError("features must have shape (n, n_features).")
    if not (len(features) == len(share) == len(site_ids)):
        raise ValueError("features, share, and site_ids must have the same length.")
    # windows are the (n, 15, 25) mixtures behind the feature rows.
    if windows is not None and len(windows) != len(features):
        raise ValueError("windows must have one window per feature row.")
    # backgrounds are the clean windows behind each mixture, same row order.
    if backgrounds is not None and len(backgrounds) != len(features):
        raise ValueError("backgrounds must have one window per feature row.")
    if feature_names is None:
        if features.shape[1] == len(FEATURE_NAMES):
            feature_names = list(FEATURE_NAMES)
        else:
            feature_names = [f"f{i}" for i in range(features.shape[1])]
    if models is None:
        models = make_models(seed)
    # background_features are window_features of each background with the
    # same hidden minutes as its mixture. They are scored, never fit on.
    score_negatives = backgrounds is not None and background_features is not None

    names = list(BASELINE_NAMES) + list(models)
    if "above_floor_share" not in feature_names:
        names.remove("above_floor")
    # One held-out prediction per row and model, filled in fold by fold.
    held_out = {name: np.full(len(share), np.nan) for name in names}
    # Predictions on the same rows with no vessel added, for the ROC-AUC.
    negative = {name: np.full(len(share), np.nan) for name in models}
    fit_seconds = dict.fromkeys(models, 0.0)
    predict_seconds = dict.fromkeys(models, 0.0)

    for site, train, test in site_folds(site_ids, max_per_site, seed):
        # PCA, k-means, and neighbors see training sites only.
        fitted = _fit_transformers(windows, backgrounds, site_ids, train, seed)
        x_train = _design(fitted, features[train], _rows(windows, train), site_ids[train])
        x_test = _design(fitted, features[test], _rows(windows, test), site_ids[test])
        if score_negatives:
            # Hide the same minutes in the background as in its mixture.
            hidden = np.isnan(windows[test])
            quiet = np.where(hidden, np.nan, backgrounds[test])
            x_negative = _design(fitted, background_features[test], quiet, site_ids[test])
        baseline = _baseline_predictions(
            features[test], share[train], feature_names, len(test)
        )
        for name, values in baseline.items():
            held_out[name][test] = values
        for name, model in models.items():
            # A fresh copy per fold, so no fold sees another's fit.
            fold_model = clone(model)
            start = time.perf_counter()
            fold_model.fit(x_train, share[train])
            fit_seconds[name] += time.perf_counter() - start
            start = time.perf_counter()
            # A share is a fraction, so clip to 0 through 1.
            held_out[name][test] = np.clip(fold_model.predict(x_test), 0.0, 1.0)
            predict_seconds[name] += time.perf_counter() - start
            if score_negatives:
                negative[name][test] = np.clip(fold_model.predict(x_negative), 0.0, 1.0)

    # One row per training window with every model's held-out prediction.
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
        # The fit used for Test.csv sees every site.
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
        # Pooled error: every held-out row counts once.
        row.update(error_summary(share, held_out[name]))
        # Site-averaged error: every site counts once, whatever its size.
        per_site = by_site[by_site["model"] == name]
        row["site_mean_rmse"] = float(per_site["rmse"].mean())
        row["site_mean_score"] = 1.0 - row["site_mean_rmse"]
        row["worst_site_rmse"] = float(per_site["rmse"].max())
        if score_negatives:
            row["site_mean_auc"] = float(per_site["auc"].mean())
        if name in models:
            # The competition rules ask for runtime and model size.
            row["cv_fit_seconds"] = fit_seconds[name]
            row["predict_ms_per_window"] = 1000.0 * predict_seconds[name] / len(share)
            if final_fit:
                row["final_fit_seconds"] = final_seconds[name]
                row["model_megabytes"] = len(pickle.dumps(result["models"][name])) / 1e6
        rows.append(row)
    scores = pd.DataFrame(rows).sort_values("site_mean_rmse").reset_index(drop=True)
    result["scores"] = scores
    # The best model is the learned one with the lowest site-averaged RMSE.
    learned = scores[scores["model"].isin(list(models))]
    result["best"] = str(learned.iloc[0]["model"])
    return result


def _rows(windows, rows):
    """Pick rows from ``windows``, which may be left out."""
    return None if windows is None else windows[rows]


def _site_auc(predictions, site, name):
    """ROC-AUC of predicted share: a site's mixtures against its backgrounds."""
    column = f"background_{name}"
    # Baselines have no background predictions.
    if column not in predictions.columns:
        return np.nan
    part = predictions[predictions["site_id"] == site]
    scores = np.concatenate([part[f"pred_{name}"], part[column]])
    # Label 1 is a mixture with a vessel added, label 0 its background.
    labels = np.concatenate([np.ones(len(part)), np.zeros(len(part))])
    return float(roc_auc_score(labels, scores))


def predict_share(result, features, windows=None, model=None):
    """Predict ``share_added_vessel`` for new windows with the final fit."""
    name = result["best"] if model is None else model
    if name == "constant_zero":
        return np.zeros(len(features))
    if name == "train_mean":
        return np.full(len(features), result["train_mean"])
    if name not in result["models"]:
        raise KeyError(f"No final fit for {name!r}. Run with final_fit=True.")
    if result["transformers"] and windows is None:
        raise ValueError("This fit used windows, so pass windows here as well.")
    # No site ids here, so neighbor backgrounds come from every training site.
    x_new = _design(result["transformers"], np.asarray(features, dtype=np.float64), windows)
    return np.clip(result["models"][name].predict(x_new), 0.0, 1.0)


def make_submission(ids, share, score=None, path=None):
    """Build the Zindi submission table, and write it when ``path`` is given."""
    share = np.asarray(share, dtype=np.float64)
    # score_added_vessel only needs to rank windows, and a larger share
    # means a vessel is more likely, so the share itself is the default.
    score = share if score is None else np.asarray(score, dtype=np.float64)
    if len(ids) != len(share) or len(share) != len(score):
        raise ValueError("ids, share, and score must have the same length.")
    if not (np.isfinite(share).all() and np.isfinite(score).all()):
        raise ValueError("Predictions must be finite.")
    if share.min() < 0.0 or share.max() > 1.0 or score.min() < 0.0 or score.max() > 1.0:
        raise ValueError("Predictions must lie between 0 and 1.")
    # Same columns and order as sample_submission.csv.
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
    """Append one submission to the leaderboard log and return the whole log."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d") if date is None else date,
        "file": str(file),
        "model": model,
        # 1 - RMSE over all site-held-out rows, the leaderboard formula.
        "cv_score": _rounded(cv_score),
        "cv_site_mean_score": _rounded(cv_site_mean_score),
        # Left blank until Zindi shows the public score.
        "public_score": _rounded(public_score),
        "notes": notes,
    }
    # Write the header only when the log is new.
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return pd.read_csv(log_path)


def _rounded(value):
    """Four decimals as text, or an empty cell for a missing value."""
    return "" if value is None else f"{float(value):.4f}"
