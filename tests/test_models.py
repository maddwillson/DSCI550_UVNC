"""Checks for the site-held-out models and the submission helpers."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uvnc.features import FEATURE_NAMES, window_features
from uvnc.models import (
    MODEL_NAMES,
    error_breakdown,
    error_summary,
    fit_site_held_out,
    log_submission,
    make_models,
    make_submission,
    predict_share,
    site_folds,
)
from uvnc.units import db_to_linear, linear_to_db


def make_table(n_sites=4, per_site=60, seed=0):
    """Small made-up training table: a bump in one minute sets the share."""
    rng = np.random.default_rng(seed)
    n = n_sites * per_site
    site_ids = np.repeat(np.arange(1, n_sites + 1), per_site)
    level = 60.0 + 5.0 * site_ids[:, None, None]
    backgrounds = level + rng.normal(0.0, 0.5, size=(n, 15, 25))
    added = np.zeros((n, 15, 25))
    minute = rng.integers(0, 15, size=n)
    strength = rng.uniform(0.0, 20.0, size=n)
    added[np.arange(n), minute, :] = db_to_linear(level[:, 0, :] + strength[:, None])
    mixture_linear = db_to_linear(backgrounds) + added
    shares = added.sum(axis=(1, 2)) / mixture_linear.sum(axis=(1, 2))
    return linear_to_db(mixture_linear), backgrounds, shares, site_ids


def small_models():
    models = make_models(seed=1)
    models["forest"].set_params(n_estimators=20, n_jobs=1)
    models["boosting"].set_params(max_iter=40, min_samples_leaf=5)
    models["tree"].set_params(min_samples_leaf=5)
    return models


def test_folds_hold_out_whole_sites():
    site_ids = np.repeat([3, 7, 9], [10, 4, 6])
    seen = []
    for site, train, held_out in site_folds(site_ids):
        assert set(site_ids[held_out]) == {site}
        assert site not in set(site_ids[train])
        assert len(train) + len(held_out) == len(site_ids)
        seen.append(site)
    assert seen == [3, 7, 9]


def test_cap_limits_training_rows_only():
    site_ids = np.repeat([1, 2, 3], [50, 5, 8])
    for site, train, held_out in site_folds(site_ids, max_per_site=6):
        assert np.bincount(site_ids[train]).max() <= 6
        assert len(held_out) == (site_ids == site).sum()


def test_one_site_is_rejected():
    try:
        list(site_folds(np.ones(5)))
    except ValueError:
        return
    raise AssertionError("one site cannot be held out from itself")


def test_error_summary():
    out = error_summary([0.0, 1.0], [0.0, 0.0])
    assert np.isclose(out["rmse"], np.sqrt(0.5))
    assert np.isclose(out["mae"], 0.5)
    assert np.isclose(out["score"], 1 - np.sqrt(0.5))


def test_fit_scores_every_model_and_baseline():
    mixtures, backgrounds, shares, site_ids = make_table()
    features = window_features(mixtures)
    result = fit_site_held_out(features, shares, site_ids, models=small_models())

    scores = result["scores"].set_index("model")
    for name in MODEL_NAMES + ("constant_zero", "train_mean", "above_floor"):
        assert name in scores.index, name
    predictions = result["predictions"]
    assert len(predictions) == len(shares)
    columns = [c for c in predictions.columns if c.startswith("pred_")]
    assert not predictions[columns].isna().any().any(), "every row is held out once"
    assert predictions[columns].min().min() >= 0.0
    assert predictions[columns].max().max() <= 1.0
    assert len(result["by_site"]) == 4 * len(columns)
    assert scores.loc["boosting", "rmse"] < scores.loc["train_mean", "rmse"]
    assert result["best"] in MODEL_NAMES
    assert result["feature_names"] == list(FEATURE_NAMES)


def test_fold_fitted_features_and_auc():
    mixtures, backgrounds, shares, site_ids = make_table()
    features = window_features(mixtures)
    result = fit_site_held_out(
        features,
        shares,
        site_ids,
        windows=mixtures,
        backgrounds=backgrounds,
        background_features=window_features(backgrounds),
        models={"boosting": small_models()["boosting"]},
    )
    assert len(result["feature_names"]) == 79 + 5 + 4 + 2
    assert result["feature_names"][-2:] == ["expected_peak", "excess_peak"]
    scores = result["scores"].set_index("model")
    assert 0.5 < scores.loc["boosting", "site_mean_auc"] <= 1.0

    predicted = predict_share(result, features[:7], mixtures[:7])
    assert predicted.shape == (7,)
    assert predicted.min() >= 0.0 and predicted.max() <= 1.0
    try:
        predict_share(result, features[:7])
    except ValueError:
        pass
    else:
        raise AssertionError("a fit with windows needs windows to predict")


def test_breakdown_by_group():
    predictions = pd.DataFrame(
        {
            "site_id": [1, 1, 2, 2],
            "share": [0.0, 0.0, 1.0, 1.0],
            "pred_a": [0.0, 0.0, 0.5, 0.5],
        }
    )
    out = error_breakdown(predictions, "site_id").set_index("group")
    assert np.isclose(out.loc[1, "rmse"], 0.0)
    assert np.isclose(out.loc[2, "rmse"], 0.5)
    assert list(out["n"]) == [2, 2]


def test_submission_matches_sample_format():
    root = Path(__file__).resolve().parents[1]
    sample = pd.read_csv(root / "sample_submission.csv")
    share = np.linspace(0.0, 1.0, len(sample))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sub" / "submission.csv"
        make_submission(sample["ID"], share, path=path)
        written = pd.read_csv(path)
    assert list(written.columns) == list(sample.columns)
    assert list(written["ID"]) == list(sample["ID"])
    assert np.allclose(written["share_added_vessel"], share)
    assert np.allclose(written["score_added_vessel"], share)

    for bad in ([0.2, 1.2], [0.2, np.nan]):
        try:
            make_submission(["a", "b"], bad)
        except ValueError:
            continue
        raise AssertionError("out-of-range or missing predictions must be rejected")


def test_log_appends_rows():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "leaderboard_log.csv"
        log_submission(path, "a.csv", "ridge", 0.7, 0.68, date="2026-11-23")
        log = log_submission(path, "b.csv", "boosting", 0.8, 0.79, public_score=0.75, notes="x")
    assert list(log["file"]) == ["a.csv", "b.csv"]
    assert np.isnan(log.loc[0, "public_score"])
    assert np.isclose(log.loc[1, "public_score"], 0.75)
    assert np.isclose(log.loc[1, "cv_score"], 0.8)


if __name__ == "__main__":
    test_folds_hold_out_whole_sites()
    test_cap_limits_training_rows_only()
    test_one_site_is_rejected()
    test_error_summary()
    test_fit_scores_every_model_and_baseline()
    test_fold_fitted_features_and_auc()
    test_breakdown_by_group()
    test_submission_matches_sample_format()
    test_log_appends_rows()
    print("model checks passed")
