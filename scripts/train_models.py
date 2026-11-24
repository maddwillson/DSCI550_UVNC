"""Score the models with whole sites held out, then write a submission.

Run from the repo root, after ``scripts/build_labeled_windows.py``:

    python scripts/train_models.py

Reads (both stay out of git, under ``data/``):
    data/labeled_windows.npz      mixtures, shares, site_ids
    data/clean_windows.parquet    the background behind each mixture, same row order

Writes:
    data/model_features.npz                 cached ``window_features`` (out of git)
    data/held_out_predictions.parquet       one held-out prediction per window (out of git)
    reports/model_scores.csv                one row per model and baseline
    reports/model_scores_by_site.csv        the same error for each held-out site
    reports/model_scores_by_missing.csv     error by number of hidden minutes
    reports/model_scores_by_share.csv       error by size of the true share
    submissions/submission_<model>.csv      Zindi file for ``Test.csv``
    submissions/leaderboard_log.csv         one line per submission file

``Test.csv`` is only read to make predictions. Nothing is fit on it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uvnc.features import window_features  # noqa: E402
from uvnc.models import (  # noqa: E402
    error_breakdown,
    fit_site_held_out,
    log_submission,
    make_submission,
    predict_share,
)
from uvnc.windows import windows_from_frame  # noqa: E402

SHARE_BINS = [0.0, 0.01, 0.1, 0.3, 0.6, 0.9, 1.0]


def load_training_table(data_dir=ROOT / "data", cache=True):
    """Load mixtures, shares, sites, backgrounds, and both feature tables.

    ``window_features`` takes a couple of minutes on the full table, so the
    result is cached next to the data. The cache is rebuilt when the shares
    it was computed from no longer match.

    Returns
    -------
    dict
        ``mixtures`` and ``backgrounds`` of shape ``(n, 15, 25)``,
        ``shares`` and ``site_ids`` of shape ``(n,)``, ``features`` for the
        mixtures, and ``background_features`` for the backgrounds with the
        same minutes hidden as their mixture.
    """
    data_dir = Path(data_dir)
    labeled = np.load(data_dir / "labeled_windows.npz", allow_pickle=True)
    table = {
        "mixtures": labeled["mixtures"],
        "shares": labeled["shares"],
        "site_ids": labeled["site_ids"],
    }
    frame = pd.read_parquet(data_dir / "clean_windows.parquet")
    if len(frame) != len(table["shares"]):
        raise ValueError(
            "clean_windows.parquet and labeled_windows.npz have different row counts. "
            "Rebuild the labeled windows from this parquet file."
        )
    if not np.array_equal(frame["site_id"].to_numpy(), table["site_ids"]):
        raise ValueError("Background rows are not in the same site order as the mixtures.")
    table["backgrounds"] = windows_from_frame(frame)

    cache_path = data_dir / "model_features.npz"
    if cache and cache_path.exists():
        saved = np.load(cache_path)
        if saved["shares"].shape == table["shares"].shape and np.allclose(
            saved["shares"], table["shares"]
        ):
            table["features"] = saved["features"]
            table["background_features"] = saved["background_features"]
            return table
    hidden = np.isnan(table["mixtures"])
    table["features"] = window_features(table["mixtures"])
    table["background_features"] = window_features(
        np.where(hidden, np.nan, table["backgrounds"])
    )
    if cache:
        np.savez_compressed(
            cache_path,
            shares=table["shares"],
            features=table["features"],
            background_features=table["background_features"],
        )
    return table


def run(table, max_per_site=None, seed=550):
    """Cross-validate and fit on the loaded training table."""
    return fit_site_held_out(
        table["features"],
        table["shares"],
        table["site_ids"],
        windows=table["mixtures"],
        backgrounds=table["backgrounds"],
        background_features=table["background_features"],
        max_per_site=max_per_site,
        seed=seed,
    )


def share_bin(share):
    """Label each true share with its bin, for the error breakdown."""
    return pd.cut(share, SHARE_BINS, include_lowest=True).astype(str)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--test", default=str(ROOT / "Test.csv"))
    parser.add_argument("--reports", default=str(ROOT / "reports"))
    parser.add_argument("--submissions", default=str(ROOT / "submissions"))
    parser.add_argument(
        "--max-per-site",
        type=int,
        default=0,
        help="cap on training rows per site; 0, the default, uses every row",
    )
    parser.add_argument("--model", default=None, help="model to submit; default is the best one")
    parser.add_argument("--seed", type=int, default=550)
    parser.add_argument("--notes", default="", help="note for the leaderboard log")
    parser.add_argument("--no-log", action="store_true", help="skip the leaderboard log")
    args = parser.parse_args(argv)

    table = load_training_table(args.data_dir)
    result = run(table, max_per_site=args.max_per_site or None, seed=args.seed)

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    predictions = result["predictions"]
    result["scores"].to_csv(reports / "model_scores.csv", index=False, float_format="%.5f")
    result["by_site"].to_csv(reports / "model_scores_by_site.csv", index=False, float_format="%.5f")
    error_breakdown(predictions, "missing_minutes").to_csv(
        reports / "model_scores_by_missing.csv", index=False, float_format="%.5f"
    )
    error_breakdown(predictions, share_bin(predictions["share"])).to_csv(
        reports / "model_scores_by_share.csv", index=False, float_format="%.5f"
    )
    predictions.to_parquet(Path(args.data_dir) / "held_out_predictions.parquet", index=False)

    test = pd.read_csv(args.test)
    test_windows = windows_from_frame(test)
    name = result["best"] if args.model is None else args.model
    share = predict_share(result, window_features(test_windows), test_windows, model=name)
    out = Path(args.submissions) / f"submission_{name}.csv"
    make_submission(test["ID"], share, path=out)

    scores = result["scores"].set_index("model")
    print(result["scores"].round(4).to_string(index=False))
    print(f"\nsubmitting {name}: test share mean {share.mean():.3f}, median {np.median(share):.3f}")
    print(f"wrote {out}")
    if not args.no_log:
        log_submission(
            Path(args.submissions) / "leaderboard_log.csv",
            file=out.name,
            model=name,
            cv_score=scores.loc[name, "score"],
            cv_site_mean_score=scores.loc[name, "site_mean_score"],
            notes=args.notes,
        )


if __name__ == "__main__":
    main()
