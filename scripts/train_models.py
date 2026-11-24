"""Score the models with whole sites held out, then write a submission."""

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

# Edges of the true-share bins used in the error breakdown.
SHARE_BINS = [0.0, 0.01, 0.1, 0.3, 0.6, 0.9, 1.0]


def load_training_table(data_dir=ROOT / "data", cache=True):
    """Load mixtures, shares, sites, backgrounds, and both feature tables."""
    data_dir = Path(data_dir)
    # Written by scripts/build_labeled_windows.py. Stays out of git.
    labeled = np.load(data_dir / "labeled_windows.npz", allow_pickle=True)
    table = {
        "mixtures": labeled["mixtures"],
        "shares": labeled["shares"],
        "site_ids": labeled["site_ids"],
    }
    # The background behind each mixture, in the same row order.
    frame = pd.read_parquet(data_dir / "clean_windows.parquet")
    if len(frame) != len(table["shares"]):
        raise ValueError(
            "clean_windows.parquet and labeled_windows.npz have different row counts. "
            "Rebuild the labeled windows from this parquet file."
        )
    if not np.array_equal(frame["site_id"].to_numpy(), table["site_ids"]):
        raise ValueError("Background rows are not in the same site order as the mixtures.")
    table["backgrounds"] = windows_from_frame(frame)

    # window_features takes a couple of minutes on the full table, so cache it.
    cache_path = data_dir / "model_features.npz"
    if cache and cache_path.exists():
        saved = np.load(cache_path)
        # Reuse the cache only if it was built from these same shares.
        if saved["shares"].shape == table["shares"].shape and np.allclose(
            saved["shares"], table["shares"]
        ):
            table["features"] = saved["features"]
            table["background_features"] = saved["background_features"]
            return table
    # Hide the same minutes in each background as in its mixture.
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
    # 0 on the command line means no cap.
    result = run(table, max_per_site=args.max_per_site or None, seed=args.seed)

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    predictions = result["predictions"]
    # One row per model and baseline.
    result["scores"].to_csv(reports / "model_scores.csv", index=False, float_format="%.5f")
    # The same error for each held-out site.
    result["by_site"].to_csv(reports / "model_scores_by_site.csv", index=False, float_format="%.5f")
    # Error by number of hidden minutes, then by size of the true share.
    error_breakdown(predictions, "missing_minutes").to_csv(
        reports / "model_scores_by_missing.csv", index=False, float_format="%.5f"
    )
    error_breakdown(predictions, share_bin(predictions["share"])).to_csv(
        reports / "model_scores_by_share.csv", index=False, float_format="%.5f"
    )
    # One held-out prediction per window. Large, so it stays under data/.
    predictions.to_parquet(Path(args.data_dir) / "held_out_predictions.parquet", index=False)

    # Test.csv is only read to make predictions. Nothing is fit on it.
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
    # One line per submission file. public_score is filled in by hand later.
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
