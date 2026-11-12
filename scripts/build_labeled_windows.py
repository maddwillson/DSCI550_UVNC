"""Mix a vessel into the clean background windows and save the training table.

Run from the repo root, after ``scripts/build_clean_windows.py``:

    python scripts/build_labeled_windows.py

Output (stays out of git, under ``data/``):
    data/labeled_windows.npz
        mixtures   (n, 15, 25)  mixture windows in dB, after the minute mask
        shares     (n,)         share_added_vessel on the cells that remain
        site_ids   (n,)         same row order as mixtures and shares

``patterns`` is left unset, so every window uses the fallback envelope and
ship spectrum. One ``--seed`` blanks the same minutes on every window.

``--added-energy`` is total linear passage power, not decibels. The default
``1e10`` is about a third of a flat window at 77 dB, near the middle of
``Test.csv``. The same total on a quieter window is a larger share.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uvnc.pipeline import build_labeled_windows  # noqa: E402
from uvnc.windows import windows_from_frame  # noqa: E402


def label_windows(frame, closest_minute=8, added_energy=1e10, seed=0, n_drop=2):
    """Turn clean test-style rows into mixtures, shares, and site ids.

    Row order is preserved. ``site_ids[i]`` belongs to ``mixtures[i]``.
    """
    if "site_id" not in frame.columns:
        raise ValueError("frame is missing the site_id column.")
    if frame.empty:
        raise ValueError("frame has no windows.")
    table = frame.reset_index(drop=True)
    backgrounds = windows_from_frame(table)
    mixtures, shares, _metadata = build_labeled_windows(
        backgrounds,
        closest_minutes=closest_minute,
        added_energies=added_energy,
        seeds=seed,
        n_drops=n_drop,
    )
    site_ids = table["site_id"].to_numpy()
    if len(site_ids) != len(shares):
        raise RuntimeError("site_id row count does not match the labeled windows.")
    return mixtures, shares, site_ids


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--windows",
        default=str(ROOT / "data" / "clean_windows.parquet"),
        help="clean windows from scripts/build_clean_windows.py",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "labeled_windows.npz"),
        help="where to write mixtures, shares, and site_ids",
    )
    parser.add_argument("--closest-minute", type=int, default=8)
    parser.add_argument("--added-energy", type=float, default=1e10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-drop", type=int, default=2)
    args = parser.parse_args(argv)

    frame = pd.read_parquet(args.windows)
    mixtures, shares, site_ids = label_windows(
        frame,
        closest_minute=args.closest_minute,
        added_energy=args.added_energy,
        seed=args.seed,
        n_drop=args.n_drop,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        mixtures=mixtures,
        shares=shares,
        site_ids=site_ids,
    )
    print(
        f"wrote {len(shares):,} windows  mixtures {mixtures.shape}  "
        f"sites {len(np.unique(site_ids))}  -> {out}"
    )


if __name__ == "__main__":
    main()
