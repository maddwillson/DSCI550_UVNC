"""Mix a vessel into the clean background windows and save the training table.

Run from the repo root, after ``scripts/build_clean_windows.py``:

    python scripts/build_labeled_windows.py

Output (stays out of git, under ``data/``):
    data/labeled_windows.npz
        mixtures          (n, 15, 25)  mixture windows in dB, after the minute mask
        shares            (n,)         share_added_vessel on the cells that remain
        site_ids          (n,)         same row order as mixtures and shares
        closest_minutes   (n,)         ship peak for that window, minutes 1 through 15
        added_energies    (n,)         linear passage power for that window
        seeds             (n,)         mask seed for that window
        n_drops           (n,)         hidden minutes for that window, 0 or 1 by default

``patterns`` is left unset, so every window uses the fallback envelope and
ship spectrum. By default each window gets its own closest minute, from 1
through 15, its own added energy, and either no hidden minute or one hidden
minute. The hidden minute is drawn here. It is not copied from which minute
``Test.csv`` leaves blank. ``--settings-seed`` makes that draw repeatable.

Added energy is total linear passage power, not decibels. The draw is
log-uniform from ``1e9`` to ``1e11``, around the old fixed total of ``1e10``.
Pass ``--closest-minute``, ``--added-energy``, ``--seed``, or ``--n-drop``
to hold that one setting fixed for every window.
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


def assign_window_settings(
    n_windows,
    closest_minute=None,
    added_energy=None,
    seed=None,
    n_drop=None,
    settings_seed=0,
):
    """Return one closest minute, energy, mask seed, and drop count per window.

    A setting left as ``None`` is drawn. A number is repeated for every
    window. Drawn drop counts are 0 or 1. Drawn closest minutes run from 1
    through 15. Drawn energies are log-uniform from ``1e9`` to ``1e11``.
    """
    if isinstance(n_windows, bool) or not isinstance(n_windows, (int, np.integer)):
        raise TypeError("n_windows must be an integer.")
    if int(n_windows) < 1:
        raise ValueError("n_windows must be at least 1.")
    n_windows = int(n_windows)
    rng = np.random.default_rng(settings_seed)
    if closest_minute is None:
        closest = rng.integers(1, 16, size=n_windows)
    else:
        closest = np.full(n_windows, int(closest_minute), dtype=np.int64)
    if added_energy is None:
        energy = 10 ** rng.uniform(9.0, 11.0, size=n_windows)
    else:
        energy = np.full(n_windows, float(added_energy), dtype=np.float64)
    if seed is None:
        seeds = np.arange(n_windows, dtype=np.int64) + np.int64(settings_seed)
    else:
        seeds = np.full(n_windows, int(seed), dtype=np.int64)
    if n_drop is None:
        drops = rng.integers(0, 2, size=n_windows)
    else:
        drops = np.full(n_windows, int(n_drop), dtype=np.int64)
    return closest, energy, seeds, drops


def label_windows(frame, closest_minute=8, added_energy=1e10, seed=0, n_drop=2):
    """Turn clean test-style rows into mixtures, shares, and site ids.

    Each setting may be one value for every window or one value per window.
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
    parser.add_argument("--closest-minute", type=int, default=None)
    parser.add_argument("--added-energy", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="one mask seed for every window; omit to draw one per window")
    parser.add_argument("--n-drop", type=int, default=None)
    parser.add_argument("--settings-seed", type=int, default=0,
                        help="seed for the per-window draw")
    args = parser.parse_args(argv)

    frame = pd.read_parquet(args.windows)
    closest, energy, seeds, drops = assign_window_settings(
        len(frame),
        closest_minute=args.closest_minute,
        added_energy=args.added_energy,
        seed=args.seed,
        n_drop=args.n_drop,
        settings_seed=args.settings_seed,
    )
    mixtures, shares, site_ids = label_windows(
        frame,
        closest_minute=closest,
        added_energy=energy,
        seed=seeds,
        n_drop=drops,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        mixtures=mixtures,
        shares=shares,
        site_ids=site_ids,
        closest_minutes=closest,
        added_energies=energy,
        seeds=seeds,
        n_drops=drops,
    )
    print(
        f"wrote {len(shares):,} windows  mixtures {mixtures.shape}  "
        f"sites {len(np.unique(site_ids))}  "
        f"peaks {len(np.unique(closest))}  "
        f"drop counts {dict(zip(*np.unique(drops, return_counts=True)))}  -> {out}"
    )


if __name__ == "__main__":
    main()
