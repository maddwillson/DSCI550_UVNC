"""Clean every South African site and save the 15-minute windows to one file.

Run from the repo root:

    python scripts/build_clean_windows.py --data-dir /path/to/South_African_Real_Acoustics

Output (both stay out of git, see .gitignore):
    data/clean_windows.parquet          one row per window: ID, site_id, site,
                                        recorder, t_start, then the 375 mNN_TOL_
                                        columns (float32, dB, same names and order
                                        as Test.csv)
    data/clean_windows_summary.csv      windows per site before and after any cap

Options:
    --stride N                  minutes between window starts (default 15, no overlap)
    --max-windows-per-site N    keep at most N windows per site, chosen at random
                                with --seed (default: keep everything)
    --seed N                    seed for the cap (default 550)
    --sites 1 2 7               only these station ids
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uvnc.cleaning import (  # noqa: E402
    N_MINUTES, WINDOW_COLUMNS, clean_background_windows, load_site,
)


def station_files(data_dir, sites=None):
    """Return the station Parquet files, optionally only some station ids."""
    folder = Path(data_dir) / "acoustics"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {folder}")
    if sites:
        wanted = {int(s) for s in sites}
        files = [f for f in files if int(f.stem.split("_")[0]) in wanted]
        if not files:
            raise ValueError(f"No station files match ids {sorted(wanted)}")
    return files


def cap_windows(windows, max_windows, seed):
    """Keep at most ``max_windows`` windows, chosen at random, in time order."""
    if max_windows is None or len(windows) <= max_windows:
        return windows
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(windows), size=max_windows, replace=False))
    return windows.iloc[keep].reset_index(drop=True)


def build(data_dir, stride=N_MINUTES, max_windows_per_site=None, seed=550, sites=None):
    """Return (windows, summary) for the requested stations."""
    parts, rows = [], []
    for path in station_files(data_dir, sites):
        raw = load_site(path)
        site = raw["site"].iloc[0]
        windows = clean_background_windows(raw, stride=stride)
        capped = cap_windows(windows, max_windows_per_site, seed)
        rows.append({"site": site, "raw_rows": len(raw),
                     "windows_before_cap": len(windows), "windows_written": len(capped)})
        if len(capped):
            parts.append(capped)
        print(f"{site:<20} raw rows {len(raw):>9,}  windows {len(windows):>6,}  written {len(capped):>6,}")
    if not parts:
        raise RuntimeError("No windows were produced.")
    windows = pd.concat(parts, ignore_index=True)
    meta = windows.drop(columns=WINDOW_COLUMNS)
    levels = windows[WINDOW_COLUMNS].astype("float32")
    return pd.concat([meta, levels], axis=1), pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True,
                        help="folder that contains acoustics/ (the unzipped South African data)")
    parser.add_argument("--out", default=str(ROOT / "data" / "clean_windows.parquet"))
    parser.add_argument("--stride", type=int, default=N_MINUTES)
    parser.add_argument("--max-windows-per-site", type=int, default=None)
    parser.add_argument("--seed", type=int, default=550)
    parser.add_argument("--sites", nargs="*", default=None)
    args = parser.parse_args(argv)

    windows, summary = build(args.data_dir, args.stride, args.max_windows_per_site,
                             args.seed, args.sites)
    if not windows["ID"].is_unique:
        raise RuntimeError("Window IDs are not unique.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out, index=False)
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nwrote {len(windows):,} windows x {windows.shape[1]} columns -> {out}")
    print(f"wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()