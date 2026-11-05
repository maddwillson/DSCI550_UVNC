"""Site cleaning, 15-minute windows, Spark summaries, and EDA.

Owner: Pragya Maruwada.
Input: raw site recordings (one Parquet file per station, from
``South_African_Real_Acoustics.zip``).
Output: one test-style row per 15-minute background window, plus a site id.
Blank cells stay NaN. Joshua mixes vessel passages into these windows.

Notes on the raw data (see the station package's DATA_DICTIONARY.md):

* The 25 common bands (31.6 Hz to 7943.3 Hz) are finite in every raw row, so
  blanks in ``Test.csv`` come from hidden minutes, not from the recorders.
  Hidden minutes are added later by ``mask_random_minutes``, not here.
* Rows are not always one full minute (``dur``), and some recorders freeze and
  repeat a flat spectrum, so rows are screened before windows are cut.
* A window is only cut from consecutive minutes of ONE recorder. Recorders
  differ in level by tens of dB, so recorder is kept as a column.
* Values stay in dB. Nothing is averaged or imputed here.
"""

import re

import numpy as np
import pandas as pd

N_MINUTES = 15
BAND_CENTRES = [
    "31.6", "39.8", "50.1", "63.1", "79.4", "100", "125.9", "158.5",
    "199.5", "251.2", "316.2", "398.1", "501.2", "631", "794.3", "1000",
    "1258.9", "1584.9", "1995.3", "2511.9", "3162.3", "3981.1", "5011.9",
    "6309.6", "7943.3",
]
RAW_BANDS = [f"TOL_{c}" for c in BAND_CENTRES]
WINDOW_COLUMNS = [
    f"m{m:02d}_TOL_{c}" for m in range(1, N_MINUTES + 1) for c in BAND_CENTRES
]
META_COLUMNS = ["site_id", "site", "recorder", "t_start"]

# Screening thresholds (documented so the report can cite them).
MIN_DURATION_S = 55.0     # a row shorter than this is not a full minute
MAX_DURATION_S = 65.0     # rows up to 360 s exist; they are not one minute
MIN_SPREAD_DB = 6.0       # flat spectra (frozen recorder) are not sound
MIN_LEVEL_DB = 20.0       # the lowest plausible band level in the real data
GAP_RANGE_S = (55.0, 65.0)  # allowed spacing between consecutive minutes


def site_id_from_name(site):
    """Return the leading numeric station id of a ``site`` string."""
    match = re.match(r"\s*(\d+)", str(site))
    if match is None:
        raise ValueError(f"No leading station id in site name {site!r}.")
    return int(match.group(1))


def qc_row_flags(site_frame):
    """Return a boolean DataFrame, one column per screening rule.

    ``True`` means the row passes that rule. Rules are kept separate so the
    EDA can report how many rows each one removes.
    """
    levels = site_frame[RAW_BANDS].to_numpy(dtype=np.float64)
    finite = np.isfinite(levels).all(axis=1)
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(levels, axis=1) - np.nanmin(levels, axis=1)
        floor = np.nanmin(levels, axis=1)
    dur = site_frame["dur"].to_numpy(dtype=np.float64)
    flags = pd.DataFrame(index=site_frame.index)
    flags["duration_ok"] = (dur >= MIN_DURATION_S) & (dur <= MAX_DURATION_S)
    flags["finite_25_bands"] = finite
    flags["spread_ok"] = finite & (spread >= MIN_SPREAD_DB)
    flags["level_ok"] = finite & (floor >= MIN_LEVEL_DB)
    # A row identical to the previous row of the same recorder is a frozen
    # reading, not a new minute.
    ordered = site_frame.sort_values(["recorder", "t"])
    same_recorder = ordered["recorder"].eq(ordered["recorder"].shift())
    same_values = (
        ordered[RAW_BANDS].diff().abs().max(axis=1).eq(0) & same_recorder
    )
    flags["not_frozen"] = ~same_values.reindex(site_frame.index).fillna(False)
    flags["row_ok"] = flags.all(axis=1)
    return flags


def clean_background_windows(site_frame, stride=N_MINUTES):
    """Return one row per clean 15-minute background window.

    Parameters
    ----------
    site_frame : pandas.DataFrame
        Raw measurements for one recording site. Needs ``site``,
        ``recorder``, ``t``, ``dur`` and the 25 ``TOL_<centre>`` columns.
    stride : int, default 15
        Minutes between window starts. 15 gives non-overlapping windows.

    Returns
    -------
    pandas.DataFrame
        Columns match ``Test.csv`` (``ID`` plus 375 band levels in dB),
        with an added site-id column. Minute positions run from 1 to 15,
        and bands within a minute run from low frequency to high frequency.
        Extra provenance columns: ``site``, ``recorder``, ``t_start``.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1 minute.")
    needed = {"site", "recorder", "t", "dur", *RAW_BANDS}
    missing = needed - set(site_frame.columns)
    if missing:
        raise ValueError(f"site_frame is missing columns: {sorted(missing)}")
    if site_frame.empty:
        return _empty_output()

    flags = qc_row_flags(site_frame)
    good = site_frame.loc[flags["row_ok"]].sort_values(["recorder", "t"])
    good = good.drop_duplicates(["recorder", "t"])
    if good.empty:
        return _empty_output()

    # A new segment starts when the recorder changes or when consecutive good
    # rows are not one minute apart (a gap, or a dropped bad row, in between).
    gap = good["t"].diff().dt.total_seconds()
    new_segment = (
        good["recorder"].ne(good["recorder"].shift())
        | ~gap.between(*GAP_RANGE_S)
    )
    segment = new_segment.cumsum().to_numpy()
    position = good.groupby(segment).cumcount().to_numpy()
    length = good.groupby(segment)["t"].transform("size").to_numpy()

    # Keep whole windows only: positions inside a full multiple of the stride
    # that still leave 15 minutes after them.
    starts = (position % stride == 0) & (position + N_MINUTES <= length)
    start_idx = np.flatnonzero(starts)
    if start_idx.size == 0:
        return _empty_output()

    levels = good[RAW_BANDS].to_numpy(dtype=np.float64)
    offsets = np.arange(N_MINUTES)
    block = levels[start_idx[:, None] + offsets[None, :]]  # (n, 15, 25)
    first = good.iloc[start_idx]
    site_ids = first["site"].map(site_id_from_name).to_numpy()
    recorder_codes = first["recorder"].str.slice(0, 8).to_numpy()
    stamps = first["t"].dt.strftime("%Y%m%dT%H%M%S").to_numpy()

    out = pd.DataFrame(block.reshape(len(start_idx), -1), columns=WINDOW_COLUMNS)
    out.insert(0, "ID", [f"s{s:02d}_{r}_{t}" for s, r, t in
                         zip(site_ids, recorder_codes, stamps)])
    out.insert(1, "site_id", site_ids.astype(int))
    out.insert(2, "site", first["site"].to_numpy())
    out.insert(3, "recorder", first["recorder"].to_numpy())
    out.insert(4, "t_start", first["t"].to_numpy())
    return out.reset_index(drop=True)


def _empty_output():
    columns = ["ID", *META_COLUMNS, *WINDOW_COLUMNS]
    return pd.DataFrame(columns=columns)


def load_site(path, columns=None):
    """Read one station Parquet file with only the columns cleaning needs."""
    base = ["site", "recorder", "t", "dur", *RAW_BANDS]
    return pd.read_parquet(path, columns=columns or base)


def clean_all_sites(paths, stride=N_MINUTES):
    """Clean every station file and stack the windows into one table."""
    frames = [clean_background_windows(load_site(p), stride) for p in paths]
    frames = [f for f in frames if len(f)]
    if not frames:
        return _empty_output()
    return pd.concat(frames, ignore_index=True)


def row_loss_table(site_frame):
    """Rows removed by each rule for one site, for the EDA notebook."""
    flags = qc_row_flags(site_frame)
    table = {"rows": len(flags)}
    for rule in flags.columns:
        if rule == "row_ok":
            continue
        table[f"fail_{rule}"] = int((~flags[rule]).sum())
    table["rows_kept"] = int(flags["row_ok"].sum())
    return pd.Series(table)


def spark_site_summaries(paths, spark=None):
    """Per-site and per-recorder summaries computed with Spark.

    Reads the Parquet files in parallel and returns a pandas DataFrame with
    row counts, duration and level statistics and the share of rows passing
    the main screening rules. Needs ``pyspark`` and a Java runtime.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    owns_session = spark is None
    if owns_session:
        spark = (
            SparkSession.builder.master("local[*]")
            .appName("uvnc-cleaning")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
    try:
        frame = spark.read.parquet(*[str(p) for p in paths])
        band_cols = [F.col(f"`{b}`") for b in RAW_BANDS]
        frame = frame.withColumn("band_max", F.greatest(*band_cols))
        frame = frame.withColumn("band_min", F.least(*band_cols))
        frame = frame.withColumn("band_mean_db", sum(band_cols) / len(band_cols))
        frame = frame.withColumn("spread", F.col("band_max") - F.col("band_min"))
        summary = frame.groupBy("site", "recorder").agg(
            F.count("*").alias("rows"),
            F.min("t").alias("first_t"),
            F.max("t").alias("last_t"),
            F.avg("dur").alias("mean_dur_s"),
            F.avg(F.when((F.col("dur") >= MIN_DURATION_S)
                         & (F.col("dur") <= MAX_DURATION_S), 1.0)
                  .otherwise(0.0)).alias("share_full_minute"),
            F.avg(F.when(F.col("spread") < MIN_SPREAD_DB, 1.0)
                  .otherwise(0.0)).alias("share_flat"),
            F.avg("band_mean_db").alias("mean_level_db"),
            F.expr("percentile_approx(band_mean_db, 0.5)").alias("median_level_db"),
            F.stddev("band_mean_db").alias("sd_level_db"),
            F.min("band_min").alias("min_db"),
            F.max("band_max").alias("max_db"),
        )
        return summary.orderBy("site", "recorder").toPandas()
    finally:
        if owns_session:
            spark.stop()