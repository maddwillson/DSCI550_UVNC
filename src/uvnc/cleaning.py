"""Site cleaning, 15-minute windows, Spark summaries, and EDA.

Owner: Pragya Maruwada.
Input: raw site recordings.
Output: one test-style row per 15-minute background window, plus a site id.
Blank cells stay NaN. Joshua mixes vessel passages into these windows.
"""


def clean_background_windows(site_frame):
    """Return one row per clean 15-minute background window.

    Parameters
    ----------
    site_frame : pandas.DataFrame
        Raw measurements for one recording site.

    Returns
    -------
    pandas.DataFrame
        Columns match ``Test.csv`` (``ID`` plus 375 band levels in dB),
        with an added site-id column. Minute positions run from 1 to 15,
        and bands within a minute run from low frequency to high frequency.
    """
    raise NotImplementedError("Pragya: cleaning, Spark summaries, and EDA.")
