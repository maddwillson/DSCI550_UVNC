# Handoff

Joshua is building the labeled training windows on `joshua/training-pipeline`.
Pragya, Madeleine, and Levi can start from the stubs in `src/uvnc/` once this
branch is pushed. Everyone still shares the report and the presentation.

## Data contract

- One window is a NumPy array of shape `(15, 25)`: 15 minute positions by 25 third-octave bands.
- A table of windows is `(n, 15, 25)`. `windows_from_frame` builds that from a test-style table.
- Flat rows match `Test.csv`: `ID` plus 375 columns named `m01_TOL_31.6` through `m15_TOL_7943.3`.
- Within each minute, columns run from low frequency to high frequency.
- Stored values are decibel band levels. A blank cell is missing, so it stays `NaN`. It is not silence and it is not zero energy.
- Add energy only after `db_to_linear`. The conversion is `10 ** (dB / 10)`. Convert back with `linear_to_db`.
- `minute_band_power` turns one audio channel into minute-by-band linear power, shape `(minutes, 25)`. A band that reaches past the Nyquist rate stays NaN. A trailing partial minute is dropped. Keep VTUAD files out of git.
- `passage_pattern` keeps the 15 minutes around the loudest minute. Its finite cells sum to 1, and the returned minute number is the closest approach inside that span.
- `mix_pattern` adds that pattern times `added_energy` in linear power. A missing background cell stays missing, and passage energy on that cell is left out of both sums.
- `vessel_share` is the training label `share_added_vessel`. It is the sum of added linear energy divided by the sum of mixture linear energy, over the same observed minute-band cells. A missing minute drops out of both sums. Call `vessel_share` again after `mask_random_minutes`.
- Build training backgrounds from the South African site files. `Test.csv` is for final predictions and for checking the 375-column layout. Fit scalers, PCA, k-means, and models on training rows.

## Owners

| Person | Module | What to implement |
| --- | --- | --- |
| Joshua Lee | `units.py`, `windows.py`, then `passage.py`, `mixing.py`, `pipeline.py` | Band-level conversion, window shape, VTUAD conversion, vessel mixing, and a single entry point |
| Pragya Maruwada | `cleaning.py` | Drop bad rows, cut 15-minute background windows, Spark summaries, EDA |
| Madeleine Willson | `features.py` | Per-band statistics, broadband level, PCA, k-means, nearest-neighbor background estimate |
| Levi Glaser | `models.py` | Ridge, tree, forest, boosting, site-held-out cross-validation, leaderboard log |

Call these functions rather than reaching into another person's file:

- Pragya's clean windows should use the same column names as `Test.csv`, plus a site id. Joshua will pass that table to `windows_from_frame`.
- Madeleine and Levi should read `build_labeled_windows` for the training table. It returns mixtures of shape `(n, 15, 25)`, `share_added_vessel` of shape `(n,)`, and one metadata dict per window (`seed`, closest minute, dropped minutes). Each row is one `build_labeled_window` call. Pass a stack of patterns for recordings. Leave patterns out for the fallback envelope and ship spectrum.
- Madeleine should turn those mixture windows into one feature row each.
- Levi should take Madeleine's feature matrix and the shares. Hold out whole sites in cross-validation.

`score_added_vessel` is Levi's diagnostic column. Its leaderboard weight is zero.
