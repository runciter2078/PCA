# PCA Analysis for SPY Data

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-%3E%3D1.3-orange)
![pandas](https://img.shields.io/badge/pandas-%E2%9C%93-150458)
![numpy](https://img.shields.io/badge/numpy-%E2%9C%93-013243)
![License](https://img.shields.io/badge/license-MIT-green)

This repository provides a Principal Component Analysis (PCA) pipeline in Python applied to SPY stock index data. It shows how to preprocess a mixed numeric/categorical dataset, scale it appropriately for PCA, select a sensible number of components, and interpret the resulting loadings — including a scree plot and a biplot.

> Disclaimer: this project is for educational and research purposes only. It does not constitute financial or investment advice.

## Repository structure

- `pca_analysis.py` - Full pipeline: data loading, preprocessing (categorical encoding, scaling), PCA fitting with automatic or manual component selection, explained variance reporting, loadings analysis and visual diagnostics.
- `README.md` — This file.
- `LICENSE` — Project license (MIT).

## Methodology

- Categorical variables are one-hot encoded instead of label-encoded with `pd.factorize`. Factorization assigns arbitrary integer codes to nominal categories, which implies an order relationship (e.g. category "2" being "between" categories "1" and "3") that does not actually exist and distorts the distances PCA relies on.
- Numeric features are standardized with `StandardScaler` by default. PCA operates on the covariance structure of the data, so on unscaled or min-max-scaled data, variables with larger raw variance end up dominating the components regardless of their actual relevance. `MinMaxScaler` is kept as an explicit, discouraged option for comparison.
- The number of components is selected automatically to reach a target cumulative explained variance (90% by default), instead of a fixed value chosen without justification. A fixed number can still be set explicitly.
- Loadings are filtered by absolute value rather than raw value: a loading of -0.35 contributes to a component exactly as much as +0.35, so the original one-sided filter silently discarded relevant negative loadings.
- Rows with missing values are dropped before scaling and PCA (both scalers and PCA fail or behave unpredictably on `NaN`), with a warning logged so the user is aware of any data loss.

## Requirements

- Python 3.9+

```text
pandas
numpy
scikit-learn>=1.3
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

## Data format

The script expects a CSV file (`SPYV3.csv` by default) containing, among others, the following columns, which are dropped before the analysis:

```text
FECHA, OPEN, MAX, MIN, CLOSE, CLASIFICADOR, FECHA.year, FECHA.day-of-month, FECHA.day-of-week
```

The following columns are treated as categorical and one-hot encoded; all remaining columns are treated as numeric and standardized:

```text
39, 41, 43, 168, 172
```

Adjust `DROP_COLUMNS` and `CATEGORICAL_COLUMNS` at the top of the script if you use a different dataset.

## Usage

```bash
python pca_analysis.py --data-path SPYV3.csv
```

Optional arguments:

```text
--scaler              "standard" (default) or "minmax"
--n-components         Fix the number of components manually
--variance-threshold   Target cumulative explained variance if --n-components is not set (default: 0.90)
--loading-threshold    Absolute loading threshold used to flag high-contribution variables (default: 0.30)
--output-dir           Output directory for artifacts (default: output)
```

Generated artifacts (under `output/`):

```text
explained_variance.csv         Explained and cumulative variance per component
scree_plot.png                 Bar + cumulative line chart of explained variance
pca_loadings.csv               Full loadings matrix (features x components)
pca_loadings_heatmap.png       Heatmap of the loadings matrix
high_loadings_PC-<i>.csv       Variables with |loading| above the threshold, per component
pca_biplot.png                 PC-1 vs PC-2 biplot with the top contributing variables
pca_scores.csv                 Transformed (PCA-projected) observations
```
## Sample dataset

A ready-to-use `SPYV3.csv` (5 years of daily SPY data with engineered technical indicators, generated 2026-08-13) is bundled in this repository so you can run the scripts above without depending on the original proprietary dataset. It was produced with `generate_spy_dataset.py`, which downloads real SPY data via `yfinance` and computes technical/categorical features under the same column names the scripts expect.

To regenerate it with fresh data:

```bash
pip install yfinance
python generate_spy_dataset.py --months 60 --output SPYV3.csv
```

Note: this is an independently engineered dataset, not a reconstruction of the original SPYV3.csv used when this project was first developed — see `generate_spy_dataset.py` for the exact feature definitions.

## Notes and limitations

- Standardizing is the recommended default for this kind of mixed-scale financial dataset; only switch to `--scaler minmax` if you have a specific reason to bound features to `[0, 1]` before PCA.
- `--variance-threshold` and `--loading-threshold` are reasonable starting points, not universal values — adjust them based on your own dataset and analysis goals.

## License

Distributed under the [MIT License](LICENSE).
