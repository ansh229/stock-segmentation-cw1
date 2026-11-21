# Market analysis and stock segmentation 2021 to 2024

6BUIS017W CWK1 reproducible analysis. The source snapshot was retrieved on 16 November 2026. This repository uses the current S&P 500 constituent list with historical Yahoo Finance data; survivorship bias applies.

## Reproduce the saved analysis

Use Python 3.11 or 3.12 and a fresh virtual environment:

```bash
python -m zipfile -e data_part1.zip data
python -m zipfile -e data_part2.zip data
python -m zipfile -e outputs.zip outputs
pip install -r requirements.txt
python analysis.py --root . --offline --selection reviewed_selection.json
```

The executed Colab notebook is also included. Its default ROOT is `/content/stock_segmentation` on Colab. The offline CLI command above reuses this repository's `data` folder and verifies history checksums. To use a different location, set --root to that folder. Online acquisition uses yfinance and may be rate-limited; preserve the supplied snapshot.

## Contents

- `data_part1.zip` and `data_part2.zip` (extract to `data/`): source histories, constituent snapshot, retrieval provenance and failure log.
- `outputs.zip` (extract to `outputs/`): features, exclusions, corporate actions, candidate k diagnostics, standardised inputs, cluster assignments, sensitivity results, figures and package versions.
- `reviewed_selection.json`: reviewed k=2 for all four models.

## Results

The analysis retained 486 of 503 shares, with 207 aligned weekly returns per share and an RFM observation date of 31 December 2024. Across all four clustering comparisons, two clusters gave the highest silhouette score among the tested values of k (2–8). Removing Monetary from the Ward RFM model gave an adjusted Rand index of 0.8735.

## Data sources

The constituent list comes from Wikipedia, and the market histories come from Yahoo Finance. Source URLs and snapshot checksums are recorded in `data/provenance.json`.
