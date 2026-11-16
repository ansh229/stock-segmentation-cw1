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
- `Stock_Segmentation_Report.docx`: report with space for student details and the eventual public repository URL.

## Reviewed findings

486 of 503 shares retained; observation date 2024-12-31; 207 aligned weekly returns per share. Each four-model comparison has its highest silhouette at k=2 among k=2..8. Dendrograms support coarse splits but also show plausible finer divisions. The inertia curves have no decisive alternative elbow. Smaller tail groups appear at higher k. RFM Monetary-removal adjusted Rand index is 0.8734706266. No forecasting or diversification claim is made.

## Before submission

Repository: https://github.com/ansh229/stock-segmentation-cw1

Complete your student details in the report and review the final wording. The assessment brief's AI rules and any tutor permission still apply. The report records AI assistance; a declaration alone does not establish permission.

Market data are third-party data from Yahoo Finance and the list is from Wikipedia. No blanket licence is asserted over those source data. Source URLs and snapshot checksums are preserved in data/provenance.json.

The three ZIP archives preserve the original directory contents and checksums. Extract them before running the offline command.
