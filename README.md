# S&P 500 stock segmentation (2021–2024)

6BUIS017W coursework. RFM observation date: 31 December 2024.

- `Stock_Segmentation_2021_2024.ipynb` is the Colab notebook.
- `data_part1.zip` and `data_part2.zip` contain the Yahoo Finance histories and Wikipedia constituent snapshot.
- `outputs.zip` contains the feature tables, exclusions, cluster assignments, figures and sensitivity results.

To rerun from the saved data, use Python 3.11 or 3.12 and run these commands from the repository folder.

```bash
python -m zipfile -e data_part1.zip data
python -m zipfile -e data_part2.zip data
python -m pip install -r requirements.txt
python analysis.py --root . --offline --selection reviewed_selection.json
```
