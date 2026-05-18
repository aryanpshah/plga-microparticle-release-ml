# Diagnosing Predictability Limits of In Vitro Drug Release from Published PLGA Microparticle Data

This repository provides the ML reproduction pipeline aligned to manuscript-sync result definitions.

---

## What this code reproduces

- **Figures:** Mechanism map (Peppas n), applicability domain (Williams plot), feature importance, burst classifier importance, AD paradox, uncertainty calibration, benchmark comparison, burst classification confusion matrix.
- **Tables / metrics:** Regression R² and MAE (Peppas n, Peppas K, Burst 24 h), burst classification accuracy, benchmark R² by model.
- **Validation:** Strict 80/20 grouped train/test split for burst classification (no leakage).

### manuscript-sync-aligned definitions

- Target engineering uses a minimum of **3 timepoints** per formulation.
- Peppas fitting uses release fraction cutoff **`Release <= 0.60`** for the first-60% region.
- Burst-risk classification uses the manuscript-sync binary definition: **high burst if `Burst_24h > 0.20`**.
- `all_predictions_and_uncertainty.csv` includes **Formulation Index** for row-level traceability.

---

## Environment setup

- **Python:** 3.10 (recommended; 3.9 minimum).
- **Commands:**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Accessing the data

The dataset is **not included** in this repository (e.g. for redistribution/licensing reasons). To run the pipeline you must obtain the data separately.

1. **Download the dataset** from Mendeley Data:  
   [https://data.mendeley.com/datasets/zzvtdrcy76/2](https://data.mendeley.com/datasets/zzvtdrcy76/2)  
   Use **Download All** to retrieve the full dataset.

2. **Create a `data/` folder** in this repository (if it does not exist) and place all downloaded files there. **Keep the original file names** from the dataset.

3. The code expects the main Excel files to be in `data/` with these names:
   - `mp_dataset_processed.xlsx`
   - `mp_dataset_initial.xlsx`  
   If the downloaded files use different names, rename them to the above (or adjust `config.py`).

4. Ensure `Time` in the processed file is in **hours** (Burst_24h is release at 24 h).

All scripts and the pipeline use the path **`data/`** for data files (via `config.DATA_DIR`). You can override it with the `DATA_DIR` environment variable or `--data-dir` when running `scripts/run_all.py`.

### Citation

If you use this dataset, please cite the original source.

> “Bao, Zeqing; Kim, Jongwhi; Kwok, Candice; Le Devedec, Frantz; Allen, Christine (2024), “A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles”, Mendeley Data, V2, doi: 10.17632/zzvtdrcy76.2

---

## How to run

From the repository root, with the venv activated:

```bash
python scripts/run_all.py
```

This single command runs the full pipeline, validation, benchmarks, and visualizations. Outputs are written to `outputs/` (created automatically).

**Optional:** `python scripts/run_all.py --fast` runs only the main pipeline and validation (no benchmarks or extra figures).

---

## Expected outputs

Generated under `outputs/`:

| Output | Description |
|--------|-------------|
| `performance_metrics.csv` | R², MAE, RMSE per target; burst classification accuracy |
| `all_predictions_and_uncertainty.csv` | Per-sample predictions and uncertainty with `Formulation Index` |
| `Figure1_MechanismMap.png` | Predicted vs actual Peppas n |
| `Figure2_ApplicabilityDomain.png` | Williams plot (Burst) |
| `Figure3_FeatureImportance.png` | Top drivers of release mechanism |
| `Figure5_BurstImportance.png` | Drivers of burst (safety) |
| `Figure6_AD_Paradox.png` | R² in safe vs high-leverage zones |
| (full run) `Figure4_UncertaintyCalibration.png`, `Figure5_BurstClassification.png`, `Figure6_Benchmarking.png`, `benchmark_results.csv` | Extra figures and benchmark table |

---

## Runtime and hardware

- CPU only; no GPU required.
- Full run: approximately 5–15 minutes. Fast run: approximately 2–5 minutes.

---

## Reproducibility

- **Determinism:** Random seed 42 is set in `config.py` and used for numpy, sklearn, and XGBoost. Train/validation/test splits are fixed.
- **Environment:** Python 3.10 and library versions are pinned in `requirements.txt`. Use the same environment for matching results.
