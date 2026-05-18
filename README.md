# Diagnosing Predictability Limits of In Vitro Drug Release from Published PLGA Microparticle Data

This repository contains the minimal code needed to run the transferred PLGA release workflow. It keeps only the core artifact-repo code paths and applies manuscript-sync logic where the two sources differ.

## Included files

- `config.py`
- `requirements.txt`
- `scripts/run_all.py`
- `src/plga_pipeline_v2.py`
- `src/rigorous_validation.py`

No manuscript assets, submission files, notebooks, figures, or dataset files are included here.

## Data access

The dataset is not redistributed in this repository. Download it from Mendeley Data and place these files in `data/`:

- `mp_dataset_processed.xlsx`
- `mp_dataset_initial.xlsx`

You can override `data/` and `outputs/` with the `DATA_DIR` and `OUTPUT_DIR` environment variables.

## Running the workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
```

`python scripts/run_all.py --fast` is also accepted for compatibility and runs the same minimal workflow.

## Manuscript-sync logic overrides

Where artifact-repo logic differed, this repository follows manuscript-sync:

- formulations with fewer than 3 release points are excluded
- the Peppas fit uses the first 60% of release with `Release <= 0.60`
- burst classification is binary with `Burst_24h > 0.20`
- exported prediction rows include `Formulation Index`

## Outputs

The workflow writes its outputs to `outputs/`, including:

- `performance_metrics.csv`
- `all_predictions_and_uncertainty.csv`
- generated figures from the core pipeline and applicability-domain analysis
