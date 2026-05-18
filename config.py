"""
Configuration for the PLGA pipeline and reproduction scripts.
Paths are relative to the repository root. Set DATA_DIR or OUTPUT_DIR
via environment variables to override.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", REPO_ROOT / "outputs"))
RAW_DATASET = "mp_dataset_processed.xlsx"
INITIAL_DATASET = "mp_dataset_initial.xlsx"
RANDOM_SEED = 42


def set_seeds() -> None:
    import random
    import numpy as np
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


FEATURE_COLS = [
    "Drug MW", "Drug LogP", "Drug TPSA", "MolLogP", "TPSA", "ExactMolWt",
    "NumHDonors", "NumHAcceptors", "RotatableBonds",
    "Polymer MW", "LA_GA_numeric", "Hydrophilicity_Index",
    "Particle Size", "Drug Loading Capacity", "Drug Encapsulation Efficiency",
]
