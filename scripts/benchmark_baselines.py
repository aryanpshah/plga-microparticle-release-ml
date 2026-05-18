"""Run baseline model benchmarks."""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.plga_pipeline_v2 import PLGAPrecisionPipeline

try:
    import config as _config
    RANDOM_SEED = _config.RANDOM_SEED
    FEATURE_COLS = _config.FEATURE_COLS
except ImportError:
    RANDOM_SEED = 42
    FEATURE_COLS = [
        "Drug MW", "Drug LogP", "Drug TPSA", "MolLogP", "TPSA", "ExactMolWt",
        "NumHDonors", "NumHAcceptors", "RotatableBonds",
        "Polymer MW", "LA_GA_numeric", "Hydrophilicity_Index",
        "Particle Size", "Drug Loading Capacity", "Drug Encapsulation Efficiency",
    ]

warnings.filterwarnings("ignore")


def run_benchmarks(
    raw_path: str,
    initial_path: str,
    output_dir: Optional[str] = None,
) -> None:
    """Run grouped-CV benchmarks."""
    out = Path(output_dir) if output_dir else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    import logging
    logging.getLogger(__name__).info("=== Running Rigorous Benchmarks ===")
    pipeline = PLGAPrecisionPipeline(raw_path, initial_path, str(out))
    pipeline.engineer_features()
    pipeline.engineer_targets()
    df = pipeline.df
    targets = ["Peppas_n", "Peppas_K", "Burst_24h"]
    feature_cols = FEATURE_COLS
    X = df[feature_cols]
    groups = df["Formulation Index"]
    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1,
            objective="reg:squarederror", random_state=RANDOM_SEED,
        ),
    }
    rf_ens = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    xgb_ens = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1,
        objective="reg:squarederror", random_state=RANDOM_SEED,
    )
    svr_ens = SVR(kernel="rbf", C=10, gamma="scale")
    stack = StackingRegressor(
        estimators=[("rf", rf_ens), ("xgb", xgb_ens), ("svr", svr_ens)],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )
    models["StackedEnsemble"] = stack
    results = []
    gkf = GroupKFold(n_splits=10)
    for target in targets:
        y = df[target]
        valid_mask = y.notna()
        X_curr = X[valid_mask]
        y_curr = y[valid_mask]
        groups_curr = groups[valid_mask]
        for name, model in models.items():
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", model),
            ])
            try:
                preds = cross_val_predict(pipe, X_curr, y_curr, cv=gkf, groups=groups_curr, n_jobs=-1)
                r2 = r2_score(y_curr, preds)
                mae = mean_absolute_error(y_curr, preds)
                rmse = np.sqrt(mean_squared_error(y_curr, preds))
                results.append({"Target": target, "Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Benchmark %s failed: %s", name, e)
    res_df = pd.DataFrame(results)
    res_df.to_csv(out / "benchmark_results.csv", index=False)


if __name__ == "__main__":
    import config as _cfg
    raw = _cfg.DATA_DIR / _cfg.RAW_DATASET
    initial = _cfg.DATA_DIR / _cfg.INITIAL_DATASET
    if not raw.exists() or not initial.exists():
        raise FileNotFoundError("Place mp_dataset_processed.xlsx and mp_dataset_initial.xlsx in data/")
    run_benchmarks(str(raw), str(initial), str(_cfg.OUTPUT_DIR))
