"""
Strict 80/20 grouped train/test validation: burst classification and AD analysis
to confirm reported 100% accuracy is not due to leakage.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_loso_restricted_summary(repo_root: Path, output_dir: Path) -> None:
    """Summarize LOSO from loso_per_study.csv (filtered by min formulations); write loso_restricted_summary.csv."""
    loso_csv = repo_root / "loso_per_study.csv"
    if not loso_csv.is_file():
        return
    out_csv = output_dir / "loso_restricted_summary.csv"
    logger.info("LOSO restricted summary: loading %s", loso_csv)
    df = pd.read_csv(loso_csv)
    df["DOI"] = df["DOI"].astype(str).str.strip()
    logger.info("Rows loaded: %d", len(df))
    targets = ["Peppas_n", "Peppas_K", "Burst_24h"]
    filters = [
        ("All studies", 0),
        ("Studies with >= 3 formulations", 3),
        ("Studies with >= 5 formulations", 5),
    ]
    rows = []
    for target in targets:
        df_t = df[df["Target"] == target].copy()
        df_t["R2_clean"] = df_t["R2"].apply(lambda x: x if abs(x) < 100 else np.nan)
        for filter_label, min_forms in filters:
            sub = df_t[df_t["N_formulations"] >= max(min_forms, 1)].copy()
            n_studies = len(sub)
            n_forms = int(sub["N_formulations"].sum())
            if n_forms > 0:
                pooled_mae = float(np.average(sub["MAE"], weights=sub["N_formulations"]))
            else:
                pooled_mae = np.nan
            pooled_r2 = float(sub["R2_clean"].mean()) if n_studies > 0 else np.nan
            median_mae = float(sub["MAE"].median()) if n_studies > 0 else np.nan
            rows.append({
                "Target": target,
                "Filter": filter_label,
                "Min_Formulations": min_forms if min_forms > 0 else 1,
                "N_Studies": n_studies,
                "N_Formulations": n_forms,
                "Pooled_R2_approx": round(pooled_r2, 4) if not np.isnan(pooled_r2) else "NA",
                "Pooled_MAE": round(pooled_mae, 4) if not np.isnan(pooled_mae) else "NA",
                "Median_Study_MAE": round(median_mae, 4) if not np.isnan(median_mae) else "NA",
                "Note": (
                    "Pooled R2 excludes per-study values |R2| > 100 (numerical artifacts from "
                    "single-formulation held-out studies). Summarized from loso_per_study.csv."
                ),
            })
            logger.info("%s | %s — N_studies=%d N_forms=%d R2~=%.4f MAE=%.4f median_MAE=%.4f",
                        target, filter_label, n_studies, n_forms, pooled_r2, pooled_mae, median_mae)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    logger.info("Saved %s", out_csv)


def rigorous_validation(raw_path: str, initial_path: str, output_dir: Optional[str] = None) -> None:
    """Run 80/20 grouped split, fit on train only, report test accuracy and AD stats."""
    logger.info("=== RIGOROUS VALIDATION: LEAKAGE CHECK & AD ANALYSIS ===")
    logger.info("Loading and engineering features (no imputation yet)...")
    pipeline = PLGAPrecisionPipeline(raw_path, initial_path, output_dir or ".")
    pipeline.engineer_features()
    pipeline.engineer_targets()
    
    df = pipeline.df
    logger.info("Total Data Shape: %s", df.shape)
    logger.info("STEP 1: Strict 80/20 Grouped Split...")
    groups = df["Formulation Index"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    logger.info("Train Set: %d samples", len(train_df))
    logger.info("Test Set: %d samples", len(test_df))
    feature_cols = FEATURE_COLS
    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]
    
    logger.info("STEP 2: Preprocessing (Fit on Train ONLY)...")
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    logger.info("STEP 3: Burst Release Classification Check...")
    target = 'Burst_24h'
    
    # Create Classes
    y_train_val = train_df[target]
    y_test_val = test_df[target]
    
    # Filter NaNs
    train_mask = y_train_val.notna()
    test_mask = y_test_val.notna()
    
    X_train_b = X_train[train_mask]
    y_train_b = y_train_val[train_mask]
    X_test_b = X_test[test_mask]
    y_test_b = y_test_val[test_mask]
    
    def get_classes(y):
        y_class = np.zeros_like(y, dtype=int)
        y_class[(y >= 10) & (y < 40)] = 1
        y_class[y >= 40] = 2
        return y_class
        
    y_train_cls = get_classes(y_train_b)
    y_test_cls = get_classes(y_test_b)
    logger.info("Training Burst Classifier on %d samples...", len(y_train_cls))
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1,
        use_label_encoder=False, objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=RANDOM_SEED,
    )
    clf.fit(X_train_b, y_train_cls)
    
    train_preds = clf.predict(X_train_b)
    test_preds = clf.predict(X_test_b)
    
    train_acc = accuracy_score(y_train_cls, train_preds)
    test_acc = accuracy_score(y_test_cls, test_preds)
    logger.info("  -> Train Accuracy: %.4f", train_acc)
    logger.info("  -> Test Accuracy: %.4f (Previous reported: 1.0)", test_acc)
    logger.info("  -> Test Confusion Matrix:\n%s", confusion_matrix(y_test_cls, test_preds))
    if test_acc < 0.99:
        logger.info("  [CONCLUSION] Leakage Confirmed. 100%% was an artifact of improper validation.")
    else:
        logger.info("  [CONCLUSION] 100%% Accuracy held! Signal is extremely strong.")
    logger.info("STEP 4: Applicability Domain Analysis (Peppas_n)...")
    target_n = 'Peppas_n'
    
    y_train_n = train_df[target_n]
    y_test_n = test_df[target_n]
    
    mask_train = y_train_n.notna()
    mask_test = y_test_n.notna()
    
    X_train_n = X_train[mask_train]
    y_train_n = y_train_n[mask_train]
    X_test_n = X_test[mask_test]
    y_test_n = y_test_n[mask_test]
    
    reg = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1,
        objective="reg:squarederror", random_state=RANDOM_SEED,
    )
    reg.fit(X_train_n, y_train_n)
    
    y_pred_test = reg.predict(X_test_n)
    
    # Calculate AD Statistics
    # H = X_test_scaled * (X_train_scaled.T * X_train_scaled)^-1 * X_train_scaled.T ... tricky for test set
    # Standard approach for Test Set AD: Distance to training centroid or similar.
    # But sticking to Williams Plot logic: Leverage of test point regarding the HAT matrix of Training Data.
    # h_i = x_i^T (X_train^T X_train)^-1 x_i
    
    # Compute (X^T X)^-1 from Training Data
    try:
        XtX_inv = np.linalg.pinv(np.dot(X_train_n.T, X_train_n))
        
        # Compute Leverage for Test Points
        levs = []
        for i in range(len(X_test_n)):
            x_vec = X_test_n[i]
            # h = x^T * (XtX)^-1 * x
            h = np.dot(x_vec.T, np.dot(XtX_inv, x_vec))
            levs.append(h)
        levs = np.array(levs)
        
        # Calculate Warning Leverage h*
        p = X_train_n.shape[1]
        n_train = X_train_n.shape[0]
        h_star = 3 * p / n_train
        
        logger.info("  Warning Leverage h*: %.4f", h_star)
        
        # Identify "Safe" vs "Unsafe"
        # Standardized Residuals not available without "true" sigma, but we can use prediction error.
        residuals = y_test_n - y_pred_test
        # Standardize by Training RMSE or similar? 
        # Williams plot usually uses standardized residuals of the *model fit*.
        # Here we check if AE is lower in low leverage.
        
        safe_mask = levs < h_star
        unsafe_mask = ~safe_mask
        
        logger.info("  Test Points in Domain: %d / %d", sum(safe_mask), len(levs))
        if sum(safe_mask) > 0:
            r2_safe = r2_score(y_test_n[safe_mask], y_pred_test[safe_mask])
            mae_safe = mean_absolute_error(y_test_n[safe_mask], y_pred_test[safe_mask])
            logger.info("  [SAFE ZONE] R2: %.4f, MAE: %.4f", r2_safe, mae_safe)
        else:
            logger.info("  [SAFE ZONE] No points.")
        if sum(unsafe_mask) > 0:
            r2_unsafe = r2_score(y_test_n[unsafe_mask], y_pred_test[unsafe_mask])
            mae_unsafe = mean_absolute_error(y_test_n[unsafe_mask], y_pred_test[unsafe_mask])
            logger.info("  [UNSAFE ZONE] R2: %.4f, MAE: %.4f", r2_unsafe, mae_unsafe)
        else:
            logger.info("  [UNSAFE ZONE] No points.")
        if sum(safe_mask) > 0 and sum(unsafe_mask) > 0:
            logger.info("  -> Improvement in Safe Zone: +%.4f R2", r2_safe - r2_unsafe)
    except Exception as e:
        logger.exception("AD Analysis Failed: %s", e)

    if output_dir:
        out_dir = Path(output_dir)
    else:
        try:
            import config as _cfg
            out_dir = _cfg.OUTPUT_DIR
        except ImportError:
            out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_loso_restricted_summary(REPO_ROOT, out_dir)


if __name__ == "__main__":
    import config as _cfg
    raw = _cfg.DATA_DIR / _cfg.RAW_DATASET
    initial = _cfg.DATA_DIR / _cfg.INITIAL_DATASET
    if not raw.exists() or not initial.exists():
        raise FileNotFoundError("Place mp_dataset_processed.xlsx and mp_dataset_initial.xlsx in data/")
    rigorous_validation(str(raw), str(initial), str(_cfg.OUTPUT_DIR.resolve()))
