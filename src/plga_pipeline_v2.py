"""PLGA release prediction pipeline."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Optional config (allows running from src/ or repo root)
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


class PLGAPrecisionPipeline:
    """Build features, targets, grouped CV metrics, and validation outputs."""

    def __init__(self, raw_path: str, initial_path: str, output_dir: Optional[str] = None) -> None:
        self.raw_df = pd.read_excel(raw_path)
        self.initial_df = pd.read_excel(initial_path)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.df = None
        self.targets: List[str] = ["Peppas_n", "Peppas_K", "Burst_24h"]
        self.models = {}
        self.results = {}
        self.ad_metrics = {}

    def _calculate_rdkit_descriptors(self, smiles: str) -> List[float]:
        """Compute RDKit descriptors from SMILES."""
        if pd.isna(smiles):
            return [np.nan] * 6
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return [np.nan] * 6
            return [
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.ExactMolWt(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol),
            ]
        except Exception:
            return [np.nan] * 6

    def engineer_features(self) -> None:
        """Build one row per Formulation Index with RDKit and polymer-derived features."""
        logger.info("STEP 1: Feature Engineering...")
        
        # One formulation row, preserving DOI for study-level validation.
        param_cols = ['Formulation Index', 'DOI', 'Drug MW', 'Drug LogP', 'Drug TPSA', 'Polymer MW', 'LA/GA',
                      'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency']
        
        existing_cols = [c for c in param_cols if c in self.raw_df.columns]
        formulation_params = self.raw_df[existing_cols].drop_duplicates(subset='Formulation Index')
        
        if 'Drug SMILES' in self.initial_df.columns:
            formulation_params = formulation_params.merge(
                self.initial_df[['Formulation Index', 'Drug SMILES']].drop_duplicates(subset='Formulation Index'),
                on='Formulation Index', 
                how='left'
            )
        
        logger.info("  - Calculating RDKit descriptors...")
        # Recalculate descriptors from SMILES rather than relying on reported values.
        if 'Drug SMILES' in formulation_params.columns:
            formulation_params['RDKit_Descriptors'] = formulation_params['Drug SMILES'].apply(self._calculate_rdkit_descriptors)
            
            rdkit_cols = ['MolLogP', 'TPSA', 'ExactMolWt', 'NumHDonors', 'NumHAcceptors', 'RotatableBonds']
            rdkit_df = pd.DataFrame(formulation_params['RDKit_Descriptors'].tolist(), columns=rdkit_cols)
            formulation_params = formulation_params.reset_index(drop=True)
            rdkit_df = rdkit_df.reset_index(drop=True)
            
            formulation_params = pd.concat([formulation_params, rdkit_df], axis=1)
        
        logger.info("  - Engineering Polymer features...")
        if 'LA/GA' in formulation_params.columns:
             formulation_params['LA_GA_numeric'] = formulation_params['LA/GA']
        else:
             formulation_params['LA_GA_numeric'] = 1.0 # default
             
        if 'Polymer MW' not in formulation_params.columns:
             if 'Polymer Mw' in self.initial_df.columns:
                  formulation_params = formulation_params.merge(self.initial_df[['Formulation Index', 'Polymer Mw']].drop_duplicates(subset='Formulation Index'), on='Formulation Index')
                  formulation_params['Polymer MW'] = formulation_params['Polymer Mw']
        
        formulation_params['Polymer_Mw_Clean'] = formulation_params['Polymer MW'].fillna(1).replace(0, 1)
        
        formulation_params['Hydrophilicity_Index'] = (1.0 / (formulation_params['LA_GA_numeric'] + 1e-6)) * (1.0 / formulation_params['Polymer_Mw_Clean'])
        
        logger.debug("formulation_params columns: %s", formulation_params.columns.tolist())
        logger.debug("formulation_params shape: %s", formulation_params.shape)
        self.df = formulation_params

    def engineer_targets(self) -> None:
        """Compute Peppas and burst targets."""
        logger.info("STEP 2: Target Engineering (Mechanistic)...")
        results = []
        
        grouped = self.raw_df.groupby('Formulation Index')
        
        for idx, group in grouped:
            if len(group) < 5: continue # Filter < 5 points
            
            group = group.sort_values('Time')
            t = group['Time'].values
            y = group['Release'].values
            
            try:
                burst_24 = np.interp(24, t, y)
            except:
                burst_24 = np.nan
                
            # Korsmeyer-Peppas: Mt/Minf = K * t^n
            # Fit to first 60% release
            mask = y <= 0.60
            if mask.sum() < 3: # Need points for fit
                # Fallback: fit all if max < 60, or take first N
                if len(y) >= 3:
                    t_fit = t[:5]
                    y_fit = y[:5]
                else:
                    t_fit = np.array([])
                    y_fit = np.array([])
            else:
                t_fit = t[mask]
                y_fit = y[mask]
                
            # Log-Log Fit: log(Q) = log(K) + n*log(t)
            # Avoid t=0, y=0
            valid = (t_fit > 0) & (y_fit > 0)
            t_log = np.log(t_fit[valid]) if valid.any() else np.array([])
            y_log = np.log(y_fit[valid]) if valid.any() else np.array([])

            n, K = np.nan, np.nan
            if len(t_log) >= 3:
                try:
                    slope, intercept = np.polyfit(t_log, y_log, 1)
                    n = slope
                    K = np.exp(intercept)
                    
                    # Retain high empirical Peppas n values for diagnostic modeling.
                    # Values above 2 are outside the conventional interpretability window
                    # but are retained to match the paired manuscript modeling export.
                except:
                    pass

            results.append({
                'Formulation Index': idx,
                'Peppas_n': n,
                'Peppas_K': K,
                'Burst_24h': burst_24
            })
                
        target_df = pd.DataFrame(results)
        logger.debug("target_df shape: %s", target_df.shape)
        if target_df["Formulation Index"].duplicated().any():
            logger.warning("Duplicates in target_df Formulation Index")
        self.df = self.df.merge(target_df, on="Formulation Index", how="inner")
        logger.info("  - Features + Targets merged. Final shape: %s", self.df.shape)

    def build_ensemble(self) -> None:
        """Define the stacked ensemble."""
        logger.info("STEP 3: Stacked Ensemble Architecture...")
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
        xgb_mod = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1,
            objective="reg:squarederror", random_state=RANDOM_SEED,
        )
        svr = SVR(kernel="rbf", C=10, gamma="scale")
        
        # Level 1 Meta-Learner
        ridge = Ridge(alpha=1.0)
        
        # Stacking Ensemble
        self.ensemble = StackingRegressor(
            estimators=[('rf', rf), ('xgb', xgb_mod), ('svr', svr)],
            final_estimator=ridge,
            cv=5, # Internal CV for stacking
            n_jobs=-1
        )
        
    def train_and_validate(self) -> None:
        """Run grouped CV for the manuscript targets and save pooled metrics."""
        logger.info("STEP 3b: Training & Validation (10-Fold Grouped)...")
        feature_cols = FEATURE_COLS
        X = self.df[feature_cols].copy().values
        # X = X.fillna(X.mean()) # LEAKAGE FIX: Do not fill globally

        
        groups = self.df['Formulation Index']
        gkf = GroupKFold(n_splits=10)
        
        metrics_list = []
        
        for target in self.targets:
            logger.info("  - Training for %s...", target)
            y = self.df[target].copy()
            
            valid_mask = y.notna()
            X_curr = X[valid_mask]
            y_curr = y[valid_mask].values
            groups_curr = groups[valid_mask].values
            
            all_actual = []
            all_pred = []
            all_std = []
            all_indices = []
            all_groups = []
            
            for train_idx, test_idx in gkf.split(X_curr, y_curr, groups=groups_curr):
                X_train, X_test = X_curr[train_idx], X_curr[test_idx]
                y_train, y_test = y_curr[train_idx], y_curr[test_idx]
                
                # Fit preprocessing inside each grouped CV fold.
                imputer = SimpleImputer(strategy='mean')
                X_train_imp = imputer.fit_transform(X_train)
                X_test_imp = imputer.transform(X_test)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imp)
                X_test_scaled = scaler.transform(X_test_imp)
                
                rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
                xgb_mod = xgb.XGBRegressor(
                    n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1,
                    objective="reg:squarederror", random_state=RANDOM_SEED,
                )
                svr = SVR(kernel="rbf", C=10, gamma="scale")
                
                stack = StackingRegressor(
                    estimators=[('rf', rf), ('xgb', xgb_mod), ('svr', svr)],
                    final_estimator=Ridge(alpha=1.0),
                    cv=3,
                    n_jobs=-1
                )
                
                stack.fit(X_train_scaled, y_train)
                preds = stack.predict(X_test_scaled)
                
                # Use base-learner spread as an uncertainty proxy.
                base_preds = []
                for name, est in stack.named_estimators_.items():
                    base_preds.append(est.predict(X_test_scaled))
                
                ensemble_std = np.std(base_preds, axis=0)
                
                all_actual.extend(y_test)
                all_pred.extend(preds)
                all_std.extend(ensemble_std)
                all_indices.extend(test_idx)
                all_groups.extend(groups_curr[test_idx])
                
            all_actual = np.array(all_actual)
            all_pred = np.array(all_pred)
            all_std = np.array(all_std)
            
            r2 = r2_score(all_actual, all_pred)
            mae = mean_absolute_error(all_actual, all_pred)
            rmse = np.sqrt(mean_squared_error(all_actual, all_pred))
            logger.info("    %s: R2=%.3f, MAE=%.3f", target, r2, mae)
            
            metrics_list.append({'Target': target, 'R2': r2, 'MAE': mae, 'RMSE': rmse})
            
            self.results[target] = pd.DataFrame({
                'Formulation Index': all_groups,
                'Actual': all_actual,
                'Predicted': all_pred,
                'Residuals': all_actual - all_pred,
                'Uncertainty': all_std
            })
            
            # Refit on full data for export
            final_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', self.ensemble)
            ])
            final_pipe.fit(X_curr, y_curr)
            self.models[target] = final_pipe

        logger.info("  - Running Burst Classification...")
        # Binary high-burst class follows the manuscript threshold.
        burst_y = self.df['Burst_24h'].copy()
        valid_b = burst_y.notna()
        X_b = X[valid_b]
        y_b_val = burst_y[valid_b].values
        groups_b = groups[valid_b].values
        
        imp_b = SimpleImputer(strategy='mean')
        X_b = imp_b.fit_transform(X_b)

        
        y_class = np.zeros_like(y_b_val, dtype=int)
        y_class[y_b_val >= 0.20] = 1
        
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1,
            use_label_encoder=False, objective="binary:logistic",
            eval_metric="logloss", random_state=RANDOM_SEED,
        )
        class_preds = cross_val_predict(clf, X_b, y_class, cv=gkf, groups=groups_b)
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
        acc = accuracy_score(y_class, class_preds)
        macro_f1 = f1_score(y_class, class_preds, average="macro")
        precision, recall, _, support = precision_recall_fscore_support(
            y_class, class_preds, labels=[0, 1], zero_division=0
        )
        logger.info("    Burst Classification Accuracy: %.3f", acc)
        
        self.results['Burst_Class'] = {
            'Formulation Index': groups_b,
            'Actual': y_class,
            'Predicted': class_preds,
            'ConfusionMatrix': confusion_matrix(y_class, class_preds)
        }
        metrics_list.append({'Target': 'Burst_Class', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'Accuracy': acc})
        pd.DataFrame([
            {"Class": int(cls), "Precision": precision[i], "Recall": recall[i], "Support": int(support[i])}
            for i, cls in enumerate([0, 1])
        ] + [
            {"Class": "macro", "Precision": np.nan, "Recall": np.nan, "Support": int(len(y_class)), "Accuracy": acc, "Macro_F1": macro_f1}
        ]).to_csv(self.output_dir / "burst_classification_metrics.csv", index=False)

        final_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1,
            use_label_encoder=False, objective="binary:logistic",
            eval_metric="logloss", random_state=RANDOM_SEED,
        )
        final_clf.fit(X_b, y_class)
        self.models["Burst_Class"] = final_clf

        out = self.output_dir
        pd.DataFrame(metrics_list).to_csv(out / "performance_metrics.csv", index=False)
        joblib.dump(self.models, out / "Final_Model.joblib")

    def analyze_applicability_domain(self) -> None:
        """Compute Williams-plot leverage and residual summaries."""
        logger.info("STEP 4: Applicability Domain (Williams Plot)...")
        feature_cols = FEATURE_COLS
        for target in self.targets:
            res_df = self.results[target]
            valid_idx = self.df[self.df[target].notna()].index
            X_raw = self.df.loc[valid_idx, feature_cols].values
            X_imp = SimpleImputer(strategy="mean").fit_transform(X_raw)
            X_scaled = StandardScaler().fit_transform(X_imp)
            
            # H = diag(X (X.T X)^-1 X.T)
            # Use pseudo-inverse for stability
            try:
                H = np.dot(X_scaled, np.linalg.pinv(np.dot(X_scaled.T, X_scaled)))
                H = np.dot(H, X_scaled.T)
                leverage = np.diagonal(H)
            except:
                leverage = np.zeros(len(X_raw))
                
            residuals = res_df['Residuals'].values
            std_resid = residuals / (np.std(residuals) + 1e-6)
            
            res_df['Leverage'] = leverage
            res_df['Std_Residual'] = std_resid
            
            # Warning Leverage h* = 3p/n
            p = X_raw.shape[1]
            n = X_raw.shape[0]
            h_star = 3 * p / n
            
            high_cert = res_df[(res_df['Leverage'] < h_star)]
            low_cert = res_df[(res_df['Leverage'] >= h_star)]
            
            acc_full = r2_score(res_df["Actual"], res_df["Predicted"])
            logger.info("  %s AD Analysis: Full R2=%.4f", target, acc_full)
            if len(high_cert) > 0:
                acc_safe = r2_score(high_cert["Actual"], high_cert["Predicted"])
                mae_safe = mean_absolute_error(high_cert["Actual"], high_cert["Predicted"])
                logger.info("    Safe Zone (Low Lev) R2: %.4f (MAE: %.4f, N=%d)", acc_safe, mae_safe, len(high_cert))
            if len(low_cert) > 0:
                acc_unsafe = r2_score(low_cert["Actual"], low_cert["Predicted"])
                mae_unsafe = mean_absolute_error(low_cert["Actual"], low_cert["Predicted"])
                logger.info("    High Leverage Zone R2: %.4f (MAE: %.4f, N=%d)", acc_unsafe, mae_unsafe, len(low_cert))
                
            self.ad_metrics[target] = {"h_star": h_star, "data": res_df}

        ad_rows = []
        for target, metrics in self.ad_metrics.items():
            data = metrics["data"]
            h_star = metrics["h_star"]
            safe = data[data["Leverage"] < h_star]
            high = data[data["Leverage"] >= h_star]
            ad_rows.append({
                "Target": target,
                "h_star": h_star,
                "N": len(data),
                "N_in_domain": len(safe),
                "N_high_leverage": len(high),
                "R2": r2_score(data["Actual"], data["Predicted"]),
                "MAE": mean_absolute_error(data["Actual"], data["Predicted"]),
                "RMSE": np.sqrt(mean_squared_error(data["Actual"], data["Predicted"])),
            })
        pd.DataFrame(ad_rows).to_csv(self.output_dir / "applicability_domain_metrics.csv", index=False)

def run_pipeline(raw_path: str, initial_path: str, output_dir: str) -> None:
    """Run the pipeline and write result CSVs."""
    try:
        import config as _c
        _c.set_seeds()
    except ImportError:
        pass
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipeline = PLGAPrecisionPipeline(raw_path, initial_path, str(out))
    pipeline.engineer_features()
    pipeline.engineer_targets()
    pipeline.build_ensemble()
    pipeline.train_and_validate()
    pipeline.analyze_applicability_domain()
    logger.info("=== Precision Pipeline Complete ===")

    all_res = []
    for target, res in pipeline.results.items():
        if isinstance(res, pd.DataFrame):
            df = res.copy()
            df["Target"] = target
            all_res.append(df)
        elif isinstance(res, dict) and "Actual" in res:
            base = {"Actual": res["Actual"], "Predicted": res["Predicted"], "Target": target}
            if "Formulation Index" in res:
                base["Formulation Index"] = res["Formulation Index"]
            df = pd.DataFrame(base)
            all_res.append(df)
    if all_res:
        pd.concat(all_res).to_csv(out / "all_predictions_and_uncertainty.csv", index=False)
        logger.info("Exported all_predictions_and_uncertainty.csv to %s", out)


if __name__ == "__main__":
    import config as _cfg
    data_dir = _cfg.DATA_DIR
    out_dir = _cfg.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = data_dir / _cfg.RAW_DATASET
    initial = data_dir / _cfg.INITIAL_DATASET
    if not raw.exists() or not initial.exists():
        raise FileNotFoundError("Place mp_dataset_processed.xlsx and mp_dataset_initial.xlsx in data/")
    run_pipeline(str(raw), str(initial), str(out_dir))
