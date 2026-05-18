"""
PLGA release prediction pipeline: feature engineering, stacked ensemble regression,
applicability domain analysis, and burst release classification.

Consumes Excel datasets (processed + initial), outputs metrics, figures, and model artifact.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})


class PLGAPrecisionPipeline:
    """
    End-to-end pipeline: features from formulation + SMILES, targets from release curves,
    10-fold group CV, stacking ensemble, applicability domain, visualizations.
    """

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
        """Compute MolLogP, TPSA, ExactMolWt, NumHDonors, NumHAcceptors, RotatableBonds from SMILES."""
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
        
        # 1. Consolidate Data
        # Extract unique formulation parameters from raw_df (which has Time/Release repeats)
        # We want one row per "Formulation Index" with all parameters
        # Columns to keep from raw_df:
        param_cols = ['Formulation Index', 'Drug MW', 'Drug LogP', 'Drug TPSA', 'Polymer MW', 'LA/GA', 
                      'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency']
        
        # Check which columns actually exist in raw_df
        existing_cols = [c for c in param_cols if c in self.raw_df.columns]
        formulation_params = self.raw_df[existing_cols].drop_duplicates(subset='Formulation Index')
        
        # Merge with initial_df to get Drug SMILES if needed (or other metadata)
        # initial_df has SMILES
        if 'Drug SMILES' in self.initial_df.columns:
            formulation_params = formulation_params.merge(
                self.initial_df[['Formulation Index', 'Drug SMILES']].drop_duplicates(subset='Formulation Index'),
                on='Formulation Index',
                how='left'
            )
        
        # 2. Chemical Featurization (RDKit)
        logger.info("  - Calculating RDKit descriptors...")
        # Recalculate Descriptors from SMILES to ensure we have "Computed" values vs "Reported" values
        # And specifically exact features like NumHDonors which might not be in raw_df
        if 'Drug SMILES' in formulation_params.columns:
            formulation_params['RDKit_Descriptors'] = formulation_params['Drug SMILES'].apply(self._calculate_rdkit_descriptors)
            
            rdkit_cols = ['MolLogP', 'TPSA', 'ExactMolWt', 'NumHDonors', 'NumHAcceptors', 'RotatableBonds']
            rdkit_df = pd.DataFrame(formulation_params['RDKit_Descriptors'].tolist(), columns=rdkit_cols)
            # Concat or assign? assign is safer to align indices
            # reset_index to be sure
            formulation_params = formulation_params.reset_index(drop=True)
            rdkit_df = rdkit_df.reset_index(drop=True)
            
            formulation_params = pd.concat([formulation_params, rdkit_df], axis=1)
        
        # 3. Polymer Engineering
        logger.info("  - Engineering Polymer features...")
        # LA/GA Ratio
        # Check if 'LA/GA' is in params
        if 'LA/GA' in formulation_params.columns:
             formulation_params['LA_GA_numeric'] = formulation_params['LA/GA']
        else:
             formulation_params['LA_GA_numeric'] = 1.0 # default
             
        # Polymer MW
        # raw_df has 'Polymer MW', initial_df has 'Polymer Mw'. 
        # We used 'Polymer MW' from raw_df in param_cols.
        if 'Polymer MW' not in formulation_params.columns:
             if 'Polymer Mw' in self.initial_df.columns:
                  # Merge it in
                  formulation_params = formulation_params.merge(
                      self.initial_df[['Formulation Index', 'Polymer Mw']].drop_duplicates(subset='Formulation Index'),
                      on='Formulation Index'
                  )
                  formulation_params['Polymer MW'] = formulation_params['Polymer Mw']
        
        # Clean MW
        formulation_params['Polymer_Mw_Clean'] = formulation_params['Polymer MW'].fillna(1).replace(0, 1)
        
        # Hydrophilicity Index
        formulation_params['Hydrophilicity_Index'] = (1.0 / (formulation_params['LA_GA_numeric'] + 1e-6)) * (1.0 / formulation_params['Polymer_Mw_Clean'])
        
        logger.debug("formulation_params columns: %s", formulation_params.columns.tolist())
        logger.debug("formulation_params shape: %s", formulation_params.shape)
        self.df = formulation_params

    def engineer_targets(self) -> None:
        """Compute Peppas_n, Peppas_K, Burst_24h from release curves (Korsmeyer-Peppas fit; time in hours)."""
        logger.info("STEP 2: Target Engineering (Mechanistic)...")
        results = []
        
        grouped = self.raw_df.groupby('Formulation Index')
        
        for idx, group in grouped:
            if len(group) < 3: continue # Filter < 3 points
            
            group = group.sort_values('Time')
            t = group['Time'].values
            y = group['Release'].values
            
            # Target C: 24h Burst
            # Interpolate or find closest
            try:
                # Simple linear interp for 24h
                burst_24 = np.interp(24, t, y)
            except:
                burst_24 = np.nan
                
            # Korsmeyer-Peppas: Mt/Minf = K * t^n
            # Fit to first 60% release
            mask = y <= 0.60
            n, K = np.nan, np.nan
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
            if valid.sum() >= 3:
                t_log = np.log(t_fit[valid])
                y_log = np.log(y_fit[valid])
                try:
                    slope, intercept = np.polyfit(t_log, y_log, 1)
                    n = slope
                    K = np.exp(intercept)
                    if n < 0 or n > 2:
                        n = np.nan
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
        """Define stacking ensemble: RF + XGBoost + SVR -> Ridge meta-learner."""
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
        """10-fold group CV, mean imputation per fold, stacking; burst binary classification."""
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
            
            # Remove NaNs
            valid_mask = y.notna()
            X_curr = X[valid_mask]
            y_curr = y[valid_mask].values
            groups_curr = groups[valid_mask].values
            
            # Storage
            all_actual = []
            all_pred = []
            all_std = []
            all_formulation_indices = []
            
            # Manual CV Loop for Uncertainty
            for train_idx, test_idx in gkf.split(X_curr, y_curr, groups=groups_curr):
                X_train, X_test = X_curr[train_idx], X_curr[test_idx]
                y_train, y_test = y_curr[train_idx], y_curr[test_idx]
                
                # Preprocessing (Impute -> Scale)
                # Fit Imputer on Train ONLY
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
                
                # Uncertainty (Std Dev of Base Learners)
                # Access fitted base learners
                base_preds = []
                for name, est in stack.named_estimators_.items():
                    base_preds.append(est.predict(X_test_scaled))
                
                # Add overall variance
                ensemble_std = np.std(base_preds, axis=0)
                
                all_actual.extend(y_test)
                all_pred.extend(preds)
                all_std.extend(ensemble_std)
                all_formulation_indices.extend(groups_curr[test_idx])
                
            # Metrics
            all_actual = np.array(all_actual)
            all_pred = np.array(all_pred)
            all_std = np.array(all_std)
            
            r2 = r2_score(all_actual, all_pred)
            mae = mean_absolute_error(all_actual, all_pred)
            rmse = np.sqrt(mean_squared_error(all_actual, all_pred))
            logger.info("    %s: R2=%.3f, MAE=%.3f", target, r2, mae)
            
            metrics_list.append({'Target': target, 'R2': r2, 'MAE': mae, 'RMSE': rmse})
            
            self.results[target] = pd.DataFrame({
                'Formulation Index': all_formulation_indices,
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
        # Binary classes aligned to manuscript-sync: Burst_24h > 0.20 is high burst risk.
        burst_y = self.df['Burst_24h'].copy()
        # Drop NaNs
        # Drop NaNs
        valid_b = burst_y.notna()
        # Note: X already has NaNs now, need to handle them. 
        # But for X_b we are slicing from X.
        X_b = X[valid_b] # .values was already done above
        y_b_val = burst_y[valid_b].values
        groups_b = groups[valid_b].values
        
        # Simple Mean Imputation for Classification check (or use pipeline)
        # Using a fresh imputer here to be safe, though global fill meant this was easy before.
        # Ideally we CV this too, but for block reporting we just need the estimator.
        # Let's rely on XGBoost handling NaNs natively or impute.
        imp_b = SimpleImputer(strategy='mean')
        X_b = imp_b.fit_transform(X_b)

        
        y_class = np.zeros_like(y_b_val, dtype=int)
        y_class[y_b_val > 0.20] = 1
        
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1,
            use_label_encoder=False, objective="binary:logistic",
            eval_metric="logloss", random_state=RANDOM_SEED,
        )
        class_preds = cross_val_predict(clf, X_b, y_class, cv=gkf, groups=groups_b)
        from sklearn.metrics import accuracy_score, confusion_matrix
        acc = accuracy_score(y_class, class_preds)
        logger.info("    Burst Classification Accuracy: %.3f", acc)
        
        self.results['Burst_Class'] = {
            'Formulation Index': groups_b,
            'Actual': y_class,
            'Predicted': class_preds,
            'ConfusionMatrix': confusion_matrix(y_class, class_preds)
        }
        metrics_list.append({'Target': 'Burst_Class', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'Accuracy': acc})

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
        """Williams plot: leverage (hat diagonal) vs standardized residuals; safe vs high-leverage R2."""
        logger.info("STEP 4: Applicability Domain (Williams Plot)...")
        feature_cols = FEATURE_COLS
        for target in self.targets:
            res_df = self.results[target]
            valid_idx = self.df[self.df[target].notna()].index
            X = self.df.loc[valid_idx, feature_cols].fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate Hat Matrix Diagonal (Leverage)
            # H = diag(X (X.T X)^-1 X.T)
            # Use pseudo-inverse for stability
            try:
                H = np.dot(X_scaled, np.linalg.pinv(np.dot(X_scaled.T, X_scaled)))
                H = np.dot(H, X_scaled.T)
                leverage = np.diagonal(H)
            except:
                leverage = np.zeros(len(X))
                
            # Standardized Residuals
            residuals = res_df['Residuals'].values
            std_resid = residuals / (np.std(residuals) + 1e-6)
            
            res_df['Leverage'] = leverage
            res_df['Std_Residual'] = std_resid
            
            # Define AD limits
            # Warning Leverage h* = 3p/n
            p = X.shape[1]
            n = X.shape[0]
            h_star = 3 * p / n
            
            # High Certainty Subset
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

    def generate_visualizations(self) -> None:
        """Save mechanism map, AD plot, feature importance, burst importance, AD paradox figures."""
        logger.info("STEP 5: Visualization...")
        out = self.output_dir
        
        # Figure 1: Mechanism Map (n values)
        plt.figure(figsize=(10, 6))
        
        # Determine mechanism zones
        # 0.43 = Fickian, 0.85 = Case II
        
        n_data = self.results['Peppas_n']
        curr_min = min(n_data['Actual'].min(), n_data['Predicted'].min())
        curr_max = max(n_data['Actual'].max(), n_data['Predicted'].max())
        
        sns.scatterplot(data=n_data, x='Actual', y='Predicted', alpha=0.6, edgecolor='w')
        plt.plot([curr_min, curr_max], [curr_min, curr_max], 'k--', lw=2)
        
        # Add zones
        plt.axvspan(0.3, 0.5, color='green', alpha=0.1, label='Fickian Diffusion')
        plt.axvspan(0.8, 0.9, color='orange', alpha=0.1, label='Case II Relaxation')
        
        plt.title('Mechanism Map: Predicted vs Actual Release Exponent (n)')
        plt.xlabel('Actual n')
        plt.ylabel('Predicted n')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "Figure1_MechanismMap.png")
        plt.close()

        # Figure 2: AD Plot for Burst
        plt.figure(figsize=(10, 6))
        ad_data = self.ad_metrics['Burst_24h']['data']
        h_star = self.ad_metrics['Burst_24h']['h_star']
        
        plt.scatter(ad_data['Leverage'], ad_data['Std_Residual'], alpha=0.6, c='blue')
        plt.axhline(y=3, color='r', linestyle='--')
        plt.axhline(y=-3, color='r', linestyle='--')
        plt.axvline(x=h_star, color='r', linestyle='--', label='Warning Leverage $h^*$')
        
        plt.fill_between([0, h_star], -3, 3, color='green', alpha=0.1, label='Applicability Domain')
        
        plt.title('Applicability Domain: Williams Plot (Burst Release)')
        plt.xlabel('Leverage (Model Space Distance)')
        plt.ylabel("Standardized Residuals")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "Figure2_ApplicabilityDomain.png")
        plt.close()

        rf_model = self.models["Peppas_n"].named_steps["model"].estimators_[0]
        importances = rf_model.feature_importances_
        feature_cols = FEATURE_COLS
        imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=False).head(10)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=imp_df, x="Importance", y="Feature", palette="viridis")
        plt.title("Top 10 Drivers of Release Mechanism (n)")
        plt.tight_layout()
        plt.savefig(out / "Figure3_FeatureImportance.png")
        plt.close()

        logger.info("Visualizations saved.")
        if "Burst_Class" in self.models:
            logger.info("  - Plotting Burst Classifier Importance...")
            clf = self.models["Burst_Class"]
            importances = clf.feature_importances_
            imp_df_burst = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=False).head(10)
            plt.figure(figsize=(10, 8))
            sns.barplot(data=imp_df_burst, x="Importance", y="Feature", palette="magma")
            plt.title("Drivers of Burst Release (Safety Failure Mode)")
            plt.tight_layout()
            plt.savefig(out / "Figure5_BurstImportance.png")
            plt.close()

        logger.info("  - Plotting AD Paradox...")
        ad_labels = []
        ad_scores = []
        for target in self.targets:
            if target not in self.ad_metrics:
                continue
            data = self.ad_metrics[target]["data"]
            h_star = self.ad_metrics[target]["h_star"]
            safe = data[data["Leverage"] < h_star]
            unsafe = data[data["Leverage"] >= h_star]
            if len(safe) > 10:
                ad_labels.append(f"{target}\n(Safe)")
                ad_scores.append(r2_score(safe["Actual"], safe["Predicted"]))
            if len(unsafe) > 10:
                ad_labels.append(f"{target}\n(High Lev)")
                ad_scores.append(r2_score(unsafe["Actual"], unsafe["Predicted"]))
        plt.figure(figsize=(10, 6))
        plt.bar(ad_labels, ad_scores, color=["green", "red"] * len(self.targets))
        plt.title("The AD Paradox: High Leverage Points often have Higher Predictability")
        plt.ylabel("R2 Score")
        plt.axhline(0, color="k", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(out / "Figure6_AD_Paradox.png")
        plt.close()


def run_pipeline(raw_path: str, initial_path: str, output_dir: str) -> None:
    """Run full pipeline and write all artifacts to output_dir."""
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
    pipeline.generate_visualizations()
    logger.info("=== Precision Pipeline Complete ===")

    all_res = []
    for target, res in pipeline.results.items():
        if isinstance(res, pd.DataFrame):
            df = res.copy()
            df["Target"] = target
            all_res.append(df)
        elif isinstance(res, dict) and "Actual" in res:
            formulation_index = res.get("Formulation Index")
            if formulation_index is None:
                formulation_index = [np.nan] * len(res["Actual"])
            df = pd.DataFrame({
                "Formulation Index": formulation_index,
                "Actual": res["Actual"],
                "Predicted": res["Predicted"],
                "Target": target,
            })
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
