"""
Refinement visualizations: uncertainty calibration, burst confusion matrix,
benchmark comparison, Burst_24h scaling audit (CSVs + figures/Fig_5.png). Reads pipeline
outputs from output_dir, writes figures there.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

logger = logging.getLogger(__name__)


def run_burst_24h_scaling_audit(repo_root: Path, out: Path) -> None:
    """Audit Burst_24h scaling in predictions CSV; correct pct-scale outliers; write CSVs and Fig_5."""
    pred_csv = out / "all_predictions_and_uncertainty.csv"
    if not pred_csv.is_file():
        return
    fig_dir = repo_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = out / "burst24h_scaling_audit.csv"
    summary_csv = out / "burst24h_scaling_summary.csv"
    counts_csv = out / "burst24h_class_counts_after_audit.csv"
    fig5_png = fig_dir / "Fig_5.png"

    logger.info("Burst_24h scaling audit: loading %s", pred_csv)
    df_all = pd.read_csv(pred_csv)
    logger.info("  Total rows: %d", len(df_all))

    if "Target" not in df_all.columns:
        raise ValueError("Column 'Target' not found in CSV. Cannot filter to Burst_24h.")

    df = df_all[df_all["Target"] == "Burst_24h"].copy()
    logger.info("  Burst_24h rows (before dedup): %d", len(df))

    actual_col = None
    for candidate in ["y_true", "Actual", "Observed", "true", "actual"]:
        if candidate in df.columns:
            actual_col = candidate
            break

    if actual_col is None:
        logger.info("Available columns: %s", list(df.columns))
        raise ValueError("Could not find an actual/observed burst column. See log for columns.")

    logger.info("  Using actual column: '%s'", actual_col)

    id_col = None
    for candidate in ["Formulation Index", "Formulation_Index", "formulation_id", "ID"]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None:
        logger.info("  No formulation ID column found; using row index.")

    if id_col is not None:
        df_form = df.groupby(id_col, as_index=False)[actual_col].first()
    else:
        df_form = df[[actual_col]].drop_duplicates().reset_index(drop=True)

    logger.info("  Unique formulations (Burst_24h): %d", len(df_form))

    burst_vals = df_form[actual_col].values.copy()

    suspicious_threshold = 1.5
    audit_rows = []
    for i, val in enumerate(burst_vals):
        if val > suspicious_threshold:
            form_id = df_form[id_col].iloc[i] if id_col is not None else f"row_{i}"
            corrected = val / 100.0
            audit_rows.append({
                "Row_Index": i,
                "Formulation_ID": form_id,
                "Original_Value": val,
                "Proposed_Corrected_Value": corrected,
                "Reason": (
                    f"Value > {suspicious_threshold}; likely percentage-scale encoding; divided by 100"
                ),
            })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(audit_csv, index=False)
    logger.info("Audit table (%d suspicious rows) saved to: %s", len(audit_rows), audit_csv)
    if len(audit_rows) > 0:
        logger.info("\n%s", audit_df.to_string(index=False))
    else:
        logger.info("  No suspicious values found (all Burst_24h <= 1.5).")

    burst_corrected = burst_vals.copy().astype(float)
    n_corrected = 0
    for row in audit_rows:
        idx = row["Row_Index"]
        burst_corrected[idx] = row["Proposed_Corrected_Value"]
        n_corrected += 1

    n_total = len(burst_corrected)
    b_min = float(np.min(burst_corrected))
    b_max = float(np.max(burst_corrected))
    b_mean = float(np.mean(burst_corrected))
    b_sd = float(np.std(burst_corrected, ddof=1))
    n_above_one = int(np.sum(burst_corrected > 1.0))

    summary_df = pd.DataFrame({
        "N": [n_total],
        "Min": [round(b_min, 4)],
        "Max": [round(b_max, 4)],
        "Mean": [round(b_mean, 4)],
        "SD": [round(b_sd, 4)],
        "N_above_1.0": [n_above_one],
        "N_corrected_from_pct": [n_corrected],
    })
    summary_df.to_csv(summary_csv, index=False)

    logger.info(
        "Summary after correction: N=%d min=%.4f max=%.4f mean=%.4f sd=%.4f "
        "N>1.0=%d N_corrected_pct=%d — saved %s",
        n_total, b_min, b_max, b_mean, b_sd, n_above_one, n_corrected, summary_csv,
    )

    n_low = int(np.sum(burst_corrected <= 0.20))
    n_high = int(np.sum(burst_corrected > 0.20))
    n_check = n_low + n_high

    counts_df = pd.DataFrame([
        {"Class": "Low/Moderate (Burst_24h <= 0.20)", "N": n_low, "Pct": round(100 * n_low / n_total, 1)},
        {"Class": "High (Burst_24h > 0.20)", "N": n_high, "Pct": round(100 * n_high / n_total, 1)},
        {"Class": "Total", "N": n_check, "Pct": 100.0},
    ])
    counts_df.to_csv(counts_csv, index=False)
    logger.info(
        "Class counts: low/moderate=%d high=%d — saved %s",
        n_low, n_high, counts_csv,
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_vals = burst_corrected.copy()

    ax.hist(
        plot_vals,
        bins=40,
        range=(0, 1.15),
        color="#264653",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.88,
    )

    ax.axvline(
        0.20,
        color="#e9c46a",
        linewidth=1.8,
        linestyle="--",
        label="0.20 threshold (binary risk split)",
    )

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Burst$_{24h}$ (fraction of cumulative release; 1 = 100%)", fontsize=12)
    ax.set_ylabel("Number of formulations", fontsize=12)
    ax.set_title("Distribution of 24-Hour Burst Release (N = 321)", fontweight="bold")

    annot_text = (
        f"Low/Moderate: {n_low} ({100 * n_low / n_total:.1f}%)\n"
        f"High: {n_high} ({100 * n_high / n_total:.1f}%)"
    )
    ax.text(
        0.97,
        0.97,
        annot_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(fig5_png, dpi=300)
    plt.close()

    logger.info("Figure 5 saved to: %s", fig5_png)


def main(output_dir: Optional[str] = None) -> None:
    """Generate Figure4 (uncertainty), Figure5 (burst confusion), Figure6 (benchmark), Fig_5 burst histogram."""
    out = Path(output_dir) if output_dir else Path(".")
    preds_path = out / "all_predictions_and_uncertainty.csv"
    bench_path = out / "benchmark_results.csv"

    if not preds_path.exists():
        return
    preds = pd.read_csv(preds_path)
    df_n = preds[preds["Target"] == "Peppas_n"].copy()
    if not df_n.empty:
        df_n["AbsError"] = (df_n["Actual"] - df_n["Predicted"]).abs()
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=df_n, x="Uncertainty", y="AbsError",
            scatter_kws={"alpha": 0.3, "color": "blue"}, line_kws={"color": "red"},
        )
        corr = df_n["Uncertainty"].corr(df_n["AbsError"])
        plt.title(f"Uncertainty Calibration (Peppas n)\nCorrelation: {corr:.2f}")
        plt.xlabel("Ensemble Uncertainty (Std Dev)")
        plt.ylabel("Absolute Prediction Error")
        plt.tight_layout()
        plt.savefig(out / "Figure4_UncertaintyCalibration.png")
        plt.close()

    df_b = preds[preds["Target"] == "Burst_Class"].copy()
    if not df_b.empty:
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(df_b["Actual"], df_b["Predicted"])
        cmn = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        sns.heatmap(
            cmn, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Low/Moderate (≤20%)", "High (>20%)"],
            yticklabels=["Low/Moderate", "High"],
        )
        plt.title("Burst Release Classification Accuracy")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.tight_layout()
        plt.savefig(out / "Figure5_BurstClassification.png")
        plt.close()

    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        bench = bench[bench["Target"].isin(["Peppas_n", "Peppas_K"])]
        if not bench.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=bench, x="Target", y="R2", hue="Model", palette="viridis")
            plt.axhline(0, color="k", linestyle="--", linewidth=1)
            plt.title("Rigorous Benchmarking: Stacked Ensemble vs Baselines")
            plt.ylabel("R2 Score (10-Fold CV)")
            plt.tight_layout()
            plt.savefig(out / "Figure6_Benchmarking.png")
            plt.close()

    repo_root = out.resolve().parent
    run_burst_24h_scaling_audit(repo_root, out)


if __name__ == "__main__":
    try:
        import config as _cfg
        main(str(_cfg.OUTPUT_DIR))
    except ImportError:
        main(".")
