"""Print result summaries from the output CSVs."""

from pathlib import Path
from typing import Optional

import pandas as pd


def _print_csv(path: Path, title: str) -> None:
    if not path.exists():
        return
    print(f"\n{title}")
    print(pd.read_csv(path).to_string(index=False))


def main(output_dir: Optional[str] = None) -> None:
    out = Path(output_dir) if output_dir else Path(".")
    _print_csv(out / "performance_metrics.csv", "Performance metrics")
    _print_csv(out / "benchmark_results.csv", "Benchmark results")
    _print_csv(out / "loso_results.csv", "LOSO results")
    _print_csv(out / "burst_classification_metrics.csv", "Burst classification metrics")
    _print_csv(out / "applicability_domain_metrics.csv", "Applicability domain metrics")


if __name__ == "__main__":
    try:
        import config as _cfg
        main(str(_cfg.OUTPUT_DIR))
    except ImportError:
        main(".")
