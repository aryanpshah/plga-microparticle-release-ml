"""
Single entrypoint to reproduce the paper's main results.

Runs: (1) main pipeline, (2) benchmarks, (3) rigorous validation,
(4) refinement visualizations. All outputs go to outputs/.
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import config
config.set_seeds()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce PLGA paper results.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip benchmarks and extra visualizations (pipeline + validation only).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory (default: config.DATA_DIR).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: config.OUTPUT_DIR).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or config.DATA_DIR
    output_dir = args.output_dir or config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / config.RAW_DATASET
    initial_path = data_dir / config.INITIAL_DATASET
    if not raw_path.exists():
        logger.error("Data not found: %s. Place %s and %s in data/ or set DATA_DIR.", raw_path, config.RAW_DATASET, config.INITIAL_DATASET)
        sys.exit(1)
    if not initial_path.exists():
        logger.error("Data not found: %s.", initial_path)
        sys.exit(1)

    # 1. Main pipeline
    logger.info("=== 1. Main pipeline ===")
    from src.plga_pipeline_v2 import run_pipeline
    run_pipeline(str(raw_path), str(initial_path), str(output_dir))

    # 2. Rigorous validation
    logger.info("=== 2. Rigorous validation ===")
    from src.rigorous_validation import rigorous_validation
    rigorous_validation(str(raw_path), str(initial_path), str(output_dir))

    if not args.fast:
        # 3. Benchmarks
        logger.info("=== 3. Benchmarks ===")
        from benchmark_baselines import run_benchmarks
        run_benchmarks(str(raw_path), str(initial_path), str(output_dir))

        # 4. Refinement visualizations (uses CSVs written to output_dir)
        logger.info("=== 4. Refinement visualizations ===")
        from visualize_refinement import main as viz_main
        viz_main(str(output_dir))

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
