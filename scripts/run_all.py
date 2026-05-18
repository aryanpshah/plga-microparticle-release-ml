"""
Single entrypoint to reproduce the repository's minimal release workflow.

Runs the main pipeline and rigorous validation only. Outputs go to outputs/.
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config
config.set_seeds()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the PLGA release workflow.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Accepted for compatibility; this minimal repository already runs only pipeline + validation.",
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

    if args.fast:
        logger.info("--fast selected; no additional steps are present in this minimal repository.")

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
