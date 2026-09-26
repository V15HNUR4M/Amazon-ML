"""
main.py — Top-level entry point for the Business Entity Resolution pipeline.

This file shows the intended wiring of pipeline stages.
Nothing is executed here until all modules are implemented.

Future pipeline order
---------------------
1. Load config
2. Load train / test data
3. Preprocess records
4. Build blocking candidates
5. Build labelled training pairs
6. Compute features
7. Train matching model
8. Select threshold on validation set
9. Run test prediction
10. Write output files

Usage (future)
--------------
python -m src.main [--mode train|predict|full]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


def train_pipeline(config, sample_source1: int = 5000, distractors: int = 5000) -> None:
    """Run the training pipeline end-to-end."""
    logger.info("Executing training pipeline...")
    from scripts.run_validation_scale_check import run_scale_check
    run_scale_check(
        sample_source1=sample_source1,
        distractors=distractors,
        random_seed=config.RANDOM_SEED,
    )


def prediction_pipeline(
    config,
    smoke_test: bool = False,
    smoke_test_n: int = 2000,
    pool_rows: int | None = None,
    batch_size: int = 10_000,
    threshold: float = 0.90,
) -> None:
    """Run the test prediction pipeline end-to-end."""
    logger.info("Executing prediction pipeline...")
    from scripts.generate_submission import run_inference
    run_inference(
        smoke_test=smoke_test,
        smoke_test_n=smoke_test_n,
        pool_rows=pool_rows,
        batch_size=batch_size,
        threshold=threshold,
        cache_dir=config.CACHE_DIR,
        output_dir=config.OUTPUT_DIR,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="Amazon ML Hackathon — Business Entity Resolution Pipeline",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "full"],
        default="predict",
        help="Pipeline mode: train, predict, or full (train+predict).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Run smoke test mode for fast validation.",
    )
    parser.add_argument(
        "--smoke-n",
        type=int,
        default=2000,
        help="Number of S1 entities for smoke test.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Match probability threshold (default: 0.90).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    from src.config import Config  # noqa: PLC0415

    config = Config()
    logger.info("Config loaded: %s", config)

    if args.mode in ("train", "full"):
        train_pipeline(config)

    if args.mode in ("predict", "full"):
        prediction_pipeline(
            config,
            smoke_test=args.smoke_test,
            smoke_test_n=args.smoke_n,
            threshold=args.threshold,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

