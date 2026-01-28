from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF pipeline for a sweep of k values.",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="18,16,14,12,10,8,6,4,2",
        help="Comma-separated k values to run.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="KRAS,TP53,GAPDH",
        help="Comma-separated target list.",
    )
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--rf-script",
        type=str,
        default="rf_rna_to_protein.py",
        help="Path to the RF pipeline script.",
    )
    return parser.parse_args()


def main() -> int:
    _setup_logging()
    args = _parse_args()

    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    if not k_values:
        logging.error("No k values provided.")
        return 1

    rf_script = Path(args.rf_script)
    if not rf_script.exists():
        logging.error("RF script not found: %s", rf_script)
        return 1

    for k in k_values:
        out_dir = Path(f"outputs_k{k}")
        if out_dir.exists():
            logging.info("Skipping k=%s because %s already exists.", k, out_dir)
            continue
        cmd = [
            sys.executable,
            str(rf_script),
            "--targets",
            args.targets,
            "--feature-selection",
            "kbest",
            "--k-best",
            str(k),
            "--n-iter",
            str(args.n_iter),
            "--n-jobs",
            str(args.n_jobs),
            "--out",
            str(out_dir),
        ]
        logging.info("Running k=%s -> %s", k, out_dir)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logging.error("Run failed for k=%s with exit code %s", k, result.returncode)
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
