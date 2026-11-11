#!/usr/bin/env python3
"""Reproduce the frozen S candidate for IEEE TrafficFlowBench 2026.

The algorithm and trained weights are unchanged. This script replaces only
machine-specific paths and the historical job's expired dispatch-time guard.
It requires the organizer's authorized release and the exact frozen v1 base.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "S_selector"
EXPECTED_BASE = "c3c6eeaabdcc1335480f33fae9cc37a0ec982f6f428f273bc256034bc2caebb0"
MODEL_HASHES = {
    "model_S_101.pt": "38080767731177e589d925cbb7a9fe1432f36697f343565e1d97cda0d3695644",
    "model_S_202.pt": "6d89cde6241fd9142a3186de2f5dfc5448e62c97fed9292ad0bbaccf308425c3",
    "model_S_303.pt": "81d4bddb3c454edc6ea2eb5eea27e86eb964d238be860f675aac0fc3b2b8df57",
}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    """Refuse missing or altered frozen source inputs."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch: {path}\nExpected {expected}\nActual   {actual}")


def reproduce(release: Path, base: Path, output: Path) -> dict:
    """Run frozen S ensemble and merge its decisions onto the exact v1 base."""
    release = release.expanduser().resolve()
    base = base.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.name != "submission.csv":
        raise ValueError("Output must be named submission.csv for a valid Kaggle ZIP.")
    if not (release / "submission_key.csv").is_file():
        raise FileNotFoundError(f"Official submission_key.csv not found in {release}")
    if not (release / "task2").is_dir() or not (release / "corridors").is_dir():
        raise FileNotFoundError("Expected official task2/ and corridors/ data folders.")
    require_hash(base, EXPECTED_BASE)
    weights = HERE / "weights" / "S"
    for filename, checksum in MODEL_HASHES.items():
        require_hash(weights / filename, checksum)
    manifest = json.loads((SOURCE / "source-manifest.json").read_text(encoding="utf-8"))
    for filename, checksum in manifest.items():
        require_hash(SOURCE / filename, checksum)

    os.environ["TFB_REL"] = str(release)
    sys.path.insert(0, str(SOURCE))
    import torch  # Imported after checking required inputs.
    import final_logic

    config = json.loads((SOURCE / "config" / "final.json").read_text(encoding="utf-8"))
    frozen_arm = config["arms"]["S"]
    if float(frozen_arm["threshold"]) != 0.5:
        raise ValueError("Frozen S classification threshold has changed.")
    portable_arm = {
        "models": [
            {"path": str(weights / Path(item["path"]).name), "sha256": item["sha256"]}
            for item in frozen_arm["models"]
        ]
    }
    if set(Path(m["path"]).name for m in portable_arm["models"]) != set(MODEL_HASHES):
        raise ValueError("Unexpected checkpoint list in frozen configuration.")
    torch.set_num_threads(4)
    arm = final_logic.load_arm(portable_arm)
    decisions, per_window = final_logic.build_predictions({"S": (arm, 0.5)})
    output.parent.mkdir(parents=True, exist_ok=True)
    info = final_logic.combine(
        decisions["S"], output, str(base), EXPECTED_BASE, release / "submission_key.csv"
    )
    zip_path = final_logic.archive(output)
    per_window.to_csv(output.with_name(output.stem + "_windows.csv"), index=False)
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "competition": "2026-ieee-big-data-traffic-flow-bench",
        "candidate": "S",
        "kaggle_submission_id": 58062952,
        "classification_threshold": 0.5,
        "source": "Frozen selector-final-v1; only path and expired dispatch guard bypassed",
        "base_sha256": EXPECTED_BASE,
        "model_sha256": MODEL_HASHES,
        "csv": str(output),
        "csv_sha256": sha256_file(output),
        "zip": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "merge_checks": info,
        "official_score_recomputed": False,
    }
    output.with_name(output.stem + "_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    """Parse the single-entry command and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="Official kaggle_public root")
    parser.add_argument("--base", type=Path, required=True, help="Frozen sub_v1.csv with verified SHA-256")
    parser.add_argument("--output", type=Path, default=HERE / "output" / "submission.csv")
    arguments = parser.parse_args()
    try:
        print(json.dumps(reproduce(arguments.release, arguments.base, arguments.output), indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        parser.exit(1, f"Reproduction failed: {error}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
