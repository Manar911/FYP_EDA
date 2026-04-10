"""
integrity.py

Startup integrity verification for EDA Core Engine.

Computes SHA-256 hashes of critical artifacts at first run and
stores expected values. On subsequent runs, compares current
hashes to expected values and warns if tampering is detected.

Artifacts checked:
    - models/lightgbm_pipeline.joblib
    - src/eda/data/airports.csv

Usage:
    from eda.integrity import check_integrity
    result = check_integrity()
    if not result.ok:
        print(result.warnings)

Design:
    - Never blocks the system — always returns a result
    - On tamper detection: logs warning, system continues with fallback
    - Hash file stored alongside artifacts for transparency
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Paths relative to project root
_HERE        = Path(__file__).resolve().parent          # src/eda/
_PROJECT_ROOT = _HERE.parent.parent                     # EDA_CoreEngine/
_HASH_FILE   = _PROJECT_ROOT / "embedded" / "integrity_hashes.json"

ARTIFACTS = {
    "airports_csv":       _HERE / "data" / "airports.csv",
    "lightgbm_model":     _PROJECT_ROOT / "models" / "lightgbm_pipeline.joblib",
}


@dataclass
class IntegrityResult:
    ok:        bool
    warnings:  List[str] = field(default_factory=list)
    details:   dict      = field(default_factory=dict)


def _sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_hashes() -> dict:
    """
    Compute and save expected hashes for all artifacts.
    Call this once after training / before deployment.

    Run from project root:
        python -c "from eda.integrity import generate_hashes; generate_hashes()"
    """
    hashes = {}
    for name, path in ARTIFACTS.items():
        if path.exists():
            hashes[name] = {
                "path":   str(path.relative_to(_PROJECT_ROOT)),
                "sha256": _sha256(path),
                "size":   path.stat().st_size,
            }
        else:
            hashes[name] = {"path": str(path), "sha256": None, "size": None}

    _HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)

    print(f"Integrity hashes saved to {_HASH_FILE}")
    for name, info in hashes.items():
        print(f"  {name}: {info['sha256'][:16]}...")
    return hashes


def check_integrity() -> IntegrityResult:
    """
    Verify SHA-256 hashes of all critical artifacts.

    Returns IntegrityResult with ok=True if all pass,
    ok=False with warnings listing any failures.

    Never raises — always returns a result so the system
    can continue with appropriate fallback behaviour.
    """
    warnings = []
    details  = {}

    # If no hash file exists yet, skip check (first run)
    if not _HASH_FILE.exists():
        return IntegrityResult(
            ok=True,
            warnings=["Integrity hash file not found — run generate_hashes() before deployment"],
            details={},
        )

    try:
        with open(_HASH_FILE) as f:
            expected = json.load(f)
    except Exception as e:
        return IntegrityResult(
            ok=False,
            warnings=[f"Could not read integrity hash file: {e}"],
        )

    for name, path in ARTIFACTS.items():
        if name not in expected:
            warnings.append(f"{name}: not in hash file")
            details[name] = "missing from hash file"
            continue

        if not path.exists():
            warnings.append(f"{name}: file not found at {path}")
            details[name] = "file missing"
            continue

        expected_hash = expected[name].get("sha256")
        if not expected_hash:
            details[name] = "no expected hash stored"
            continue

        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            warnings.append(
                f"{name}: hash mismatch — file may have been modified"
            )
            details[name] = f"TAMPERED (expected {expected_hash[:12]}... got {actual_hash[:12]}...)"
        else:
            details[name] = f"OK ({actual_hash[:12]}...)"

    return IntegrityResult(
        ok=len(warnings) == 0,
        warnings=warnings,
        details=details,
    )