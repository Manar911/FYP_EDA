"""
integrity.py  —  EDA Security

Startup integrity verification for critical artifacts.

"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_HERE      = Path(__file__).resolve().parent    # EDA_embedded/core/
_ROOT      = _HERE.parent                       # EDA_embedded/
_HASH_FILE = _ROOT / "integrity_hashes.json"

ARTIFACTS = {
    "airports_csv":   _HERE / "data" / "airports.csv",
    "lightgbm_model": _ROOT / "model" / "lightgbm_pipeline.joblib",
}


@dataclass
class IntegrityResult:
    ok:       bool
    warnings: List[str] = field(default_factory=list)
    details:  dict      = field(default_factory=dict)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_hashes() -> dict:
    """
    Compute and save expected hashes.
    Run once before deployment from EDA_embedded/:

        python -c "from core.integrity import generate_hashes; generate_hashes()"
    """
    hashes = {}
    for name, path in ARTIFACTS.items():
        if path.exists():
            hashes[name] = {
                "path":   str(path.relative_to(_ROOT)),
                "sha256": _sha256(path),
                "size":   path.stat().st_size,
            }
        else:
            hashes[name] = {"path": str(path), "sha256": None, "size": None}
            print(f"WARNING: {name} not found at {path}")

    with open(_HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)

    print(f"Integrity hashes saved to {_HASH_FILE}")
    for name, info in hashes.items():
        h = info["sha256"]
        print(f"  {name}: {h[:16]}..." if h else f"  {name}: NOT FOUND")
    return hashes


def check_integrity() -> IntegrityResult:
    """
    Verify hashes of all critical artifacts at startup.
    Never raises — always returns a result.
    """
    warnings = []
    details  = {}

    if not _HASH_FILE.exists():
        return IntegrityResult(
            ok=True,
            warnings=["Hash file not found — run generate_hashes() before deployment"],
        )

    try:
        with open(_HASH_FILE) as f:
            expected = json.load(f)
    except Exception as e:
        return IntegrityResult(
            ok=False,
            warnings=[f"Could not read hash file: {e}"],
        )

    for name, path in ARTIFACTS.items():
        if name not in expected:
            warnings.append(f"{name}: not in hash file")
            details[name] = "missing from hash file"
            continue

        if not path.exists():
            warnings.append(f"{name}: file not found")
            details[name] = "file missing"
            continue

        expected_hash = expected[name].get("sha256")
        if not expected_hash:
            details[name] = "no expected hash stored"
            continue

        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            warnings.append(f"{name}: hash mismatch — file may have been modified")
            details[name] = f"TAMPERED"
        else:
            details[name] = f"OK ({actual_hash[:12]}...)"

    return IntegrityResult(
        ok=len(warnings) == 0,
        warnings=warnings,
        details=details,
    )