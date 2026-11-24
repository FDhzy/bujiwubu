"""Verify assets independently of ML libraries."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
expected = json.loads((root / "model-manifest.json").read_text(encoding="utf-8"))
for rel, spec in expected.items():
    p = root / rel
    if not p.is_file() or p.stat().st_size != spec["bytes"]:
        raise SystemExit(f"File missing or truncated: {rel}")
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise SystemExit(f"SHA-256 mismatch: {rel}")
    print(f"OK {rel}")

