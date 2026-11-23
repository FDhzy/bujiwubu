#!/usr/bin/env python3
"""Restore and verify archived runtime models. Uses only the standard library."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "model-archive" / "manifest.json"
BLOCK_SIZE = 1024 * 1024


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            digest.update(block)
    return path.stat().st_size, digest.hexdigest()


def under_root(path, root):
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path outside checkout: {path}")
    return result


def assemble(parts, root, output):
    digest = hashlib.sha256()
    size = 0
    for entry in parts:
        source = under_root(entry["path"], root)
        part_hash = hashlib.sha256()
        part_size = 0
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
                part_hash.update(block)
                digest.update(block)
                part_size += len(block)
                size += len(block)
                output.write(block)
        if (part_size, part_hash.hexdigest()) != (entry["bytes"], entry["sha256"]):
            raise ValueError(f"Damaged model chunk: {entry['path']}")
    return size, digest.hexdigest()


def restore(model, root):
    target = under_root(model["path"], root)
    expected = (model["bytes"], model["sha256"])
    if target.exists():
        if checksum(target) != expected:
            raise ValueError(f"Refusing to overwrite a different file: {target}")
        return "verified"
    if not model["parts"]:
        raise FileNotFoundError(f"Tracked model missing: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".restore-", dir=target.parent, delete=False) as out:
            temp = Path(out.name)
            result = assemble(model["parts"], root, out)
        if result != expected:
            raise ValueError(f"Reassembled model checksum mismatch: {target}")
        if target.exists():
            raise FileExistsError(f"Destination appeared during restore: {target}")
        os.replace(temp, target)
        temp = None
        return "restored"
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    models = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
    for model in models:
        print(f"{restore(model, ROOT)}: {model['path']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"Model restore stopped: {exc}", file=sys.stderr)
        sys.exit(1)
