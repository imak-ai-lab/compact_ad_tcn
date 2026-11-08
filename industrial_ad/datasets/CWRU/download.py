"""Download the CWRU .mat files listed in the catalogue and write a manifest.

    python -m industrial_ad.datasets.CWRU.download --dst ./data/CWRU

The official files live at https://engineering.case.edu/sites/default/files/<id>.mat
(the Bearing Data Center pages link straight to them). Mirrors exist, but the
manifest records the URL actually used plus the md5 of every file, so a later
reader can tell which copy produced the numbers. Nothing here is parsed or
resampled — that happens in `dataset.py`, once, under the run's own metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from industrial_ad.datasets.CWRU.catalog import RECORDINGS

OFFICIAL_BASE = "https://engineering.case.edu/sites/default/files/"


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_all(dst: str | Path = "./data/CWRU", base_url: str = OFFICIAL_BASE, retries: int = 3) -> dict:
    import requests  # already a project dependency (PU download)

    dst = Path(dst)
    raw = dst / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = dst / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}

    for rec in RECORDINGS:
        target = raw / rec.mat_name
        if target.exists() and target.stat().st_size > 0:
            entry = manifest["files"].get(str(rec.file_id))
            if entry and entry.get("md5") == md5_of(target):
                continue
        url = base_url + rec.mat_name
        for attempt in range(1, retries + 1):
            try:
                print(f"[cwru] {rec.file_id} ({rec.bearing_id}, {rec.load_hp} HP) <- {url}", flush=True)
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                if len(response.content) < 10_000:
                    raise RuntimeError(f"suspiciously small payload ({len(response.content)} bytes) for {url}")
                target.write_bytes(response.content)
                break
            except Exception as exc:  # noqa: BLE001 - retry loop, error is re-raised below
                print(f"[cwru]   attempt {attempt}/{retries} failed: {exc!r}", flush=True)
                if attempt == retries:
                    raise
                time.sleep(5 * attempt)
        manifest["files"][str(rec.file_id)] = {
            "file": rec.mat_name,
            "bearing_id": rec.bearing_id,
            "load_hp": rec.load_hp,
            "fs_hz_published": rec.fs_hz,
            "url": url,
            "bytes": target.stat().st_size,
            "md5": md5_of(target),
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["complete"] = all(str(r.file_id) in manifest["files"] for r in RECORDINGS)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[cwru] manifest: {manifest_path} complete={manifest['complete']}", flush=True)
    return manifest


def check_present(dst: str | Path = "./data/CWRU") -> list[str]:
    """Return the names of catalogue files missing under <dst>/raw (empty = all present)."""
    raw = Path(dst) / "raw"
    return [r.mat_name for r in RECORDINGS if not (raw / r.mat_name).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst", default=os.environ.get("CWRU_ROOT", "./data/CWRU"))
    parser.add_argument("--base-url", default=OFFICIAL_BASE)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        missing = check_present(args.dst)
        print("missing:", missing if missing else "none")
        raise SystemExit(1 if missing else 0)
    download_all(args.dst, base_url=args.base_url)


if __name__ == "__main__":
    main()
