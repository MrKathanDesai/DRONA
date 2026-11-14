#!/usr/bin/env python3
"""Download the DroneRF dataset and turn it into PSD features.

    python scripts/prepare_rf.py                   # download, then process
    python scripts/prepare_rf.py --verify          # check only, download nothing
    python scripts/prepare_rf.py --download-only   # leave it running overnight
    python scripts/prepare_rf.py --process-only    # archives already here
    python scripts/prepare_rf.py --only 00000_L1 00000_H1   # a few archives
    python scripts/prepare_rf.py --process-only --workers 3 # bound memory use

Fetches 23 RAR archives (4.0 GB) from Mendeley Data, checks each against its
published SHA-256, and streams them through bsdtar or 7-Zip, so the 45 GB of
CSV inside never touches the disk. Safe to interrupt and re-run: partial
downloads resume.

Writes data/rf_features/{X.npy, index.parquet, config.json}. X is
(windows, 2, bins) in dB, band 0 = lower half, band 1 = upper half. Row i of X
is row i of index.parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from dronedet.data import rf as R
from dronedet.data.manifest import LeakageError, validate
from dronedet.paths import DATA_DIR

RAW = DATA_DIR / "raw_dronerf"
OUT = DATA_DIR / "rf_features"
LISTING = RAW / "files.json"

API = "https://data.mendeley.com/public-api/datasets/f4c2b4n755"
VERSION = 1
N_ARCHIVES = 23
TIMEOUT = 120
RETRIES = 20
CHUNK = 1 << 20
USER_AGENT = "dronedet/1.0 (research dataset download)"


def ok(msg):
    print(f"  [ok]   {msg}", flush=True)


def warn(msg):
    print(f"  [WARN] {msg}", flush=True)


def step(msg):
    print(f"\n{msg}\n" + "-" * len(msg), flush=True)


def _open(url: str, **headers):
    # Mendeley answers 403 to urllib's default User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def _get_json(url: str):
    with _open(url) as r:
        return json.load(r)


def list_archives() -> list[dict]:
    """The archive list with checksums, from the API, cached once fetched."""
    if LISTING.exists():
        return json.loads(LISTING.read_text())
    out = []
    for folder in _get_json(f"{API}/folders/{VERSION}"):
        for f in _get_json(f"{API}/files?folder_id={folder['id']}&version={VERSION}"):
            c = f["content_details"]
            out.append(dict(name=f["filename"], folder=folder["name"], size=f["size"],
                            sha256=c["sha256_hash"], url=c["download_url"]))
    if len(out) != N_ARCHIVES:
        raise RuntimeError(f"API lists {len(out)} archives, expected {N_ARCHIVES}")
    out.sort(key=lambda a: R.archive_key(a["name"]))
    RAW.mkdir(parents=True, exist_ok=True)
    LISTING.write_text(json.dumps(out, indent=2))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _resume(a: dict, part: Path) -> None:
    """Append to `part` until it is complete. Raises on a dropped connection."""
    have = part.stat().st_size if part.exists() else 0
    if have >= a["size"]:
        return
    with _open(a["url"], Range=f"bytes={have}-") as r:
        mode = "ab" if r.status == 206 else "wb"     # 200: server ignored Range
        with open(part, mode) as f:
            shutil.copyfileobj(r, f, CHUNK)


def fetch(a: dict) -> str:
    path, part = RAW / a["name"], RAW / (a["name"] + ".part")
    if path.exists():
        return "present"
    for attempt in range(RETRIES):
        try:
            _resume(a, part)
            break
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            done = part.stat().st_size / 1e6 if part.exists() else 0
            print(f"    {a['name']}: {e!r} at {done:.0f} MB, retrying", flush=True)
            time.sleep(min(2 ** attempt, 60))
    else:
        raise RuntimeError(f"{a['name']}: gave up after {RETRIES} attempts")
    if sha256(part) != a["sha256"]:
        part.unlink()
        raise RuntimeError(f"{a['name']}: checksum mismatch, deleted, re-run")
    part.rename(path)
    return "downloaded"


def download(archives: list[dict], jobs: int) -> bool:
    missing = [a for a in archives if not (RAW / a["name"]).exists()]
    if not missing:
        return True
    left = sum(a["size"] - ((RAW / (a["name"] + ".part")).stat().st_size
                            if (RAW / (a["name"] + ".part")).exists() else 0)
               for a in missing)
    free = shutil.disk_usage(RAW).free
    print(f"  {len(missing)} archives, {left / 1e9:.2f} GB to fetch, "
          f"{free / 1e9:.1f} GB free, {jobs} connections")
    if left > free:
        warn("not enough disk space")
        return False

    good, t0 = True, time.time()
    with ThreadPoolExecutor(jobs) as pool:
        jobs_by_future = {pool.submit(fetch, a): a for a in missing}
        for k, fut in enumerate(as_completed(jobs_by_future), 1):
            name = jobs_by_future[fut]["name"]
            try:
                fut.result()
                ok(f"[{k}/{len(missing)}] {name}  ({(time.time() - t0) / 60:.1f} min)")
            except RuntimeError as e:
                warn(str(e))
                good = False
    return good


def verify(archives: list[dict]) -> bool:
    good = True
    for a in archives:
        path, part = RAW / a["name"], RAW / (a["name"] + ".part")
        if not path.exists():
            have = f", {part.stat().st_size / 1e6:.0f} MB so far" if part.exists() else ""
            warn(f"{a['name']} missing{have}")
            good = False
        elif sha256(path) != a["sha256"]:
            warn(f"{a['name']} checksum mismatch; delete it and re-run")
            good = False
        else:
            ok(a["name"])
    return good


def _one_archive(path: str, cfg: R.RFConfig) -> list[tuple]:
    out = []
    for name, x in R.iter_archive(path):
        bui, band, seg = R.parse_member(name)
        psd, level = R.segment_psd(x, cfg)
        out.append((bui, band, seg, psd, level))
    return out


def process(archives: list[dict], cfg: R.RFConfig = R.DEFAULT,
            workers: int | None = None) -> bool:
    paths = [str(RAW / a["name"]) for a in archives]
    print(f"  streaming {len(paths)} archives with "
          f"{workers or 'one worker per CPU'} worker(s) ...")
    t0, parts = time.time(), defaultdict(dict)
    # Each worker holds one 10 M-sample CSV line as text (~0.3 GB peak), so
    # --workers bounds memory; the default is one per CPU.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one_archive, p, cfg) for p in paths]
        for k, fut in enumerate(as_completed(futures), 1):
            for bui, band, seg, psd, level in fut.result():
                parts[(bui, seg)][band] = (psd, level)
            print(f"    {k}/{len(paths)}  ({time.time() - t0:.0f} s)", flush=True)

    # A segment is only usable with both halves of the band.
    both = sorted(k for k, v in parts.items() if len(v) == 2)
    if len(both) != len(parts):
        warn(f"{len(parts) - len(both)} segments have only one band; dropped")
    if not both:
        warn("no complete segments")
        return False

    seg = pd.DataFrame([dict(bui=b, segment=s, **R.decode_bui(b)) for b, s in both])
    seg = R.assign_splits(seg)

    n = cfg.windows_per_segment
    X = np.empty((len(both) * n, 2, cfg.n_bins), dtype=np.float32)
    level = np.empty((len(both) * n, 2), dtype=np.float32)
    for i, key in enumerate(both):
        for j, band in enumerate("LH"):
            X[i * n:(i + 1) * n, j], level[i * n:(i + 1) * n, j] = parts[key][band]

    df = seg.loc[seg.index.repeat(n)].reset_index(drop=True)
    df["start_s"] = np.tile(np.arange(n) * cfg.window_s, len(both))
    df["path"] = [f"raw_dronerf/{b}_{s:03d}" for b, s in zip(df.bui, df.segment)]
    df["level_db_low"], df["level_db_high"] = level[:, 0], level[:, 1]
    df["source"] = "dronerf"
    df["sample_rate"] = cfg.sample_rate
    df["duration"] = cfg.window_s
    df["exclude_reason"] = ""

    try:
        stats = validate(df)
        ok(f"splits valid: {stats['clips']} windows, {stats['groups']} groups, "
           f"{stats['drone_clips']} drone / {stats['background_clips']} background")
    except LeakageError as e:
        if len(archives) == N_ARCHIVES:
            raise
        warn(f"partial run, some splits are incomplete: {e}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "X.npy", X)
    df.to_parquet(OUT / "index.parquet", index=False)
    (OUT / "config.json").write_text(json.dumps({
        "config": asdict(cfg),
        "fingerprint": cfg.fingerprint(),
        "archives": sorted(R.archive_key(a["name"]) for a in archives),
        "shape": list(X.shape),
    }, indent=2))

    print(f"  X {X.shape} {X.dtype}, {X.nbytes / 1e6:.0f} MB, "
          f"dB {X.min():.1f} .. {X.max():.1f}, "
          f"non-finite {int((~np.isfinite(X)).sum())}")
    print(df.groupby(["label", "split_tier1"]).group_id.nunique()
            .unstack(fill_value=0).to_string())

    good = True
    counts = seg.drone_model.fillna("background").value_counts()
    for model, want in R.EXPECT_SEGMENTS.items():
        have = int(counts.get(model, 0))
        (ok if have == want else warn)(f"{model}: {have} segments (expected {want})")
        good &= have == want
    print(f"  written to {OUT}")
    return good


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check checksums, download nothing")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--process-only", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    help="archive keys, e.g. 00000_L1 11000_H")
    ap.add_argument("--jobs", type=int, default=4, help="parallel downloads")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel archive-processing workers (default: one per CPU; "
                         "about 0.3-0.4 GB RAM each, so use 2-4 on a 16 GB machine "
                         "that is doing other work)")
    args = ap.parse_args()
    if args.workers is not None and args.workers < 1:
        ap.error("--workers must be at least 1")

    if not any((shutil.which("bsdtar"), shutil.which("7z"),
                shutil.which("7z.exe"),
                Path(r"C:\Program Files\7-Zip\7z.exe").is_file())) \
            and not (args.verify or args.download_only):
        print("Archive processing requires bsdtar or 7-Zip (7z.exe).")
        return 1

    archives = list_archives()
    if args.only:
        unknown = set(args.only) - {R.archive_key(a["name"]) for a in archives}
        if unknown:
            print(f"unknown archive keys: {sorted(unknown)}")
            return 1
        archives = [a for a in archives if R.archive_key(a["name"]) in args.only]

    if args.verify:
        step("Verifying DroneRF archives")
        return 0 if verify(archives) else 1

    if not args.process_only:
        step("Downloading DroneRF  (Allahham et al., Data in Brief 2019)")
        if not download(archives, args.jobs):
            print("\n  incomplete; re-run the same command to resume")
            return 1
        ok("archives present and checksummed")
    if args.download_only:
        return 0

    step("Processing: raw RF -> PSD per 25 ms window")
    missing = [a["name"] for a in archives if not (RAW / a["name"]).exists()]
    if missing:
        warn(f"{len(missing)} archives missing, e.g. {missing[0]}")
        return 1
    return 0 if process(archives, workers=args.workers) else 1


if __name__ == "__main__":
    raise SystemExit(main())
