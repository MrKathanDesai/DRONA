#!/usr/bin/env python3
"""Fetch every dataset this project needs, and verify what arrived.

    python scripts/download_data.py            # fetch anything missing
    python scripts/download_data.py --verify   # check only, download nothing

Safe to re-run: it skips anything already present and intact. Nothing here is
committed to git (data/ is gitignored), so this script IS the record of where
the data came from — treat it as part of the method, not a convenience.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# --- what we expect to end up with -----------------------------------------
EXPECT = {
    "droneaudio": {"wav": 23408, "drone": 1332, "background": 10372},
    "svanstrom": {"wav": 90},
    "esc50_meta": {"rows": 2001},          # 2000 clips + header
}


def ok(msg):
    print(f"  [ok]   {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def step(msg):
    print(f"\n{msg}\n" + "-" * len(msg))


# --- 1. DroneAudioDataset ---------------------------------------------------
def get_droneaudio(verify_only=False):
    step("1. DroneAudioDataset  (Al-Emadi et al., IWCMC 2019)")
    dest = DATA / "raw_droneaudio"
    binary = dest / "Binary_Drone_Audio"

    if not binary.exists():
        if verify_only:
            warn("missing")
            return False
        print("  cloning ~800 MB, this takes a few minutes...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "-q",
             "https://github.com/saraalemadi/DroneAudioDataset.git", str(dest)],
            check=True)
        shutil.rmtree(dest / ".git", ignore_errors=True)

    n_drone = len(list((binary / "yes_drone").glob("*.wav")))
    n_bg = len(list((binary / "unknown").glob("*.wav")))
    e = EXPECT["droneaudio"]
    good = n_drone == e["drone"] and n_bg == e["background"]
    (ok if good else warn)(
        f"{n_drone} drone + {n_bg} background "
        f"(expected {e['drone']} + {e['background']})")
    return good


# --- 2. ESC-50 metadata -----------------------------------------------------
def get_esc50(verify_only=False):
    step("2. ESC-50 metadata  (Piczak, 2015) — labels only, no audio")
    dest = DATA / "esc50_meta.csv"
    url = ("https://raw.githubusercontent.com/karolpiczak/ESC-50/"
           "master/meta/esc50.csv")

    if not dest.exists():
        if verify_only:
            warn("missing")
            return False
        print("  downloading...")
        urllib.request.urlopen  # noqa: B018  (keep import obvious)
        with urllib.request.urlopen(url, timeout=90) as r:
            dest.write_bytes(r.read())

    rows = len(dest.read_text().splitlines())
    good = rows == EXPECT["esc50_meta"]["rows"]
    (ok if good else warn)(f"{rows} rows (expected "
                           f"{EXPECT['esc50_meta']['rows']})")
    print("     used to look up the true category of each background clip,")
    print("     rather than decoding it from the filename (which disagrees")
    print("     with the official metadata on ~20 of 10,000 files).")
    return good


# --- 3. Svanstrom multi-sensor (audio portion only) -------------------------
def get_svanstrom(verify_only=False):
    step("3. Svanstrom multi-sensor dataset  (Halmstad, 2021) — audio only")
    dest = DATA / "raw_svanstrom"
    audio = dest / "Audio"
    url = ("https://zenodo.org/api/records/5500576/files/"
           "DroneDetectionThesis/Drone-detection-dataset-v1.0.0.zip/content")

    if not audio.exists():
        if verify_only:
            warn("missing")
            return False
        dest.mkdir(parents=True, exist_ok=True)
        zpath = dest / "_svanstrom.zip"
        print("  downloading 311 MB from Zenodo...")
        with urllib.request.urlopen(url, timeout=600) as r, \
                open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        print("  extracting the 90 audio files (the 650 videos are skipped)")
        with zipfile.ZipFile(zpath) as z:
            wavs = [m for m in z.namelist() if m.lower().endswith(".wav")]
            z.extractall(dest, members=wavs)
        inner = next(dest.glob("DroneDetectionThesis-*/Data/Audio"))
        inner.rename(audio)
        shutil.rmtree(inner.parent.parent, ignore_errors=True)
        zpath.unlink(missing_ok=True)

    files = sorted(audio.glob("*.wav"))
    classes = {}
    for p in files:
        classes[p.stem.rsplit("_", 1)[0]] = classes.get(
            p.stem.rsplit("_", 1)[0], 0) + 1
    good = len(files) == EXPECT["svanstrom"]["wav"]
    (ok if good else warn)(f"{len(files)} clips — {classes}")
    print("     44.1 kHz, 10 s each, recorded OUTDOORS. Held back as the")
    print("     tier-3 test set: a different corpus, different hardware.")
    print("     Never trained on.")
    return good


# --- 4. Pretrained encoder (PANNs MobileNetV1, AudioSet) -------------------
PANNS = {
    "MobileNetV1_mAP=0.389.pth": (
        "https://zenodo.org/records/3987831/files/MobileNetV1_mAP=0.389.pth?download=1",
        "a419303e1c88aa1b9d2ac3811563d371"),
    "class_labels_indices.csv": (
        "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/"
        "master/metadata/class_labels_indices.csv", None),
}


def get_pretrained(verify_only=False):
    step("4. Pretrained encoder  (Kong et al., PANNs 2020) — 24 MB")
    dest = DATA / "pretrained"
    dest.mkdir(parents=True, exist_ok=True)
    good = True
    for name, (url, md5) in PANNS.items():
        path = dest / name
        if not path.exists():
            if verify_only:
                warn(f"{name} missing")
                good = False
                continue
            print(f"  downloading {name} ...")
            with urllib.request.urlopen(url, timeout=600) as r, \
                    open(path, "wb") as f:
                shutil.copyfileobj(r, f)
        if md5 and hashlib.md5(path.read_bytes()).hexdigest() != md5:
            warn(f"{name} checksum mismatch; delete it and re-run")
            good = False
        else:
            ok(name)
    print("     Frozen AudioSet encoder; only a small head is trained on top.")
    return good


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check what is present, download nothing")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("  DATASETS" + ("  (verify only)" if args.verify else ""))
    print("=" * 66)

    results = [get_droneaudio(args.verify),
               get_esc50(args.verify),
               get_svanstrom(args.verify),
               get_pretrained(args.verify)]

    print("\n" + "=" * 66)
    if all(results):
        print("  all datasets present and the counts match")
        print("  next: python scripts/build_manifest.py")
        return 0
    print("  something is missing or incomplete — see warnings above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
