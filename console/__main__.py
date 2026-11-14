"""python -m console [--host 0.0.0.0] [--port 8080]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="Drona: live acoustic drone detection")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to serve other devices (microphone then needs HTTPS)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--mic-model", choices=["original", "speaker"], default="original",
                    help="speaker = detector also trained on played-back sound")
    a = ap.parse_args()
    import os
    os.environ["DRONA_MIC_MODEL"] = a.mic_model
    import uvicorn
    print(f"Drona is listening on http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}")
    uvicorn.run("console.server:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
