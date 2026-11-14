"""Get audio into the detector from anywhere: any audio or video file, a
link (YouTube and the other sites yt-dlp supports, or a direct file URL),
or a microphone recording.

Everything is decoded to 16 kHz mono WAV bytes, so the rest of the pipeline
(dronedet.live) sees one format. Decoding uses soundfile when it can (WAV,
FLAC, OGG, MP3) and otherwise ffmpeg from the imageio-ffmpeg package, which
ships its own binary: no system install. Video files keep only their sound.

Only the first `MAX_SECONDS` are analysed, so a long upload cannot tie up
the CPU for minutes. Downloads are size-capped. The detector only ever sees
the sound; it is the user's responsibility to have the right to use a clip.
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
MAX_SECONDS = 300               # analyse at most 5 minutes
MAX_DOWNLOAD_BYTES = 300 * 2**20
MAX_SOURCE_SECONDS = 60 * 60    # refuse links to anything longer than an hour
TIMEOUT = 60

# What the upload widget accepts. Video containers are decoded for sound only.
AUDIO_TYPES = ["wav", "flac", "ogg", "oga", "opus", "mp3", "m4a", "aac", "wma", "aif", "aiff"]
VIDEO_TYPES = ["mp4", "m4v", "mov", "webm", "mkv", "avi", "3gp"]
UPLOAD_TYPES = AUDIO_TYPES + VIDEO_TYPES

MEDIA_EXT = re.compile(r"\.(" + "|".join(UPLOAD_TYPES) + r")$", re.I)


class MediaError(RuntimeError):
    """A clip that cannot be fetched or decoded, with a message for the user."""


@dataclass(frozen=True)
class Clip:
    wav: bytes          # 16 kHz mono WAV
    title: str
    seconds: float      # duration analysed
    truncated: bool     # True if the source was longer than MAX_SECONDS


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _to_clip(y: np.ndarray, title: str, source_seconds: float) -> Clip:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise MediaError("The file contains no audio.")
    truncated = source_seconds > MAX_SECONDS + 0.5
    y = y[: MAX_SECONDS * SR]
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    return Clip(buf.getvalue(), title, len(y) / SR, truncated)


def _decode_soundfile(data: bytes) -> tuple[np.ndarray, float] | None:
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        return None
    y = y.mean(axis=1)
    seconds = len(y) / sr
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    return y, seconds


def _decode_ffmpeg(path: Path) -> tuple[np.ndarray, float]:
    exe = ffmpeg_exe()
    if not exe:
        raise MediaError("This format needs ffmpeg: pip install imageio-ffmpeg "
                         "(or convert the file to WAV first).")
    # Decode a little more than we keep, to know whether it was truncated.
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(SR), "-t", str(MAX_SECONDS + 1),
           "-f", "f32le", "pipe:1"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=180, check=False)
    except subprocess.TimeoutExpired as e:
        raise MediaError("Decoding took too long; try a shorter clip.") from e
    if out.returncode != 0 or not out.stdout:
        msg = out.stderr.decode(errors="ignore").strip().splitlines()
        raise MediaError("Could not read audio from this file"
                         + (f": {msg[-1]}" if msg else ".")
                         + " Is there a sound track?")
    y = np.frombuffer(out.stdout, dtype="<f4").copy()
    return y, len(y) / SR


def decode_bytes(data: bytes, name: str = "upload") -> Clip:
    """Any audio or video file's bytes -> 16 kHz mono WAV clip."""
    if not data:
        raise MediaError("The file is empty.")
    got = _decode_soundfile(data)
    if got is None:
        suffix = Path(name).suffix or ".bin"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / f"input{suffix}"
            p.write_bytes(data)
            got = _decode_ffmpeg(p)
    y, seconds = got
    return _to_clip(y, Path(name).name, seconds)


# --- links ---------------------------------------------------------------
def classify_url(url: str) -> str:
    """'direct' (a media file URL), 'site' (a page for yt-dlp, e.g. YouTube)
    or raises MediaError for something that is not an http(s) link."""
    u = urllib.parse.urlparse(url.strip())
    if u.scheme not in ("http", "https") or not u.netloc:
        raise MediaError("Paste a full link starting with http:// or https://")
    return "direct" if MEDIA_EXT.search(u.path) else "site"


def _download_direct(url: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "dronedet/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read(MAX_DOWNLOAD_BYTES + 1)
    except Exception as e:
        raise MediaError(f"Could not download the file: {e}") from e
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise MediaError(f"The file is larger than {MAX_DOWNLOAD_BYTES // 2**20} MB.")
    return data, Path(urllib.parse.urlparse(url).path).name or "download"


def _download_site(url: str) -> tuple[bytes, str]:
    try:
        import yt_dlp
    except ImportError as e:
        raise MediaError("Links to YouTube and other sites need yt-dlp: "
                         "pip install yt-dlp. You can also download the video "
                         "yourself and upload the file.") from e
    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmp) / "media.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "max_filesize": MAX_DOWNLOAD_BYTES,
            "socket_timeout": TIMEOUT,
            "match_filter": yt_dlp.utils.match_filter_func(
                f"duration <=? {MAX_SOURCE_SECONDS}"),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as e:
            raise MediaError(f"Could not get audio from this link: {e}") from e
        files = [p for p in Path(tmp).iterdir() if p.is_file()]
        if not files:
            raise MediaError("Nothing was downloaded: the video may be longer than "
                             "an hour, private, age-restricted or region-locked.")
        title = (info or {}).get("title") or "linked video"
        return files[0].read_bytes(), f"{title}{files[0].suffix}"


def fetch_url(url: str) -> Clip:
    """A YouTube (or other site) link, or a direct media-file URL -> clip."""
    kind = classify_url(url)
    data, name = _download_direct(url) if kind == "direct" else _download_site(url)
    return decode_bytes(data, name)
