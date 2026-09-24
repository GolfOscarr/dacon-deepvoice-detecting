"""M tier: what a file says it is, without decoding it.

This is the tier that runs on **100%** of the corpus, and it is the whole input
to the shortcut audit (docs/EDA/06 X1). That audit is a *population* statement:
"does container predict the label" cannot be answered on a sample stratified by
source, because stratifying by source is exactly what destroys the effect being
measured. So everything here must be cheap enough to run 1.7 M times.

Critical: nothing here is a quality judgement. `bit_rate` is recorded and never
thresholded -- docs/data/10 rejects filtering on bitrate outright, because
bandwidth correlates with the telephone subset we most need.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from eda.extract import register_metadata

__all__ = ["FFPROBE_COLUMNS", "MP3_COLUMNS", "ffprobe_row", "mp3_header_row"]

FFPROBE_COLUMNS = (
    "probe_ok", "probe_error",
    "container", "format_long_name", "codec_name", "codec_long_name",
    "orig_sr", "orig_channels", "bit_rate", "duration_s",
    "bits_per_raw_sample", "sample_fmt", "encoder", "n_streams",
)

MP3_COLUMNS = ("header_ok", "xing_tag", "lame_tag", "lame_version", "id3_tag")


def _first_audio_stream(streams: list[dict]) -> dict:
    for s in streams:
        if s.get("codec_type") == "audio":
            return s
    return {}


def _num(value: Any, cast):
    if value in (None, "", "N/A"):
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


@register_metadata("ffprobe", FFPROBE_COLUMNS, "probe_ok")
def ffprobe_row(path: Path, probe: Any) -> dict[str, Any]:
    """Container, codec, native rate, channels, bitrate, duration, encoder.

    `orig_sr` and `orig_channels` are the two columns the rest of the package
    depends on: the native plane is `load_audio(path, sample_rate=orig_sr)`,
    which short-circuits the resampler (`training/render.py`'s
    `resample_poly_to` returns the input unchanged when the rates match), so a
    wrong `orig_sr` silently turns the native plane into a resampled one.
    """
    proc = subprocess.run(
        [probe.ffprobe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, check=False, timeout=probe.timeout_s)
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace").strip()[-300:] or "ffprobe returned no output"
        return {**{c: None for c in FFPROBE_COLUMNS}, "probe_ok": False, "probe_error": err}

    meta = json.loads(proc.stdout)
    fmt = meta.get("format", {}) or {}
    streams = meta.get("streams", []) or []
    audio = _first_audio_stream(streams)
    tags = {k.lower(): v for k, v in {**(fmt.get("tags") or {}),
                                      **(audio.get("tags") or {})}.items()}
    return {
        "probe_ok": True,
        "probe_error": None,
        "container": fmt.get("format_name"),
        "format_long_name": fmt.get("format_long_name"),
        "codec_name": audio.get("codec_name"),
        "codec_long_name": audio.get("codec_long_name"),
        "orig_sr": _num(audio.get("sample_rate"), int),
        "orig_channels": _num(audio.get("channels"), int),
        # Caveat: prefer the stream's bitrate; container-level bit_rate counts
        # tag and cover-art bytes, which for a short mp3 with an embedded image
        # is a large fraction of the file.
        "bit_rate": _num(audio.get("bit_rate") or fmt.get("bit_rate"), int),
        "duration_s": _num(audio.get("duration") or fmt.get("duration"), float),
        "bits_per_raw_sample": _num(audio.get("bits_per_raw_sample"), int),
        "sample_fmt": audio.get("sample_fmt"),
        "encoder": tags.get("encoder"),
        "n_streams": len(streams),
    }


@register_metadata("mp3_header", MP3_COLUMNS, "header_ok")
def mp3_header_row(path: Path, probe: Any) -> dict[str, Any]:
    """Xing/Info gapless header, LAME encoder tag, ID3 presence.

    docs/EDA/01 A1 asks for these by name. They are the cheapest provenance
    fingerprint that exists: the LAME version string identifies the encoder
    build, and the Xing/Info frame says whether the file was written by an
    encoder that recorded its own padding -- which is the same header whose
    absence `training/render.py` measured at 1169-1708 extra samples when
    setting `_PAD_TOL`.

    Caveat: scanning only the first `header_bytes` is deliberate. The Xing frame
    sits in the first audio frame; a full-file scan would cost a read of the
    corpus and buy a tag nobody uses.
    """
    with open(path, "rb") as fh:
        head = fh.read(probe.header_bytes)
    if not head:
        return {**{c: None for c in MP3_COLUMNS}, "header_ok": False}
    lame_at = head.find(b"LAME")
    version = None
    if lame_at >= 0:
        raw = head[lame_at + 4:lame_at + 9]
        version = raw.decode("ascii", "ignore").strip() or None
    return {
        "header_ok": True,
        "xing_tag": b"Xing" in head or b"Info" in head,
        "lame_tag": lame_at >= 0,
        "lame_version": version,
        "id3_tag": head[:3] == b"ID3",
    }
