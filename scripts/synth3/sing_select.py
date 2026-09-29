#!/usr/bin/env python3
"""N2a/N2b: select the songs whose VOCAL stem becomes singing voice (docs/training/15 §3).

    python scripts/synth3/sing_select.py --work /data/project/private/dacon-corpus/interim/_sing_work

One deterministic selection (seed 0) for BOTH sides, written to
`<work>/selection.csv`, which `sing_separate.py` consumes. The rules are the
same on both sides so the separation pipeline is never a label cue:

  * one excerpt per source song, at most `--excerpt-s` (30 s) from the MIDDLE:
    offset = max(0, (duration - 30) / 2)  (FMA clips are 30 s, so they are whole);
  * files are shuffled within their group with seed 0 and taken until the
    group's raw-hour target is met.

Groups:
  * fake -- one per SONICS generator version (manifest `speaker_ref_id`
    sonics/<version>), `--fake-hours` raw each;
  * real -- `fma`: fma_small tracks the v4 manifest already holds as cell-5
    whole files (vocals by the EDA's VAD), then fma_medium tracks (HF mirror
    benjamin-paine/free-music-archive-medium) in `fma_allow.csv`, in vocal-heavy
    top genres, at most `--per-artist` per artist; `musan-music`: every MUSAN
    music file annotated vocals = Y.

With `--aisong <dir>` it instead writes `<work>/selection_aisong.csv` for the
ACE-Step songs (`sing_acestep_gen.py`): side fake, group `acestep-<lang>`,
every song tiled into consecutive 30 s windows (the same excerpt length and
code path; only the number of windows per song differs).

With `--topup <version> --topup-hours H` it writes `<work>/selection_topup.csv`:
more SONICS songs of that version, drawn by the same rule from the songs not
yet in `selection.csv` (used to balance kept hours across versions).

With `--components` it writes `<work>/selection_components.csv`: EVERY v4
music component (row_kind component, pool C or D), the whole file tiled in
consecutive 30 s windows, side real for C and fake for D -- the lead's
vocal-bearing-"instrumental" screen (`sing_components.py` aggregates it).

fma_medium audio is extracted from the parquet shards as the ORIGINAL mp3
bytes to raw/fma/fma_medium_hf/audio/<id[:3]>/<id>.mp3 (no re-encode), so a
track that passes can also be a whole-file cell-5 candidate.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from processing.corpus import _artist_atom  # noqa: E402

CORPUS = pathlib.Path("/data/project/private/dacon-corpus")
MANIFEST = CORPUS / "manifests/strategy-v4/manifest.parquet"
MUSAN = CORPUS / "interim/musan/openslr-17/musan/music"
FMA_META = CORPUS / "interim/fma/fma_small/fma_metadata/tracks.csv"
FMA_ALLOW = CORPUS / "interim/_licences/fma_allow.csv"
MEDIUM = CORPUS / "raw/fma/fma_medium_hf"
# FMA top-level genres whose tracks are mostly sung (Spoken excluded: speech, not singing)
VOCAL_GENRES = {"Hip-Hop", "Pop", "Rock", "Folk", "Soul-RnB", "Country", "Blues", "International"}


def take_hours(df: pd.DataFrame, hours: float, excerpt_s: float, rng) -> pd.DataFrame:
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    cum = np.cumsum(np.minimum(df["duration_s"].to_numpy(), excerpt_s)) / 3600.0
    n = int(np.searchsorted(cum, hours)) + 1
    return df.iloc[:min(n, len(df))].copy()


def fma_medium(tracks: pd.DataFrame, have: set[int], hours: float, per_artist: int,
               excerpt_s: float, rng) -> pd.DataFrame:
    """Pick fma_medium tracks and extract their mp3 bytes."""
    import pyarrow.parquet as pq
    allow = set(pd.read_csv(FMA_ALLOW)["id"].astype(int))
    top = tracks[("track", "genre_top")]
    ok = tracks.index.isin(list(allow)) & top.isin(VOCAL_GENRES) & ~tracks.index.isin(list(have))
    ok &= tracks[("set", "subset")].isin(["small", "medium"]).to_numpy()
    cand = pd.DataFrame({"tid": tracks.index[ok],
                         "artist_id": tracks.loc[ok, ("artist", "id")].to_numpy(),
                         "artist": tracks.loc[ok, ("artist", "name")].to_numpy(),
                         "licence": tracks.loc[ok, ("track", "license")].to_numpy()})
    cand = cand.iloc[rng.permutation(len(cand))]
    cand = cand.groupby("artist_id", sort=False).head(per_artist)
    n = int(hours * 3600 / excerpt_s) + 1
    cand = cand.head(n).set_index("tid")
    want = set(cand.index)
    out = MEDIUM / "audio"
    got = {}
    for shard in sorted((MEDIUM / "data").glob("train-*.parquet")):
        f = pq.ParquetFile(shard)
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
            for a in t:
                tid = int(pathlib.Path(a["path"]).stem)
                if tid not in want:
                    continue
                dst = out / f"{tid:06d}"[:3] / f"{tid:06d}.mp3"
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_suffix(".part")
                    tmp.write_bytes(a["bytes"])
                    tmp.rename(dst)
                got[tid] = dst
    cand = cand.loc[[t for t in cand.index if t in got]]
    print(f"fma_medium: {len(want)} wanted, {len(cand)} found in the shards")
    import soundfile as sf
    durs = []
    for tid in cand.index:
        try:
            durs.append(sf.info(str(got[tid])).duration)
        except Exception:                                   # noqa: BLE001
            durs.append(np.nan)
    rel = [str(got[t].relative_to(CORPUS)) for t in cand.index]
    df = pd.DataFrame({
        "file_id": [f"fma-medium:{t:06d}.mp3" for t in cand.index], "path": rel,
        "duration_s": durs, "corpus": "fma",
        # as processing.corpus: the name atom, else (a name with no [a-z0-9]) the id atom
        "speaker_ref_id": [x if isinstance(x, str) else f"fma_artist_{int(i)}"
                           for x, i in zip(map(_artist_atom, cand["artist"]), cand["artist_id"])],
        "source_name": [f"fma_artist_{int(a)}" for a in cand["artist_id"]],
        "licence": cand["licence"].to_numpy(), "in_manifest": False})
    return df.dropna(subset=["duration_s"])


def aisong(work: pathlib.Path, root: pathlib.Path, excerpt_s: float) -> int:
    import json
    import soundfile as sf
    rows = []
    for line in (root / "gen_log.jsonl").read_text().splitlines():
        g = json.loads(line)
        dur = sf.info(str(root / g["file"])).duration
        for k in range(max(1, int(dur // excerpt_s))):
            rows.append({"file_id": f"aisong-kr-en:{g['file']}", "path": str((root / g["file"]).relative_to(CORPUS)),
                         "duration_s": dur, "corpus": "aisong-kr-en", "speaker_ref_id": None,
                         "source_name": None, "side": "fake", "group": f"acestep-{g['lang']}",
                         "offset_s": round(k * excerpt_s, 3),
                         "excerpt_s": round(min(excerpt_s, dur - k * excerpt_s), 3),
                         "out_prefix": f"acestep-{g['lang']}/{g['id']}_w{k}"})
    df = pd.DataFrame(rows)
    df.index.name = "sel_idx"
    df.to_csv(work / "selection_aisong.csv")
    print(df.groupby("group").agg(windows=("path", "size"), songs=("path", "nunique"),
                                  hours=("excerpt_s", lambda x: x.sum() / 3600)).round(2).to_string())
    return 0


def topup(args, m: pd.DataFrame) -> int:
    cols = ["file_id", "path", "duration_s", "corpus", "speaker_ref_id", "source_name"]
    used = set(pd.read_csv(args.work / "selection.csv")["file_id"])
    fake = m[(m["corpus"] == "sonics") & (m["row_kind"] == "whole_file")
             & (m["speaker_ref_id"] == f"sonics/{args.topup}") & ~m["file_id"].isin(used)][cols]
    sel = take_hours(fake, args.topup_hours, args.excerpt_s, np.random.default_rng([args.seed, 4]))
    sel["side"], sel["group"] = "fake", args.topup
    sel["offset_s"] = np.maximum(0.0, (sel["duration_s"] - args.excerpt_s) / 2).round(3)
    sel["excerpt_s"] = np.minimum(sel["duration_s"], args.excerpt_s).round(3)
    sel["out_prefix"] = sel["group"] + "/" + sel["path"].map(lambda p: pathlib.Path(p).stem)
    sel = sel.reset_index(drop=True)
    sel.index.name = "sel_idx"
    sel.to_csv(args.work / "selection_topup.csv")
    print(f"topup {args.topup}: {len(sel)} songs, {sel['excerpt_s'].sum() / 3600:.2f} h")
    return 0


def components(args, m: pd.DataFrame) -> int:
    import hashlib
    c = m[(m["row_kind"] == "component") & m["pool"].isin(["C", "D"])]
    c = c.sort_values(["pool", "file_id"], kind="stable")        # C (real) first
    rows = []
    for r in c.itertuples(index=False):
        h = hashlib.sha1(r.file_id.encode()).hexdigest()[:16]
        n = max(1, int(np.ceil(r.duration_s / args.excerpt_s - 1e-6)))
        for k in range(n):
            ex = min(args.excerpt_s, r.duration_s - k * args.excerpt_s)
            if k and ex < 4.0:                               # a tail shorter than one chunk
                continue
            rows.append({"file_id": r.file_id, "path": r.path, "duration_s": r.duration_s,
                         "corpus": r.corpus, "pool": r.pool,
                         "side": "real" if r.pool == "C" else "fake",
                         "group": f"{r.pool}-{r.corpus}", "offset_s": round(k * args.excerpt_s, 3),
                         "excerpt_s": round(ex, 3), "out_prefix": f"{r.pool}-{r.corpus}/{h}_w{k}"})
    df = pd.DataFrame(rows)
    df.index.name = "sel_idx"
    args.work.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.work / "selection_components.csv")
    print(df.groupby("group").agg(windows=("path", "size"), files=("file_id", "nunique"),
                                  hours=("excerpt_s", lambda x: x.sum() / 3600)).round(2).to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--fake-hours", type=float, default=7.0, help="raw hours per SONICS version")
    ap.add_argument("--medium-hours", type=float, default=45.0, help="raw fma_medium hours")
    ap.add_argument("--per-artist", type=int, default=8)
    ap.add_argument("--excerpt-s", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aisong", type=pathlib.Path, default=None)
    ap.add_argument("--topup", default=None)
    ap.add_argument("--components", action="store_true")
    ap.add_argument("--topup-hours", type=float, default=0.0)
    args = ap.parse_args()
    if args.aisong:
        return aisong(args.work, args.aisong, args.excerpt_s)

    m = pd.read_parquet(MANIFEST)
    if args.topup:
        return topup(args, m)
    if args.components:
        return components(args, m)
    cols = ["file_id", "path", "duration_s", "corpus", "speaker_ref_id", "source_name"]
    rows = []

    fake = m[(m["corpus"] == "sonics") & (m["row_kind"] == "whole_file")][cols]
    for i, ver in enumerate(sorted(fake["speaker_ref_id"].unique())):
        rng = np.random.default_rng([args.seed, 1, i])
        sel = take_hours(fake[fake["speaker_ref_id"] == ver], args.fake_hours, args.excerpt_s, rng)
        sel["side"], sel["group"] = "fake", ver.split("/", 1)[1]
        rows.append(sel)

    # real: fma_small cell-5 whole files (vocals per the EDA), MUSAN vocals=Y
    small = m[(m["corpus"] == "fma") & (m["row_kind"] == "whole_file") & (m["cell"] == 5)][cols]
    small = small.assign(side="real", group="fma", in_manifest=True)
    rows.append(small)
    ann = []
    for sub in sorted(p for p in MUSAN.iterdir() if (p / "ANNOTATIONS").exists()):
        a = pd.read_csv(sub / "ANNOTATIONS", sep=r"\s+", header=None, usecols=[0, 2],
                        names=["name", "vocals"])
        a["path"] = f"interim/musan/openslr-17/musan/music/{sub.name}/" + a["name"] + ".wav"
        ann.append(a[a["vocals"] == "Y"])
    ann = pd.concat(ann)
    musan = m[(m["corpus"] == "musan-music") & m["path"].isin(ann["path"])][cols]
    musan = musan.drop_duplicates("path").assign(side="real", group="musan-music", in_manifest=True)
    print(f"musan vocals=Y: {len(ann)} annotated, {len(musan)} in the manifest")
    rows.append(musan)

    tracks = pd.read_csv(FMA_META, index_col=0, header=[0, 1])
    have = set(small["path"].str.extract(r"/(\d+)\.mp3$")[0].astype(int))
    rng = np.random.default_rng([args.seed, 2, 1])
    med = fma_medium(tracks, have, args.medium_hours, args.per_artist, args.excerpt_s, rng)
    rows.append(med.assign(side="real", group="fma"))

    df = pd.concat(rows, ignore_index=True)
    df["in_manifest"] = df["in_manifest"].fillna(True).astype(bool)
    df["offset_s"] = np.maximum(0.0, (df["duration_s"] - args.excerpt_s) / 2).round(3)
    df["excerpt_s"] = np.minimum(df["duration_s"], args.excerpt_s).round(3)
    df["out_id"] = df["path"].map(lambda p: pathlib.Path(p).stem)
    assert not df.duplicated(["side", "group", "out_id"]).any(), "output id collision"
    df["out_prefix"] = df["group"] + "/" + df["out_id"]
    rng = np.random.default_rng([args.seed, 3])
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df.index.name = "sel_idx"
    args.work.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.work / "selection.csv")
    print(df.groupby(["side", "group"]).agg(files=("path", "size"),
                                            hours=("excerpt_s", lambda x: x.sum() / 3600),
                                            artists=("speaker_ref_id", "nunique")).round(2).to_string())
    print(f"wrote {args.work / 'selection.csv'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
