#!/usr/bin/env python3
"""Build a small, validated corpus and manifest from the raw S3 store.

    python3 scripts/build_test_corpus.py --root /data/corpus/test-v1
    python3 scripts/build_test_corpus.py --root /data/corpus/test-v1 --extract-only
    python3 scripts/build_test_corpus.py --root /data/corpus/test-v1 --manifest-only

This is the *smoke* corpus: a few hundred files per pool, chosen so the training
pipeline can be exercised end to end on real audio. It is explicitly not a
training corpus -- it is not DOSS-capped, not family-balanced, and it carries a
placeholder in pool D (see below). `scheme_version` is `test-v1` so nothing built
here can be compared against a real run.

*Sources are streamed and truncated, not downloaded whole.* Common Voice English
alone is 88 GiB against a few hundred wanted files, so each tar.gz is read from
S3 through a pipe and abandoned once the quota is met. Nothing is staged.

🔴 *Pool D is a PLACEHOLDER and the corpus is unusable for a real result.*
Nothing in the acquired store contains fake *music*. CompSpoof V2's second
component is environmental sound (EnvSDD / VGGSound / AudioCaps / UrbanSound) --
docs/data/11 says so; docs/data/12's one-line summary does not. But the sampler
cannot run without pool D at all: `check_mix` refuses any cell mix whose
fake-music share is outside [0.2, 0.8] because "the loss is unsound without it".
So spoofed environmental components stand in, and every such row carries
PLACEHOLDER in `source_name`. The repo's own leak tripwire catches the
consequence unaided: on one fold the music EER came out at 0.0201 against a 0.03
floor, and L1 fired with "suspect the split, not the model".

⚠️ *Durations reach 4-10 s, not 4-60 s.* CompSpoof ships fixed 4.00 s clips, and
`sampler.py` filters components to `duration_s >= duration_range[0]`, so a 20 s
draw has no fake music at all -- 0 of 400 pool-D rows reach 20 s. The competition
test set is 4-60 s, so `segmentation: whole_file` at the long end is outside what
this corpus can build.

🔴 *`source_name` granularity is load-bearing, and getting it wrong validates
clean.* It is one of `folds.GROUPING_KEYS`, so a per-corpus value unions every
generator family in that corpus into one inseparable atom through the transitive
closure: `source_name="mlaad"` on 300 files collapsed 9 families into 1 group, 6
groups for 1620 rows, and `build_folds` then reported infeasible at every fold
count. `validate_manifest` passes either way -- it checks column semantics, not
whether anything can rotate. Fake pools therefore use `gen_<family>`, as
`synthetic_manifest` does, and MUSAN's partitions use their real sub-corpora.

*Everything is probed, never assumed.* `duration_s`, `orig_sr`, `orig_channels`
and `container` come from the file, because docs/pipelines P-S1 says never to
trust the extension and the eval server hands us mixed containers.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys
import tarfile
import zipfile

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from training.manifest import POOL_LABELS, REQUIRED_COLUMNS, validate_manifest  # noqa: E402

DEFAULT_BUCKET = "hyeonseop-s3"
DEFAULT_PREFIX = "dacon-deepfake-detection/data/raw"
SCHEME = "test-v1"
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")

#: (subdir, key, suffix, quota, per_group, group_at, must_contain)
#: `per_group`/`group_at` spread a quota over a tar's directory structure --
#: `group_at=-2` is the speaker directory in LibriTTS and Zeroth layouts, so a
#: quota lands across speakers rather than all inside the first one.
TAR_PLAN = (
    ("ljspeech",     "ljspeech/mdc-1.1/payload/ljspeech.tar.gz",             ".wav",  200, 10**9, -1, ""),
    ("zeroth",       "zeroth-korean/openslr-40/payload/zeroth_korean.tar.gz", ".flac", 200, 8,    -2, ""),
    ("libritts",     "libritts-r/openslr-141/payload/dev_clean.tar.gz",      ".wav",  200, 8,    -2, ""),
    ("musan_music",  "musan/openslr-17/payload/musan.tar.gz",                ".wav",  400, 10**9, -1, "musan/music/"),
    ("musan_noise",  "musan/openslr-17/payload/musan.tar.gz",                ".wav",  120, 10**9, -1, "musan/noise/free-sound/"),
    ("musan_noise2", "musan/openslr-17/payload/musan.tar.gz",                ".wav",  120, 10**9, -1, "musan/noise/sound-bible/"),
)


def _aws_stream(bucket: str, key: str) -> subprocess.Popen:
    return subprocess.Popen(["aws", "s3", "cp", f"s3://{bucket}/{key}", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def extract_from_tar(bucket: str, key: str, dest: pathlib.Path, suffix: str,
                     quota: int, per_group: int, group_at: int,
                     must_contain: str) -> int:
    """Stream one tar.gz out of S3, take `quota` members, then stop.

    Early exit is the point: closing the stream once the quota is met avoids
    decompressing the rest of a 10-88 GiB archive.
    """
    dest.mkdir(parents=True, exist_ok=True)
    proc = _aws_stream(bucket, key)
    seen: dict[str, int] = {}
    taken = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tf:
            for member in tf:
                if taken >= quota:
                    break
                if not member.isfile() or not member.name.endswith(suffix):
                    continue
                if must_contain and must_contain not in member.name:
                    continue
                parts = member.name.split("/")
                group = parts[group_at] if len(parts) > abs(group_at) else "_"
                if seen.get(group, 0) >= per_group:
                    continue
                with tf.extractfile(member) as src:
                    (dest / pathlib.Path(member.name).name).write_bytes(src.read())
                seen[group] = seen.get(group, 0) + 1
                taken += 1
    finally:
        proc.stdout.close()
        proc.kill()
    return taken


def extract_mlaad(bucket: str, prefix: str, dest: pathlib.Path,
                  n_generators: int, per_generator: int) -> int:
    """Pool B. MLAAD is stored as individual files, structure intact.

    🔴 `fake/<language>/<generator>/` IS the split axis and the DOSS capping key,
    so the generator survives into the local filename. docs/data/12 records an
    earlier attempt that flattened these to basenames and destroyed it.
    """
    dest.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(
        ["aws", "s3", "ls", "--recursive", f"s3://{bucket}/{prefix}/mlaad/v9/payload/"],
        capture_output=True, text=True, check=True).stdout.splitlines()
    keys = [ln.split()[-1] for ln in listing if ln.strip().endswith(".wav")]
    chosen: dict[str, list[str]] = {}
    for k in sorted(keys):
        parts = k.split("/")
        gen = f"{parts[-3]}/{parts[-2]}"
        if gen not in chosen and len(chosen) >= n_generators:
            continue
        bucket_list = chosen.setdefault(gen, [])
        if len(bucket_list) < per_generator:
            bucket_list.append(k)
    taken = 0
    for gen, ks in chosen.items():
        for k in ks:
            tag = gen.replace("/", "__")
            out = dest / f"{tag}__{pathlib.Path(k).name}"
            subprocess.run(["aws", "s3", "cp", f"s3://{bucket}/{k}", str(out), "--quiet"],
                           check=True)
            taken += 1
    return taken


def extract_compspoof_env(bucket: str, prefix: str, dest: pathlib.Path,
                          quota: int, per_group: int) -> int:
    """Pool D's PLACEHOLDER: spoofed ENVIRONMENTAL components, not fake music.

    The layout carries the labels -- `env_sources/<corpus>/spoofed/<generator>/` --
    so the label CSV is not needed, and in the challenge splits it could not be
    used anyway: every metadata column but `audio_path` and `label` is redacted.
    Note the directory is `spoofed`, not `spoof`.
    """
    dest.mkdir(parents=True, exist_ok=True)
    key = f"{prefix}/compspoof-v2/ESDD2/payload/eval_source.tar.gz"
    proc = _aws_stream(bucket, key)
    seen: dict[str, int] = {}
    taken = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tf:
            for member in tf:
                if taken >= quota:
                    break
                if not member.isfile() or not member.name.endswith(AUDIO_SUFFIXES):
                    continue
                parts = member.name.split("/")
                if "env_sources" not in parts or "spoofed" not in parts:
                    continue
                i = parts.index("env_sources")
                # the GENERATOR is the family key, as MLAAD's generator dir is;
                # the corpus above it would collapse distinct generators into one
                family = f"{parts[i + 1]}-{parts[i + 3] if len(parts) > i + 4 else 'unknown'}"
                if seen.get(family, 0) >= per_group:
                    continue
                with tf.extractfile(member) as src:
                    (dest / f"{family}__{pathlib.Path(member.name).name}").write_bytes(src.read())
                seen[family] = seen.get(family, 0) + 1
                taken += 1
    finally:
        proc.stdout.close()
        proc.kill()
    return taken


def extract_rirs(bucket: str, prefix: str, dest_root: pathlib.Path, quota: int) -> int:
    """Pool E, from the RIRS zip, which must be local before it can be read.

    ⚠️ `simulated_rirs` is deliberately skipped: an impulse response is a
    convolution kernel, not ambient audio, and pool E is the noise role.
    """
    local = dest_root / "rirs_noises.zip"
    if not local.exists():
        subprocess.run(
            ["aws", "s3", "cp",
             f"s3://{bucket}/{prefix}/rirs-noises/openslr-28/payload/rirs_noises.zip",
             str(local), "--quiet"], check=True)
    taken = 0
    with zipfile.ZipFile(local) as z:
        for sub, tag in (("pointsource_noises", "rirs_pointsource"),
                         ("real_rirs_isotropic_noises", "rirs_isotropic")):
            out = dest_root / "audio" / tag
            out.mkdir(parents=True, exist_ok=True)
            n = 0
            for name in z.namelist():
                if n >= quota:
                    break
                if f"/{sub}/" not in name or not name.endswith(".wav"):
                    continue
                (out / pathlib.Path(name).name).write_bytes(z.read(name))
                n += 1
            taken += n
    return taken


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: pathlib.Path) -> dict:
    info = sf.info(str(path))
    return dict(duration_s=float(info.duration), orig_sr=int(info.samplerate),
                orig_channels=int(info.channels), container=info.format.lower())


def rows_for(root: pathlib.Path, subdir: str, pool: str, *, speaker_of, source_of,
             family_of=None, domain_of=None) -> list[dict]:
    out: list[dict] = []
    for path in sorted((root / "audio" / subdir).glob("*")):
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        vp, mp, vf, mf = POOL_LABELS[pool]
        family = family_of(path) if family_of else None
        out.append({
            "file_id": f"{subdir}:{path.stem}",
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "row_kind": "component",
            "pool": pool,
            "cell": None,
            **describe(path),
            "label_voice_present": vp, "label_music_present": mp,
            "label_voice_fake": vf, "label_music_fake": mf,
            "artifact_family": family,
            "source_name": source_of(path),
            "speaker_ref_id": speaker_of(path),
            "pair_id": None,
            "dup_group": None,
            "domain_key": domain_of(path) if domain_of else None,
            "slice": "train", "fold": None, "scheme_version": SCHEME,
            "validity_mask_ref": None,
            # Same convention as `synthetic_manifest`: a generator directory is
            # definitive, a publisher's folder layout is reported.
            "label_confidence": "exact" if family is not None else "reported",
            "aug_strength": 0.0,
        })
    return out


def build_manifest(root: pathlib.Path) -> pd.DataFrame:
    """Assemble and validate. Raises rather than writing a bad manifest."""
    rows: list[dict] = []
    # -- pool A: real voice ------------------------------------------------- #
    rows += rows_for(root, "ljspeech", "A",
                     speaker_of=lambda p: "ljspeech_LJ",        # single-speaker corpus
                     source_of=lambda p: "ljspeech")
    rows += rows_for(root, "zeroth", "A",                       # <spk>_<grp>_<utt>.flac
                     speaker_of=lambda p: f"zeroth_{p.stem.split('_')[0]}",
                     source_of=lambda p: "zeroth-korean")
    rows += rows_for(root, "libritts", "A",                     # <spk>_<chapter>_...
                     speaker_of=lambda p: f"libritts_{p.stem.split('_')[0]}",
                     source_of=lambda p: "libritts-r")
    # -- pool B: fake voice ------------------------------------------------- #
    rows += rows_for(root, "mlaad", "B",
                     speaker_of=lambda p: f"mlaad_{p.stem.split('__')[1]}",
                     source_of=lambda p: f"gen_{p.stem.split('__')[1]}",
                     family_of=lambda p: p.stem.split("__")[1],
                     domain_of=lambda p: f"mlaad::{p.stem.split('__')[0]}/{p.stem.split('__')[1]}")
    # -- pool C: real music ------------------------------------------------- #
    rows += rows_for(root, "musan_music", "C",
                     speaker_of=lambda p: "musan_" + "-".join(p.stem.split("-")[:-1]),
                     source_of=lambda p: "musan-music-" + "-".join(p.stem.split("-")[1:-1]))
    # -- pool D: PLACEHOLDER, see the module docstring ---------------------- #
    rows += rows_for(root, "compspoof_env_spoof", "D",
                     speaker_of=lambda p: f"compspoof_{p.stem.split('__')[0]}",
                     source_of=lambda p: f"gen_PLACEHOLDER_{p.stem.split('__')[0]}",
                     family_of=lambda p: p.stem.split("__")[0],
                     domain_of=lambda p: f"compspoof::{p.stem.split('__')[0]}")
    # -- pool E: noise ------------------------------------------------------ #
    for subdir in ("musan_noise", "musan_noise2"):
        rows += rows_for(root, subdir, "E",
                         speaker_of=lambda p: "musan_" + "-".join(p.stem.split("-")[:-1]),
                         source_of=lambda p: "musan-noise-" + "-".join(p.stem.split("-")[1:-1]))
    for subdir in ("rirs_pointsource", "rirs_isotropic"):
        rows += rows_for(root, subdir, "E",
                         speaker_of=lambda p, t=subdir: t,
                         source_of=lambda p, t=subdir: t)

    df = pd.DataFrame(rows)[list(REQUIRED_COLUMNS)]
    for col in ("label_voice_present", "label_music_present",
                "label_voice_fake", "label_music_fake", "fold"):
        df[col] = df[col].astype("Int64")

    # 🔴 `validate_manifest` does NOT check pool coverage, and a manifest missing
    # a whole pool passes it clean: an `audio/` subdirectory renamed out from
    # under this script dropped all 400 pool-C rows and the result validated,
    # 1687 rows with no real music in it at all. The sampler needs every pool --
    # `check_mix` refuses a cell mix whose music share is outside [0.2, 0.8], and
    # `_draw_component` raises on an empty role -- so the shortfall would surface
    # much later, as an infeasible fold table or a refused draw.
    missing = sorted({"A", "B", "C", "D", "E"} - set(df["pool"].dropna().unique()))
    if missing:
        raise ValueError(
            f"manifest has no rows for pool(s) {missing}. Expected audio under "
            f"<root>/audio/ for every pool; `validate_manifest` would accept this "
            f"and the sampler would refuse to draw from it later")

    return validate_manifest(df)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=pathlib.Path, required=True,
                   help="corpus root; audio lands under <root>/audio/")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--extract-only", action="store_true", help="fetch audio, no manifest")
    p.add_argument("--manifest-only", action="store_true",
                   help="build the manifest from audio already under --root")
    p.add_argument("--mlaad-generators", type=int, default=10)
    p.add_argument("--mlaad-per-generator", type=int, default=30)
    p.add_argument("--placeholder-quota", type=int, default=400)
    p.add_argument("--rirs-quota", type=int, default=90)
    args = p.parse_args()

    root: pathlib.Path = args.root
    if not args.manifest_only:
        for subdir, key, suffix, quota, per_group, group_at, must in TAR_PLAN:
            n = extract_from_tar(args.bucket, f"{args.prefix}/{key}",
                                 root / "audio" / subdir, suffix, quota,
                                 per_group, group_at, must)
            print(f"  {subdir:<16} {n:>4} files")
        print(f"  {'mlaad':<16} "
              f"{extract_mlaad(args.bucket, args.prefix, root / 'audio' / 'mlaad', args.mlaad_generators, args.mlaad_per_generator):>4} files")
        print(f"  {'compspoof(PH)':<16} "
              f"{extract_compspoof_env(args.bucket, args.prefix, root / 'audio' / 'compspoof_env_spoof', args.placeholder_quota, 100):>4} files")
        print(f"  {'rirs':<16} {extract_rirs(args.bucket, args.prefix, root, args.rirs_quota):>4} files")
    if args.extract_only:
        return 0

    df = build_manifest(root)
    out = root / "manifest.parquet"
    df.to_parquet(out, index=False)
    print(f"\nVALIDATED -> {out}  ({len(df)} rows)")
    print(df.groupby(["pool", "source_name"]).agg(
        n=("file_id", "size"),
        hours=("duration_s", lambda s: round(s.sum() / 3600, 2))).to_string())
    print("\n⚠️ pool D is a PLACEHOLDER (spoofed environmental, not fake music) "
          "and durations reach 4-10 s, not 4-60 s. See the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
