"""Family ``knnvc``: kNN-VC (bshall/knn-vc, MIT; WavLM-Large features + HiFi-GAN, 16 kHz).

venv: /data/project/private/dacon-venvs/synth-knnvc
  uv pip install torch torchaudio soundfile numpy
Weights come through torch.hub (TORCH_HOME=/data/project/private/dacon-weights/synth/torch).

Voice conversion: a Zeroth utterance (1 or 2 consecutive-in-list utterances of
the same source speaker, 4-15 s) is converted onto a *different* Zeroth target
speaker whose matching set is built from ``--n-ref`` of that speaker's
utterances. ``prompt_speaker`` is the target speaker; ``text_source_utts`` are
the converted source utterances (whose text the output carries).

  python scripts/synth/synth_knnvc.py --n-files 3000 --shard 0/2
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_common as sc  # noqa: E402

FAMILY = "knnvc"
MODEL = "bshall/knn-vc"
LICENCE = "MIT"
SR = 16000


def make_vc_jobs(n_files: int, seed: int, utts: list[sc.Utt], n_ref: int) -> list[sc.Job]:
    rng = random.Random(sc._family_seed(FAMILY, seed))
    spk = sc.by_speaker(utts)
    speakers = sorted(spk)
    rng.shuffle(speakers)
    # fixed reference set per target speaker (deterministic)
    refs = {s: rng.sample([u for u in spk[s] if 3.0 <= u.duration_s <= 12.0] or spk[s],
                          min(n_ref, len(spk[s]))) for s in speakers}
    jobs = []
    for i in range(n_files):
        tgt = speakers[i % len(speakers)]
        while True:
            src_spk = rng.choice(speakers)
            if src_spk != tgt:
                break
        cands = [u for u in spk[src_spk] if u.duration_s <= 15.0]
        a = rng.choice(cands)
        srcs = [a]
        if a.duration_s < 4.0 or rng.random() < 0.45:
            b = rng.choice(cands)
            if b.utt_id != a.utt_id and a.duration_s + b.duration_s <= 15.0:
                srcs = [a, b]
        job = sc.Job(i, f"{FAMILY}_{i:06d}", " ".join(u.text for u in srcs), [u.utt_id for u in srcs],
                     prompt_speaker=tgt, prompt_utts=refs[tgt], seed=seed * 1_000_003 + i)
        jobs.append(job)
    return jobs


def main() -> None:
    ap = sc.standard_argparser(FAMILY, default_files=3000, default_seed=202)
    ap.add_argument("--n-ref", type=int, default=8, help="reference utterances per target speaker")
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True,
                            pretrained=True, device=dev)
    revision = "github bshall/knn-vc@c616845c4e309e24d5927f15adbdf277a3d65358 torch.hub-master prematched=True release-v0.1-weights"

    utts = sc.build_index()
    jobs = sc.shard_jobs(make_vc_jobs(args.n_files, args.seed, utts, args.n_ref), args.shard)
    w = sc.FamilyWriter(FAMILY, MODEL, revision, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    sc.log(f"{FAMILY}: {len(jobs)} jobs in shard, {len(todo)} to do, device={dev}")

    matching: dict[str, torch.Tensor] = {}
    by_id = {u.utt_id: u for u in utts}
    for j in todo:
        if j.prompt_speaker not in matching:
            feats = [knn_vc.get_features(torch.from_numpy(sc.load_prompt(u, SR)[0]).to(dev))
                     for u in j.prompt_utts]
            matching[j.prompt_speaker] = torch.concat(feats, dim=0)
        src = np.concatenate([sc.trim_silence(sc.load_prompt(by_id[u], SR)[0], SR) for u in j.text_utts])
        with torch.no_grad():
            q = knn_vc.get_features(torch.from_numpy(src).to(dev))
            out = knn_vc.match(q, matching[j.prompt_speaker], topk=args.topk)
        w.write(j, out.float().cpu().numpy(), SR)
        w.progress()
        if args.limit and w.n_written >= args.limit:
            break
    print(w.summary())


if __name__ == "__main__":
    main()
