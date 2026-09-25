"""zh family ``chatterbox-mtl``: ResembleAI/chatterbox multilingual (T3 0.5B Llama speech-token
LM + S3Gen flow matching + HiFT, 24 kHz, MIT), zero-shot cloning with language_id="zh".

venv: /data/project/private/dacon-venvs/synth2-chatterbox (built by the Korean S2 track; used
read-only). Weights: HF ResembleAI/chatterbox under HF_HOME=/data/project/private/dacon-weights/synth2/hf.
Chinese input goes through the package's own front end (pkuseg segmentation + Cangjie codes).
`generate` always applies the Perth implicit watermark; it is replaced by a no-op here, as
round 1 disabled OpenVoice's wavmark, so a watermark cannot become a shortcut feature for the
detector (the audio is internal training data for a fake-voice detector, never distributed).

Prompt: one zh REAL utterance (3-10 s, no transcript needed); text: 1-2 FLEURS sentences.
Per-file seeded draws: exaggeration U(0.35, 0.7), cfg_weight U(0.3, 0.6), temperature U(0.7, 0.9).

  python scripts/synth2/zh_synth_chatterbox.py --n-files 4000 --max-hours 7.2
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zh_common as zc  # noqa: E402

FAMILY = "chatterbox-mtl"
MODEL = "ResembleAI/chatterbox (multilingual t3_mtl23ls_v2)"
LICENCE = "MIT"


def main() -> None:
    ap = zc.standard_argparser(FAMILY, default_files=4000, default_seed=525)
    args = ap.parse_args()

    import perth
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    class _NoWatermark:
        def apply_watermark(self, wav, sample_rate):
            return wav
    # this venv's perth leaves PerthImplicitWatermarker = None (its optional dep is missing),
    # which breaks the constructor; the watermark is disabled anyway (see the docstring)
    perth.PerthImplicitWatermarker = _NoWatermark
    # the zh front end loads Cangjie5_TC.json via hf_hub_download(cache_dir=<snapshot dir>),
    # which misses offline; zh text then goes in as raw hanzi and comes out as gibberish
    # (CPU dry run: CER 0.98). Resolve it from the normal HF cache instead, and check it loaded.
    import chatterbox.models.tokenizers.tokenizer as cb_tok
    cb_tok.hf_hub_download = lambda repo_id, filename, cache_dir=None, **kw: hf_hub_download(repo_id, filename)
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    snap = Path(snapshot_download("ResembleAI/chatterbox", allow_patterns=["ve.pt"]))
    tts = ChatterboxMultilingualTTS.from_pretrained(device=dev)
    sr = tts.sr
    tts.watermarker = _NoWatermark()
    n_cj = len(tts.tokenizer.cangjie_converter.word2cj)
    assert n_cj > 1000, f"Cangjie map not loaded ({n_cj} entries)"
    assert tts.tokenizer.cangjie_converter.segmenter is not None, "pkuseg missing"

    prompts = zc.load_prompts(with_text=False)
    jobs = zc.shard_jobs(zc.make_jobs(FAMILY, args.n_files, args.seed, prompts), args.shard)
    w = zc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    zc.log(f"{FAMILY}: {len(prompts)} prompts, {len(jobs)} jobs in shard, {len(todo)} to do, sr={sr}, dev={dev}")

    for j in todo:
        if zc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        # chatterbox's AlignmentStreamAnalyzer registers forward hooks on the attention layers at
        # every generate() and never removes them; each stale hook copies attention to CPU at every
        # step, so throughput decays linearly (measured: half speed after ~225 files per process).
        for layer in tts.t3.tfmr.layers:
            layer.self_attn._forward_hooks.clear()
        try:
            wav = tts.generate(j.text, language_id="zh", audio_prompt_path=str(j.prompt.abs_path),
                               exaggeration=rng.uniform(0.35, 0.7), cfg_weight=rng.uniform(0.3, 0.6),
                               temperature=rng.uniform(0.7, 0.9))
        except Exception as e:
            zc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        w.write(j, wav[0].float().cpu().numpy(), sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
