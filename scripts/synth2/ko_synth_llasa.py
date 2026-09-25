"""Family ``llasa``: HKUSTAudio/Llasa-1B-Multilingual (Llama-3.2-1B LM over XCodec2 speech
tokens, 50 tok/s, 16 kHz, CC-BY-NC-4.0) + HKUSTAudio/xcodec2 (CC-BY-NC-4.0), zero-shot cloning.

venv: /data/project/private/dacon-venvs/synth2-llasa (torch 2.5.0, xcodec2 0.1.5; pinned
transformers 4.48.3 and torchao 0.7.0: transformers 5.x fails to import PreTrainedModel with
torch 2.5 and torchao 0.18 needs torch.int1). Weights under HF_HOME=/data/project/private/dacon-weights/synth2/hf.

Recipe = the model card's prompted TTS: the user turn holds prompt transcript + target text
inside <|TEXT_UNDERSTANDING_START|>...<|TEXT_UNDERSTANDING_END|>, the assistant turn is
prefixed with the prompt's XCodec2 tokens; sampling top_p=1, temperature U(0.75, 0.9) seeded.
The generated tokens are decoded together with the prompt tokens and the prompt's samples are
cut off (as on the card).

  python scripts/synth2/ko_synth_llasa.py --n-files 6000 --shard 0/2 --max-hours 4.5
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ko_common as kc  # noqa: E402

FAMILY = "llasa"
MODEL = "HKUSTAudio/Llasa-1B-Multilingual + HKUSTAudio/xcodec2"
LICENCE = "CC-BY-NC-4.0"
LM, CODEC = "HKUSTAudio/Llasa-1B-Multilingual", "HKUSTAudio/xcodec2"


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=6000, default_seed=1303).parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xcodec2.modeling_xcodec2 import XCodec2Model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rev = Path(snapshot_download(LM)).name[:12] + "+xcodec2@" + Path(snapshot_download(CODEC)).name[:12]
    tok = AutoTokenizer.from_pretrained(LM)
    lm = AutoModelForCausalLM.from_pretrained(LM, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    codec = XCodec2Model.from_pretrained(CODEC).to(dev).eval()
    end_id = tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    sr = 16000

    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts, prompt_max_s=9.0), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, rev, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do, dev={dev}")
    for j in todo:
        if kc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        p = j.prompt_utts[0]
        try:
            pa, _ = kc.load_audio(p.abs_path, sr)
            pa = kc.trim_silence(pa, sr)
            with torch.no_grad():
                code = codec.encode_code(input_waveform=torch.from_numpy(pa).float().unsqueeze(0).to(dev))
                prefix = "".join(f"<|s_{int(i)}|>" for i in code[0, 0, :].tolist())
                text = f"<|TEXT_UNDERSTANDING_START|>{p.text} {j.text}<|TEXT_UNDERSTANDING_END|>"
                chat = [{"role": "user", "content": "Convert the text to speech:" + text},
                        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + prefix}]
                ids = tok.apply_chat_template(chat, tokenize=True, return_tensors="pt",
                                              continue_final_message=True).to(dev)
                n_prefix = code.shape[-1]
                # bound the new tokens by the prompt's speaking rate (50 tokens/s): the Korean
                # model often never emits END, and an unbounded run costs minutes per file
                rate = len(p.text.replace(" ", "")) / max(len(pa) / sr, 1.0)
                est_s = len(j.text.replace(" ", "")) / max(rate, 2.0)
                max_new = int(min(est_s * 1.5 + 1.0, 20.0) * 50)
                t0 = time.time()
                out = lm.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=end_id,
                                  max_new_tokens=max_new, eos_token_id=end_id, do_sample=True, top_p=1.0,
                                  temperature=rng.uniform(0.75, 0.9))
                gen = out[0][ids.shape[1] - n_prefix:]
                n_new = out.shape[1] - ids.shape[1]
                ended = gen[-1].item() == end_id
                if ended:
                    gen = gen[:-1]
                kc.log(f"{j.name}: {n_new} new tokens (cap {max_new}, END={ended}) in {time.time()-t0:.1f}s")
                if not ended:  # runaway: cut at the estimated length, QC (CER) judges it
                    gen = gen[: n_prefix + int(est_s * 1.15 * 50)]
                toks = tok.batch_decode(gen, skip_special_tokens=True)
                sids = [int(t[4:-2]) for t in toks if t.startswith("<|s_") and t.endswith("|>")]
                wav = codec.decode_code(torch.tensor(sids, device=dev).view(1, 1, -1))[0, 0].float().cpu().numpy()
            wav = wav[n_prefix * 320:]  # 50 tokens/s at 16 kHz
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        w.write(j, np.asarray(wav, np.float32), sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
