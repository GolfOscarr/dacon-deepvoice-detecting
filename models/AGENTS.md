# Using the model

Everything that turns audio into the five submission logits.

**The one rule: candidates A and B are the same class.** `DeepVoiceNet` is built from a
`ModelConfig`; A names one frontend and points all five branches at it, B names two and gives
the file branch both. If you find yourself writing a second model class, the config is missing
a field.

Why each choice was made: [`docs/architecture/`](../docs/architecture/README.md).
This file is how to *use* it.

| Module | Use it for |
|---|---|
| `models.config` | Loading and validating a config. Rejects unknown keys |
| `models.frontends` | Encoder wrappers, normalised to one `(B, T, D)` contract. `stub`, `beats` and `xlsr_300m` are wired |
| `models.heads` | The SED head and frequency pooling |
| `models.model` | `DeepVoiceNet`, checkpoint save/load |
| `models.losses` | The masked multi-task objective |
| `models.outputs` | Logits → probabilities; cross-window pooling |
| `models.audio` | Channel policy and band restriction (pass `lengths`) |

---

## 🔴 The branches are outputs, not input types

The single most likely misreading. **Every file passes through every branch; nothing is routed.**
The "voice branch" is the branch that *produces* `VOICE_FAKE_PROB`, not the branch for
voice-only files. Routing by input type is candidate D, rejected because a 혼합 file has both
components present and independently fake, which leaves a hard switch undefined.

---

## The whole path, once

Everything else on this page is a detail of these six lines.

```python
import torch

from models.audio import prepare_waveform
from models.config import load_model_config
from models.model import DeepVoiceNet

cfg = load_model_config("configs/b_stub.yaml")
model = DeepVoiceNet(cfg).eval()

stereo = torch.randn(2, 2, 16_000 * 5)                    # (B, C, S) as decoded
wav = prepare_waveform(stereo, cfg.audio)                 # -> (B, S) per audio.channels
lengths = torch.tensor([16_000 * 5, 16_000 * 3])          # real samples per file

with torch.no_grad():
    probs = model.submission_probs(model(wav, lengths))   # {column: (B,) float64}
```

⚠️ Two of those six lines are easy to skip and both are wrong to skip. `prepare_waveform`
applies the channel policy — the model takes mono `(B, S)` and will reject `(B, C, S)`. And
`lengths` is what makes the frame mask correct; without it every padded frame counts as real
audio.

---

## Build a model

```python
from models.config import load_model_config
from models.model import DeepVoiceNet

cfg = load_model_config("configs/b_stub.yaml")   # B's shape, weightless frontends
model = DeepVoiceNet(cfg).eval()
print(model.columns)          # branch name -> submission column
```

⚠️ Real checkpoints are **not wired yet**, which is why the snippet loads `b_stub.yaml` rather
than `b_three_branch.yaml`. `build_frontend` raises for anything but `stub`, because the licences
for SSLAM, EAT and W2V-BERT 2.0 are unverified
([09 C1–C3](../docs/architecture/09-open-questions.md)) and a licence forbidding third-party
provision makes a checkpoint unusable *at all* here — so downloading first and checking later is
the wrong order. `b_stub.yaml` is candidate B's exact shape with weightless encoders; swap the
file once the licences clear.

## Run it

```python
import torch

wav = torch.randn(2, 16_000 * 5)          # (B, samples), mono, 16 kHz
lengths = torch.tensor([16_000 * 5, 16_000 * 3])
out = model(wav, lengths)

out["file"]["clip_logits"]                # (B,)  attention-pooled
out["file"]["frame_logits"]               # (B, T) per-frame evidence
out["file"]["attention"]                  # (B, T) where the model looked
```

`lengths` is what makes the frame mask correct. Omit it only when every file in the batch is
the same length — otherwise padding is treated as audio and a file's score starts depending on
what is batched with it, which rule 2.4 forbids.

🔴 **The mask travels with the output**, as `out[branch]["mask"]`, and `submission_probs` and
`multitask_loss` both use it automatically. That is deliberate: `clip_logits` are already
padding-safe because attention excludes masked frames, but `frame_max` is not — and a
`frame_max` taken over padded frames moved a submitted probability from 0.519 to 0.847 for the
same file, depending only on what shared its batch. Do not recompute `frame_max` yourself
without passing a mask.

⚠️ `lengths` also decides where the *waveform* stops: `Frontend.forward` zeroes everything past
each row's `lengths` before encoding. `frames_for` rounds up, so a row whose length is not a
multiple of the frontend hop has a last valid frame that is **part padding** and is masked *in* —
without the zeroing, the pad content reached the score through it (measured: 1.35e-3 on a
submitted probability). It is a no-op on the shipped path, where padding is zeros on both sides;
what it buys is that the guarantee no longer depends on everyone remembering to pad with zeros.

🔴 Test fixtures must use lengths that are **not** whole multiples of the hop — `tests/test_model.py::ragged`.
Every padding guard here once used `lengths = SR * 4` (200 whole 320-sample frames), so the tests
defending this repo's most-repeated defect class had never exercised a partial boundary frame,
which is the normal case for `U(4, 60)` s audio.

## Produce submission numbers

```python
probs = model.submission_probs(out)      # {column: (B,) float64 in (0, 1)}
print(sorted(probs))
```

`submission_probs` honours `file_head.mode`: `learned` reads the file branch, while `noisy_or`
and `max` combine the component and presence columns analytically. G3 records the FILE
construction as an open question with no prior art, so all three must genuinely differ — and a
test asserts they do.

🔴 Never blend two sigmoids. `branch_logit` blends `clip` and `frame_max` in **logit space** and
`to_probability` squashes once, in float64, with a non-saturating map. Saturating the operating
point took EER 0.0950 → 0.3017 in measurement, and rank normalisation — the usual fix for the
ties it creates — is forbidden by rule 2.4.

## Train

```python
from models.config import LossConfig
from models.losses import multitask_loss

targets = {
    "voice_fake":    torch.tensor([1.0, 0.0]),
    "music_fake":    torch.tensor([0.0, 1.0]),
    "file_fake":     torch.tensor([1.0, 1.0]),
    "voice_present": torch.tensor([1.0, 0.0]),   # ground truth, not predictions
    "music_present": torch.tensor([1.0, 1.0]),
}
total, parts = multitask_loss(out, targets, cfg, LossConfig())
total.backward()
```

`voice_present` / `music_present` are **ground truth**. They mask the component losses so that a
voice-fake loss is never taken on a music-only file — mirroring the official metric, which
computes Voice EER only over voice-present files. `parts` is a per-head breakdown for the
experiment ledger.

✅ `LossConfig().weights` is **metric-proportional** — File `.45` / Music `.27` / Voice `.18` /
presence `.05` each. An earlier default weighted all five equally, inherited from PC-Mix whose
metric weighted its components equally and ours does not. Set by argument, not sweep
([training/02 §4](../docs/training/02-the-loss.md#4-per-head-weights--metric-proportional-by-argument));
[09 B11](../docs/architecture/09-open-questions.md) is closed.

🔴 **All five keys are required.** `LossConfig.__post_init__` rejects a partial `weights` dict,
however the config was built. `weights: {file: 0.45}` used to load clean — every key it carried
*was* known — and `multitask_loss` filled the four absent heads in at a `.get(..., 1.0)` fallback,
training both presence heads at 20× the 0.05 the metric gives them. The fallback is gone; the loss
indexes.

⚠️ The weight *in effect* is `w_c / p_c`, not `w_c` — `_masked_mean` divides by the present-count,
so a masked head is amplified by how rare its component is. **Both are logged**: `parts` carries
`<branch>/p_c` and `<branch>/w_eff` beside each head's loss, and `training.loop` averages them into
the pass row. Keys with a `/` are diagnostics, not loss terms. `docs/training/02 §4` says outright
not to tune `w_c` without reading `w_eff`, and T2 cannot be read as specified without it.

🔴 **The clip-vs-`frame_max` blend is not a loss knob.** It is `SEDHeadConfig.clip_weight`, in
the *model* config, and the loss reads that same field — so training and inference cannot
disagree about the objective. ⚠️ It now defaults to **1.0 (clip only)**: supervising the utterance
and frame levels through one shared head measured 0.71–3.63 EER points worse than utterance-only
([training/02 §3](../docs/training/02-the-loss.md#3--clipweight--10--the-one-change-worth-engineer-days)).
At 1.0 `frame_max` does not reach the submitted score, which makes the rule-2.4 `frame_max` guards
vacuous — the tests force the blend on rather than inheriting the default. A separate loss-side `frame_weight` used to exist and was
documented as the frame-*supervision* weight of `04 §4`, a different quantity; anyone tuning it
per that section was tuning the blend.

## Inspecting what you built

```python
print(model.columns)                      # branch -> submission column
print(f"{model.n_parameters():,} params, {model.n_parameters(True):,} trainable")
print({name: fe.fps for name, fe in model.frontends.items()})
```

⚠️ A large gap between total and trainable is expected and is the point: `freeze: true` means
the encoder contributes parameters but no gradients. If the two numbers are equal on a config
that says `freeze: true`, something is wrong — there is a test for exactly that.

## Checkpoints

```python
from models.model import load_checkpoint, save_checkpoint

save_checkpoint(model, "model/model.pt")
restored = load_checkpoint("model/model.pt")
```

The config is stored beside the weights and the model is rebuilt from it before a **strict**
`load_state_dict`. That is what makes the load an assertion rather than a coincidence — and it is
the guard `script.py` needs, since a silently mis-shaped model would otherwise write a
well-formed `submission.csv` full of 0.5 and score exactly 0.5000 without raising.

🔴 A pretrained frontend (`beats`, `xlsr_300m`) reads its checkpoint **at construction**, from
the stored `frontends.<name>.weights` — the training machine's absolute path, which does not
exist on the offline test server. `load_checkpoint(path, weights={name: dir})` replaces it, and
`script.py` passes `shipped_weights(model_dir)`: one directory per frontend under
`model/weights/<frontend>/` (the BEATs `.pt`; the XLS-R snapshot with `config.json`). The strict
load then overwrites every weight, so which copy is read changes no number.

## Windows, if you ever tile

⚠️ **`DeepVoiceNet` refuses `segmentation.mode: tiling`** — it consumes a whole waveform, and
windowing belongs to the inference script. `aggregate_windows` is therefore a standalone helper
for that script rather than something the model calls:

```python
from models.config import AggregationConfig
from models.outputs import aggregate_windows

window_logits = torch.randn(4, 12)        # (files, windows), from your own windowing
file_logits = aggregate_windows(window_logits, AggregationConfig(kind="topk_mean", k=3))
```

⚠️ **Every order-statistic aggregator carries a duration bias** — measured spread over 1–12
windows on identical content:

| `max` | `top-k mean` | `quantile` | `mean` |
|---|---|---|---|
| 1.63 | 1.18 | 1.08 | **0.004** |

Long files score higher than short ones on identical content, inside a ranking that pools
4–60 s files. Only `mean` is neutral, and it dilutes — which is the problem the SED head exists
to solve. Both shipped configs therefore use `whole_file`, where the dilemma does not arise.


---

## What the config does *not* do yet

🔴 A knob that validates and then does nothing is worse than a missing knob — it makes an
ablation report a difference it never tested. So anything unimplemented **raises** rather than
being ignored:

| Setting | Behaviour today |
|---|---|
| `frontends.layers` (truncation) | ✅ `beats` and `xlsr_300m` delete the blocks past it; ❌ the stub raises |
| `frontends.adapter.kind != none` | ✅ `lora` on `beats`/`xlsr_300m` (`fc1`/`fc2` map to XLS-R's `intermediate_dense`/`output_dense`; a target matching nothing raises); ❌ the stub raises |
| `segmentation.mode: tiling` | ❌ `DeepVoiceNet` raises — windowing belongs to the inference script |
| `frontends.freeze` | ✅ applied (encoder frozen; the GeM exponent stays trainable — it is ours) |
| `distill.stop_gradient` | ✅ applied — branch heads get detached features |
| `file_head.mode` | ✅ all three implemented |
| `audio.band_hz` | ✅ applied, as a brick wall in the rFFT domain |
| `audio.channels` | ✅ applied by `models.audio.prepare_waveform`, which the data path calls |
| `runtime.*`, `audio.min/max_seconds`, `aggregation.*` | consumed by the inference script / data loader |

`tests/test_model.py::test_no_config_field_is_silently_ignored` enforces this: every field must
be read by a model module or listed in `models.model.CONSUMED_ELSEWHERE` with its owner.
