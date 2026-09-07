# 06 — Code-Level Recipes (pulled from Kaggle notebooks)

Source: notebooks pulled via `kaggle kernels pull` on 2026-09-05. These are **actual
implementations**, not prose summaries — ★ throughout unless noted.

| Notebook | Votes | Ref |
|---|---|---|
| BC2026 Distilled-SED | 268 | `tuckerarrants/bc2026-distilled-sed` |
| EfficientNet-B0 Train, BirdCLEF'25 | 953 | `kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25` |
| BC25: separation voice from data | 192 | `kdmitrie/bc25-separation-voice-from-data` |
| Post-Processing / Power Adjustment | 355 | `myso1987/post-processing-with-power-adjustment-for-low-rank` |
| BirdCLEF 2024 1st place inference | 123 | `chemrovkirill/birdclef-2024-1st-place-inference` |

---

## 🔴 1. SED attention head — the fix for "short component in a long file"

`[BC2026 Distilled-SED]`, itself derived from the BirdCLEF 2025 1st-place inference notebook.
The notebook's own justification is *exactly* our problem statement:

> "Instead of Global Average Pooling (GAP), which dilutes a brief vocalization across the full
> 5-second window, the SED head makes **per-frame predictions** and aggregates them via learned
> attention weights. A species that calls for 0.3 seconds gets a sharp attention spike at those
> frames, rather than being averaged with 4.7 seconds of background."

Replace "a species that calls for 0.3 s" with "a fake voice segment occupying 3 s of a 60 s file"
and this is our `VOICE_FAKE_PROB` pooling problem verbatim.

```python
class GeMFreqPool(nn.Module):
    """Generalized Mean pooling over frequency. Learnable p starts at 3.0
    (sharper than mean, softer than max)."""
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):                      # (B, C, F, T)
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p).mean(dim=2)
        return x.pow(1.0 / p)                  # (B, C, T)

# head
self.gem_freq = GeMFreqPool(p_init=3.0)
self.dense = nn.Sequential(nn.Dropout(0.25), nn.Linear(C, 512),
                           nn.ReLU(inplace=True), nn.Dropout(0.5))
self.att = nn.Conv1d(512, num_classes, 1)      # attention over time
self.cla = nn.Conv1d(512, num_classes, 1)      # frame-wise logits

# forward
h = self.gem_freq(h).permute(0, 2, 1)          # (B, T, C)
h = self.dense(h).permute(0, 2, 1)             # (B, 512, T)
norm_att        = torch.softmax(torch.tanh(self.att(h)), dim=-1)
framewise_logits = self.cla(h)
clip_logits      = torch.sum(norm_att * framewise_logits, dim=2)
```

**Transfer**: this is a drop-in structure for all five of our heads. `GeMFreqPool` with learnable
`p` is a principled interpolation between mean and max pooling — precisely the tradeoff we flagged
as open ([survey/10 G5](../survey/10-open-questions.md)). Let the model learn it.

## 🔴 2. Loss: clip + frame-max, 50/50

```python
frame_max_logits = framewise.max(dim=1).values
bce_clip  = F.binary_cross_entropy_with_logits(clip_logits,      lb, reduction="none")
bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
bce = 0.5 * bce_clip + 0.5 * bce_frame
```

and **the same blend at inference**:

```python
p_blend = 0.5 * torch.sigmoid(clip_logits) + 0.5 * torch.sigmoid(frame_max)
```

**Transfer** 🔴: `frame_max` is the "any part of this file is fake ⇒ file is fake" operator, and
`clip` is the "overall character" operator. Our label semantics (`FILE_FAKE = OR over components`,
component fake if *any* generated segment) match this decomposition almost exactly. Train both,
blend both. Cheap and directly on-target.

## 🔴 3. Embedding distillation with stop-gradient

The single largest reported gain in the notebook: **0.898 with distillation vs 0.876 without**
(HGNet-B0) ☆ *(attributed to a Kaggle discussion; treat the number as indicative)*.

```
Loss = BCE(0.5·clip + 0.5·frame-max) + α · MSE(student_emb, frozen_teacher_emb)   # α = 1.0
```

The trick is the **stop-gradient**:

```python
h = self.backbone(x)
distill_emb = self.distill_head(h)      # GAP + Linear -> teacher embedding dim
h_cls = h.detach()                      # <-- SED head does NOT update the backbone
```

> "Without the stop-gradient, the classification loss and distillation loss would fight over the
> backbone's feature representation. With it, the backbone is purely a '[teacher] student'."

**Transfer** 🔴: our equivalent teacher is a frozen SSL encoder — **XLS-R / W2V-BERT for the voice
head, BEATs / EAT / SSLAM for the music head** ([survey/05](../survey/05-models.md)). This gives us
a way to get foundation-model representation quality into a **small, fast** backbone that fits our
3.0 s/file budget — instead of running the big model at inference. That directly addresses the
runtime problem in [survey/05](../survey/05-models.md#runtime-budget-reality-check):
**distill the big frontend into a small student, then throw the distillation head away.**

## 4. Temporal smoothing across windows

```python
from scipy.ndimage import convolve1d
GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
```
Applied across the sequence of 5 s windows within a file. ✅ Legal for us — it uses only
within-file information ([competition/04 rule 2.4](../competition/04-rules.md)).

⚠️ But note it smooths *toward* neighbours, which fights the frame-max "any part fake" semantics.
Test it per head: plausible for presence, questionable for fake.

## 5. Waveform augmentation (concrete parameters)

`[BC2026 Distilled-SED]` — three augmentations, each at **p = 0.5**:

```python
AUG_PROB               = 0.5
AUG_GAIN_DB_RANGE      = (-6.0, 6.0)      # "simulates varying recording distances"
AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)     # "simulates environmental noise"
# + time shift ±0.5 s                     # "simulates temporal misalignment"

def apply_aug(w):
    if np.random.random() < AUG_PROB:
        w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
    if np.random.random() < AUG_PROB:
        # additive noise scaled to a target SNR
        ...
    return w
```

Note how **modest** these are compared to what we planned — gain ±6 dB, SNR 10–30 dB. The heavy
lifting is done by MixUp, not by signal mangling.

## 🔴 6. MixUp — and cross-domain MixUp

```python
MIXUP_PROB   = 0.5
MIXUP_ALPHA  = 0.4
MIXUP_HARD   = True        # union labels, NOT weighted blend
lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
lb  = np.maximum(lb1, lb2) if MIXUP_HARD else (lam*lb1 + (1-lam)*lb2)
```

Two variants run **independently**:
- **Focal–Focal MixUp**: blends two clean training recordings
- 🔴 **Focal–Soundscape MixUp**: blends a clean recording with a *target-domain* window,
  > "bridging the domain gap between clean focal audio and noisy real-world soundscapes."

**Transfer** 🔴: the second variant is the important one. Our analogue is **mixing self-generated
clean audio with real-world degraded audio** (ASVspoof21 LA telephony, MUSAN, real Jamendo tracks)
so the model sees synthetic content *in deployment-like acoustic context*. This is a
domain-bridging technique that does **not** require touching DACON's test set — legal where
pseudo-labeling is not. Add it to [data/06](../data/06-augmentation-spec.md).

Also note `MIXUP_HARD = True` — **union of labels via `max`**, matching `[BirdCLEF 2024, 3rd]` and
our own composition rule ([data/02](../data/02-label-taxonomy.md)). Three independent sources now
agree on `max`.

## 7. Spectrogram augmentation (concrete)

`[EfficientNet-B0 Train, BirdCLEF'25]`:

```python
# Time masking, p=0.5: 1-3 masks, each 5-20 frames wide
# Freq masking, p=0.5: 1-3 masks, each 5-20 bins tall
# Brightness/contrast, p=0.5:
gain = random.uniform(0.8, 1.2); bias = random.uniform(-0.1, 0.1)
spec = torch.clamp(spec * gain + bias, 0, 1)
```
Training config: `efficientnet_b0`, `drop_rate=0.2`, `drop_path_rate=0.2`, AdamW `lr=5e-4`,
`weight_decay=1e-5`, CosineAnnealingLR to `1e-6`, batch 32, `BCEWithLogitsLoss`, 5 folds.

## 8. Mel parameters actually used

| Notebook | sr | n_fft | hop | n_mels | fmin | fmax | shape |
|---|---|---|---|---|---|---|---|
| BirdCLEF'25 B0 | 32000 | 1024 | 512 | 128 | 50 | 14000 | 256×256 |
| BirdCLEF'25 postproc | 32000 | 1024 | 512 | 128 | 50 | 16000 | — |
| BC2026 SED | 32000 | — | 512 | **256** | — | — | — |

⚠️ Scale for our 16 kHz: at hop 512 / sr 32000 a 5 s window gives ~313 frames. At 16 kHz we'd
halve the hop (256) to keep comparable time resolution, and `fmax` is capped at 8000 regardless.

## 🔴 9. Voice detection — Silero VAD

`[BC25: separation voice from data]` uses **Silero VAD** to find human speech inside recordings:

```python
model, (get_speech_timestamps, _, read_audio, _, _) = torch.hub.load(
    repo_or_dir='snakers4/silero-vad', model='silero_vad')
speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True, threshold=0.5)
```
☆ Community finding in that thread: **lowering the threshold from 0.5 to 0.4** improved recall.

**Transfer to us** — three immediate uses:
1. **Verify Pool C is vocal-free** ([data/04](../data/04-sources.md) flagged Jamendo's uploader-
   supplied "instrumental" tag as untrustworthy). This is the verification pass.
2. A cheap, strong **prior for `VOICE_PRESENT_PROB`**, and a sanity check on the presence head.
3. Segment boundaries for **sequential composition** and for segment-level fake labels.

⚠️ `torch.hub.load` downloads at runtime — must be vendored into `model/` for the offline eval
server ([competition/02](../competition/02-submission.md)).

## 10. Energy-based segmentation (dependency-free)

Same notebook, for locating segments without a model:

```python
chunk_len = 0.05                       # seconds
power = wav ** 2
chunk = int(chunk_len * sr)
power = np.pad(power, (0, int(np.ceil(len(power)/chunk)*chunk - len(power))))
power = power.reshape((-1, chunk)).sum(axis=1)
power_dB = 10 * np.log10(power)
x = power_dB - (-50)                   # threshold at -50 dB
intersections = np.where(x[:-1] * x[1:] < 0)[0]   # threshold crossings
```
✅ Useful for the EDA "per-band energy / SNR proxy" view and for the statistics-T style
quality filter ([kaggle/05 A5](05-transferable-playbook.md)).

## 11. Engineering: waveform cache

`[BC2026 Distilled-SED]` caches decoded audio as **`.pt` int16 tensors** rather than re-decoding
each epoch. With ~240 h of 16 kHz mono training audio, int16 caching is both compact and fast.
Worth doing from day one.

---

## ❌ 12. What does not transfer: power adjustment for low-ranked columns

`[Post-Processing / Power Adjustment]` raises low-ranked *columns* to a power to sharpen a
multi-class ranking:

```python
tail_cols = np.argsort(-p.max(axis=0))[top_k:]
p[:, tail_cols] = p[:, tail_cols] ** exponent
```

Irrelevant for us: our five columns are scored by **independent** per-column EER/AUC, so
cross-column reweighting cannot change any score (EER is invariant to monotone per-column
transforms). Recorded here only so nobody re-derives it.

⚠️ Note also the author's own warning — *"While this improves the LB score, please note that it
may be overfitting to the leaderboard."* With Private = Public in our competition, that failure
mode is worse, not better.
