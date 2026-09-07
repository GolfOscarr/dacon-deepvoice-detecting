# 01 — The Sample Contract

What a training sample *is*, as data, before any audio is touched.

---

## 1. The split that everything else follows from

```
sample_spec(rng, manifest, slice, fold) -> SampleSpec     # pure, no I/O, microseconds
render(spec, manifest)                  -> RenderedSample # decode + DSP, milliseconds
```

🔷 This is our own design decision, not inherited from a source recipe. It exists because four
otherwise-expensive guarantees become cheap once the *decision* to build a sample is a serializable
value separate from the audio:

| Property | Without the split | With it |
|---|---|---|
| **Shortcut audit** (`P(T \| L) = P(T)`, [E-S2](../data/07-eda-plan.md)) | decode a corpus, fit a probe | a frequency table over 10⁵ specs, milliseconds, no audio |
| **Eval-set freeze** | store a seed and hope the sampler never changes | store the specs |
| **Byte reproducibility** ([A-S2](../data/06-augmentation-spec.md), [R9](../data/09-risks-and-checks.md)) | replay the whole RNG stream | `render(spec) == render(spec)` |
| **Provenance for DACON** | reconstruct after the fact | the spec set *is* the ledger |

🔴 **The eval-set point is the load-bearing one.** A seed only reproduces a composition while the
sampler code is unchanged, and EER is **not** invariant to cell composition within a class —
measured 0.034 → 0.297 ([validation/02](../validation/02-metric-harness.md)). Any edit to the
sampler would silently move the evaluation set underneath every historical result. Freezing specs
removes that failure mode entirely; freezing a seed does not.

---

## 2. The manifest

One row per **source file**, built by joining the
[provenance ledger](../data/08-build-plan.md#provenance-ledger) to
[`folds.parquet`](../validation/01-split-scheme.md#-the-table-is-keyed-on-components-not-on-composed-files)
on `file_id`. The pipeline reads this and nothing else about the corpus.

```
file_id · path · sha256 · row_kind(component|whole_file)
pool(A-E|null) · cell(1-9|null)
duration_s · orig_sr · orig_channels · container
label_voice_present · label_music_present · label_voice_fake · label_music_fake
artifact_family · source_name · speaker_ref_id · pair_id · dup_group · domain_key
slice(train|val|shadow|probe) · fold(0-4|null) · scheme_version
validity_mask_ref · label_confidence(exact|reported) · aug_strength
```

| Field | Note |
|---|---|
| `row_kind` | 🔴 The field the rest of the pipeline branches on. `component` rows are *drawn and composed*; `whole_file` rows are used as-is |
| `pool` / `cell` | Exactly one is non-null. Component rows carry `pool ∈ {A..E}`; whole-file rows carry `cell ∈ {1..9}` |
| `domain_key` | `source_name × generator_model`, the DOSS capping key ([papers/05](../papers/05-generalization.md)). Null for real components — capping applies to fake domains |
| `validity_mask_ref` | Pointer into the Layer-1 sidecar ([data/10 §1](../data/10-preprocessing-and-filtering.md#1-two-layer-architecture)); the boolean 50 ms mask of usable audio. **Nothing here rewrites audio** |
| `label_confidence` | `exact` for anything we generated or composed, `reported` for scraped audio. Feeds the two-tier loss ([architecture/08 §2](../architecture/08-training-recipe.md#2-the-loss)) |
| `aug_strength` | Per-source multiplier ([A-B7](../data/06-augmentation-spec.md)) — LibriTTS read speech tolerates more mangling than ASVspoof21 LA telephony |

⚠️ The ledger's `pool(A–E)` has no value for a natural song or an AI song, which are whole-file
rows. That is why `pool` is nullable here and why `row_kind` exists rather than being inferred.

---

## 3. `SampleSpec`

The complete, serializable decision. **Every field is drawn before any file is opened.**

```python
@dataclass(frozen=True)
class ComponentDraw:
    file_id: str
    role: str            # "voice" | "music" | "noise"
    source_offset_s: float
    duration_s: float
    target_start_s: float        # where it lands on the sample timeline
    gain_db: float
    is_mixup_partner: bool = False

@dataclass(frozen=True)
class SampleSpec:
    sample_id: int               # stable index within the epoch
    epoch: int
    seed: int                    # (sample_id, epoch, seed) is the whole RNG key
    scheme_version: str          # must match the manifest's

    duration_s: float            # drawn FIRST -- the timeline everything sits on
    cell: int                    # 1-9
    render_mode: str             # "composed" | "whole_file"
    structure: str               # "overlap" | "sequential"
    components: tuple[ComponentDraw, ...]
    crossfade_ms: float          # sequential only; 0 = hard cut

    transforms: tuple[tuple[str, dict], ...]   # ordered (name, sampled params)
    normalize: dict                            # the test-chain draw (A-S1)

    voice_present: int
    music_present: int
    voice_fake: int | None       # None where the component is absent
    music_fake: int | None
    file_fake: int
```

Three properties of this type, each deliberate:

- 🔴 **`duration_s` is field one, not a final crop.** Components are placed on the final timeline,
  so nothing is rendered and thrown away — expensive given the codec round-trips in
  [A-S3](../data/06-augmentation-spec.md) — and frame targets are exact by construction rather
  than a post-hoc re-slice.
- 🔴 **The five labels are a pure function of `cell` and `components`.** `transforms` and
  `normalize` cannot reach them. This is `P(T | L) = P(T)` made structural.
- **`file_fake` is stored but never hand-written.** It is computed once at construction by
  `file_fake_label(...)` from [`metrics.dacon`](../../metrics/AGENTS.md) and asserted thereafter
  ([05 I4](05-invariants.md#1-cheap--spec-level-no-audio-decoded)). The competition defines it as
  OR over *present* components, and there is exactly one implementation of that in this repo.
  ⚠️ Pass **`0`**, not `None`, for an absent component's fake label when calling it: `None`
  survives `np.asarray(None).astype(bool)` as `True` and is only saved by the presence mask
  zeroing it. That is incidental, not a contract.

### Worked example — one cell-6 sample (real voice over fake music)

```python
SampleSpec(
    sample_id=8_213, epoch=3, seed=0, scheme_version="v1",
    duration_s=23.4,
    cell=6, render_mode="composed", structure="overlap",
    components=(
        ComponentDraw("libritts_f_0041", "voice", source_offset_s=12.0,
                      duration_s=23.4, target_start_s=0.0, gain_db=-9.2),
        ComponentDraw("suno_v3_00871",  "music", source_offset_s=41.5,
                      duration_s=23.4, target_start_s=0.0, gain_db=0.0),
    ),
    crossfade_ms=0.0,
    transforms=(("rawboost", {"variant": "linear"}),
                ("gain_jitter", {"db": 2.1}),
                ("noise", {"partition": "musan_noise", "snr_db": 17.4})),
    normalize={"resampler": "soxr_vhq", "container": "mp3", "bitrate": 96,
               "channels": "mono"},
    voice_present=1, music_present=1, voice_fake=0, music_fake=1, file_fake=1,
)
```

The voice sits **9.2 dB below** the music — the quiet-vocal case that
[A-A3](../data/06-augmentation-spec.md) says to over-sample, because detection tracks stem energy
(vocals 65–80% TPR vs accompaniment 97–98%) and models generalize low-SNR → high-SNR but not the
reverse.

---

## 4. `RenderedSample`

What `render()` returns, and the only thing the collator sees.

```python
@dataclass
class RenderedSample:
    wav: Tensor                       # (C, S) float32, as decoded -- NOT downmixed
    sample_rate: int                  # always 16_000
    targets: dict[str, int]           # the five keys of losses.TARGET_FOR_COLUMN
    frame_intervals: dict[str, tuple[tuple[float, float, int], ...]]
    spec: SampleSpec                  # carried through for the ledger
```

⚠️ **`wav` is `(C, S)`, not mono.** The channel policy is `AudioConfig.channels`, applied by
[`models.audio.prepare_waveform`](../../models/AGENTS.md) at the *same call site* in training and
inference. Downmixing in the dataset would fork that policy into two places, disable the
[A-B3](../data/06-augmentation-spec.md) channel augmentations, and make the `mid_side` leak test
([09 B8](../architecture/09-open-questions.md)) impossible to run.

🔴 **`frame_intervals` are absolute times in seconds, never a rasterized frame grid.** Frame rate
is a property of the frontend; the collator does not know it and must not guess. Rasterization
happens inside the branch that owns its own time base. This is the same class of bug as the
`align_time` rule-2.4 violation, which stayed latent only because both shipped configs happen to
use 50 fps ([`PROGRESS.md`](../../PROGRESS.md)) — a hard-coded grid in the collator would
reintroduce it somewhere the existing tests do not look.

Keyed by branch (`voice` / `music` / `file`), values are `(start_s, end_s, label)`. They exist only
for composed rows; whole-file rows carry an empty tuple.

⚠️ **The default loss does not consume these.** `SEDHeadConfig.clip_weight` is committed at **1.0**
(clip-only), so frame targets are *produced but unused* until the **T1** ablation or a MulBS split
needs them ([training/02 §3](../training/02-the-loss.md#3--clipweight--10--the-one-change-worth-engineer-days)).
🔴 **Produce them anyway.** T1 is the first training experiment and cannot run without them, they
are near-free at composition time (the placement arithmetic is already done), and retrofitting them
later means re-rendering the corpus.

⚠️ That asymmetry — frame labels exist on **composed** rows only, which is also where the
composition shortcut lives — is why the frame-loss weight is a *shortcut-exposure* knob and not
merely an accuracy knob. ★ Now measured rather than inferred: EER above 14% at zero concatenation
boundaries ([02 §3](02-sampler.md#3-the-composed-fraction-is-a-constraint-not-a-knob)).
