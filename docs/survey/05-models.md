# 05 — Pretrained Model Catalog

Filter applied: must be **publicly accessible**, at minimum **non-commercial use permitted**,
loadable **offline** from weights shipped inside `submit.zip` (≤10 GB), and runnable within
**3.0 s/file** on an L4 (22.4 GiB).

⚠️ **All licenses below are unverified.** Check each model card before use — and remember that
weights we ship are part of the deliverable ([04-rules](../competition/04-rules.md)).

## Speech SSL frontends (→ `VOICE_FAKE_PROB`)

| Model | HF id | Params | Notes |
|---|---|---|---|
| **wav2vec2-XLS-R 300M** | `facebook/wav2vec2-xls-r-300m` | 300M | The default anti-spoofing frontend. 436k h, 128 languages (VoxPopuli, MLS, CommonVoice, BABEL, VoxLingua107). Also 1B / 2B variants — AT-ADD runner-up fused all three scales. |
| **W2V-BERT 2.0** | `facebook/w2v-bert-2.0` | ~580M | **AT-ADD Track 1 winner's frontend.** From Seamless. ⚠️ verify license — Seamless components vary (some CC-BY-NC). |
| **WavLM Large** | `microsoft/wavlm-large` | ~317M | Strong alternative; used as content encoder in several SVC systems. |

## General-audio / music encoders (→ `MUSIC_FAKE_PROB`, presence, routing)

| Model | Source | Notes |
|---|---|---|
| **BEATs** | [microsoft/unilm](https://github.com/microsoft/unilm/blob/master/beats/README.md) | ☆ MIT weights (`BEATs_iter3_plus_AS2M.pt`). AT-ADD winner's **router**. |
| **EAT** | [cwx-worst-one/EAT](https://github.com/cwx-worst-one/EAT) | IJCAI 2024. AT-ADD winner's **non-speech branch** (EAT-large). HF `AutoModel.from_pretrained` supported. |
| **SSLAM** | [ta012/SSLAM](https://github.com/ta012/SSLAM) · [hf](https://huggingface.co/ta012/SSLAM) | ICLR 2025, **AudioSet-2M 50.2 mAP**. Trained on audio *mixtures* → best fit for our polyphonic case. Drop-in for EAT weights. |
| **MERT v1** | `m-a-p/MERT-v1-95M` / `-330M` | 160k h music, RVQ-VAE + CQT teachers, 7 conv + 12 transformer layers, SOTA on 14 MIR tasks. ⚠️ CC-BY-NC (verify). ⚠️ Trained at 24 kHz — **untested at 16 kHz**. ⚠️ MERT-AASIST was the 46.4%-EER cross-generator failure in MusicDET. |
| **PANNs CNN14** | **preinstalled** (`panns-inference==0.1.1`) | mAP 0.439. Zero packaging risk. |
| **CLAP** | LAION / MS | TISMIR used CLAP embeddings + SVM for F1>0.96 in-distribution AI-music detection (but see the 16 kHz caveat). |

## Backends / heads

| Backend | Notes |
|---|---|
| **AASIST / AASIST3** | Graph-attention. The default anti-spoofing backend; used by *both* AT-ADD winners. |
| **Bi-Mamba (Fake-Mamba)** | ASRU 2025, best published In-the-Wild EER 5.85%. [code](https://github.com/xuanxixi/Fake-Mamba) |
| **Mamba-Attention hybrid (XLSR-MamBo)** | ACL 2026 Findings. [code](https://github.com/saki-ciallo/XLSR-MamBo) |
| **SLS classifier** | ACM MM 2024, XLS-R + SLS. |
| **Adapter-MFA / HA-MoE** | Multi-layer fusion / mixture-of-experts over SSL layers (Wav2DF-TSL). |
| Simple MLP over band-SNR + chunk score | The hybrid-stems approach ([03](03-sota-singing-mixed.md)) |

⚠️ Mamba requires `mamba-ssm` / `causal-conv1d` — **CUDA compilation at install time**, and we
have a **10-minute pip budget with no internet beyond pip**. Verify a prebuilt wheel exists for
`torch 2.7.1+cu128 / py3.11 / CUDA 12.8` before committing to a Mamba backbone. Attention/AASIST
backends have no such risk.

## Source separation

| Model | Availability | Notes |
|---|---|---|
| **HT-Demucs** | **preinstalled** (`demucs==4.0.1`, plus `julius`, `diffq`) | Trained at 44.1 kHz. ⚠️ Out-of-domain on our 16 kHz / telephone audio. Free to try. |
| **Mel-Band RoFormer** | [paper](https://arxiv.org/abs/2310.01809) | ★ SOTA MSS; mel-scale **overlapping** subbands; outperforms BS-RoFormer on vocals/drums/other on MUSDB18-HQ. Would need weights shipped. |
| **BS-RoFormer** | [paper](https://arxiv.org/pdf/2309.02612) · [code](https://github.com/lucidrains/BS-RoFormer) | Band-split + RoPE hierarchical transformer, ByteDance. |

**Recommendation**: separation is a *feature extractor* for per-band SNR, not a preprocessing
step ([03](03-sota-singing-mixed.md)). Given that, HT-Demucs (preinstalled, zero packaging cost)
is the right first thing to try; RoFormer only if SNR quality demonstrably limits the head.

## Runtime budget reality check

60 min / 1,200 files = **3.0 s/file** on one L4, all models combined, including audio decode.

| Configuration | Rough feasibility |
|---|---|
| 1× XLS-R 300M + 1× BEATs/EAT, 4 crops | comfortable |
| + HT-Demucs separation per file | tight — measure before committing |
| 3-model XLS-R fusion (0.3B+1B+2B) as in AT-ADD | likely over budget |
| MERT-330M + XLS-R + BEATs + Demucs | over budget |

## Off-the-shelf AI-music detectors

See [02-sota-music.md](02-sota-music.md) — `lofcz/ai-music-detector` (resamples to 16 kHz, the
one most likely to survive our chain), `intrect/artifactnet` (4.2M params, CC BY-NC 4.0).
Treat as **baselines to beat**, not as components, until validated at 16 kHz.
