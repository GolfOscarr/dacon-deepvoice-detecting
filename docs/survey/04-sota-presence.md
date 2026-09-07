# 04 — Voice / Music Presence Heads

Combined weight **0.10** (0.05 each), metric **ROC-AUC** (easier than 1−EER). Near-solved by
AudioSet-pretrained tagging. **Do it correctly, cheaply, early — then stop investing.**

## Key facts

- Both `VOICE_PRESENT_PROB` and `MUSIC_PRESENT_PROB` map to AudioSet classes directly
  (`Speech`, `Singing`, `Music`, and instrument classes).
- ☆ AudioSet is itself **>40% speech and music** clips — the two classes we need are the two
  best-represented in the pretraining data.
- ROC-AUC is rank-only → no threshold calibration needed.
- ⚠️ Our presence predictions do **not** decide which samples enter the Voice/Music EER pools —
  the organizers use their own ground truth. So presence errors don't corrupt the fake heads.

## Model options

| Model | AudioSet mAP | Notes |
|---|---|---|
| ★ **SSLAM** (ICLR 2025) | **0.502** | Self-supervised on audio *mixtures* with a Source Retention Loss — explicitly designed for **polyphonic overlapping sources**, i.e. our mixed case. ViT-B. [github](https://github.com/ta012/SSLAM) · [hf](https://huggingface.co/ta012/SSLAM) · inference code identical to EAT |
| ★ **EAT** (IJCAI 2024) | ~0.48–0.49 | [github](https://github.com/cwx-worst-one/EAT); AT-ADD Track 2 winner used EAT-large for non-speech |
| ★ **BEATs** | ~0.48 | AT-ADD Track 2 winner used a **frozen BEATs classifier as the audio-type router**. MIT-licensed weights (`BEATs_iter3_plus_AS2M.pt`). [repo](https://github.com/microsoft/unilm/blob/master/beats/README.md) |
| ★ **PANNs (CNN14)** | 0.439 | **Preinstalled on the eval server** (`panns-inference==0.1.1`, `torchlibrosa==0.1.0`) → zero packaging risk. Wavegram-Logmel-CNN variant uses both waveform and log-mel. [paper](https://arxiv.org/pdf/1912.10211) · [code](https://github.com/qiuqiangkong/audioset_tagging_cnn) |

☆ SSL pretraining on unlabelled AudioSet pushed mAP 0.485 → 0.502 (BEATs → EAT → SSLAM).

## Recommendation

1. **Start with PANNs** — preinstalled, no offline-packaging risk, and 0.439 mAP is plenty for a
   0.10-weight ROC-AUC task. Ship a working presence head on day one.
2. **Upgrade to BEATs or SSLAM** only if LB probing ([03-evaluation](../competition/03-evaluation.md))
   shows the presence heads are actually costing us points.
3. The same encoder likely doubles as the **audio-type router / shared trunk** for the fake
   heads — that's where BEATs/SSLAM earns its packaging cost, not on the presence score itself.

## Edge cases that will decide the presence heads

| Case | Correct labels (per competition rules) |
|---|---|
| A cappella / vocals only | voice ✅, music ❌ |
| Instrumental / backing track only | voice ❌, music ✅ |
| Song (vocals + accompaniment) | voice ✅, music ✅ — **vocals count as voice** |
| Speech over background music | voice ✅, music ✅ |
| Voice then music **sequentially** | voice ✅, music ✅ |
| Speech + non-music noise (traffic, applause) | voice ✅, music ❌ — ⚠️ applause/crowd is a classic false-positive for "music" |
| Telephone speech with hold-music segment | voice ✅, music ✅ |

⚠️ Segment-level aggregation matters: a 60s file with 3s of music must still score high on
`MUSIC_PRESENT`. **Max/attention pooling, not mean.**
