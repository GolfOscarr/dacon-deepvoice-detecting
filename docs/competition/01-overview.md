# 01 — Competition Overview

## Identity

| | |
|---|---|
| Title | 딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회 |
| Competition ID | 236749 |
| Theme (주제) | 신종 보이스피싱 범죄 대응을 위한 AI 딥보이스 탐지 모델 개발 |
| 주최 (Host) | 행정안전부, 한국지능정보사회진흥원 (NIA) |
| 주관 (Supervision) | 국립과학수사연구원 (NFS, National Forensic Service) |
| 운영 (Operation) | 데이콘 (DACON) |
| Total prize | ₩42,000,000 |
| Eligibility | 대한민국 국민 누구나 (Korean nationals only) |

## Background (배경) — verbatim

> 생성형 인공지능과 음성합성 기술의 고도화로 실제 음성과 구분하기 어려운 딥보이스의 생성·활용이
> 확산되고 있습니다. 딥보이스가 보이스피싱, 음성 사칭 및 허위정보 생성 등에 악용됨에 따라, 다양한
> 생성기술과 실제 통신환경에서 발생하는 음성 변형에 대응할 수 있는 탐지 기술의 중요성이 높아지고
> 있습니다.
>
> 또한 생성형 AI를 활용한 음성뿐만 아니라 음악 등 다양한 형태의 오디오 생성·변조가 가능해짐에 따라,
> 딥보이스 탐지 역시 다양한 오디오 환경을 종합적으로 고려할 필요가 있습니다. 특히 하나의 오디오에
> 음성 또는 음악이 단독으로 존재하거나 함께 포함될 수 있는 환경에서, 각 유형의 존재 여부와 생성·변조
> 여부를 정확하게 판별할 수 있는 탐지 기술이 요구되고 있습니다.

**Reading between the lines** — the organizer is a forensic institute. They care about (a)
generalization to *unseen* generators, (b) robustness to *telephone channel* degradation, and
(c) music as well as speech. The 2nd-stage rubric (see `03-evaluation.md`) explicitly rewards
"다양한 생성방식과 음성환경에 대한 일반화 가능성" — generalization, not leaderboard score alone.

## The task

For **each audio file**, predict five probabilities in `[0, 1]`:

| Column | Meaning |
|---|---|
| `FILE_FAKE_PROB` | probability the file as a whole is FAKE |
| `VOICE_FAKE_PROB` | probability the **voice component** is FAKE |
| `MUSIC_FAKE_PROB` | probability the **music component** is FAKE |
| `VOICE_PRESENT_PROB` | probability voice is present in the file |
| `MUSIC_PRESENT_PROB` | probability music is present in the file |

So it is a **5-head multi-task problem**: 3 spoof-detection heads + 2 audio-tagging heads.

## Audio types (오디오 유형)

| Type | Definition |
|---|---|
| 음성 (voice) | 사람의 발화 또는 **보컬만** 포함된 오디오 |
| 음악 (music) | **보컬이 없는** 반주·악기음만 포함된 오디오 |
| 혼합 (mixed) | 음성과 음악이 **동시에 또는 순차적으로** 포함된 오디오 |

> (\*) 보컬은 **음성 성분으로 분류**합니다. 따라서 보컬과 반주가 함께 포함된 노래는 **혼합** 오디오에 해당합니다.

⚠️ This is a critical labeling convention: **a normal sung song = voice present AND music
present (mixed)**. "Music only" means instrumental/backing track with no vocals.

Also note "**순차적으로**" — voice and music may appear *sequentially* in the same file, not just
overlapped. Segment-level analysis with pooling is therefore natural (and explicitly allowed).

## REAL(0) / FAKE(1) criteria

- AI로 **생성된** 음성 또는 음악 성분 → **FAKE**
- AI로 생성되지 않은 실제 원천의 음성 또는 음악 성분 → **REAL**
- 음성과 음악 중 **하나라도 FAKE이면 파일 전체를 FAKE**로 분류
  → `FILE_FAKE = VOICE_FAKE OR MUSIC_FAKE` (logical OR over present components)
- 실제 원천 오디오에 **품질 개선, 잡음 제거, 음량 조정** 등 성분 자체를 새로 생성하지 않는
  **후처리만 적용된 경우 → REAL**

⚠️ The last rule matters a lot: denoised / enhanced / normalized real audio is still REAL.
Naive artifact detectors that fire on "processed-sounding" audio will produce false positives.
Conversely, vocoder-resynthesis *is* generation → FAKE.

Implication for the file head: `FILE_FAKE_PROB ≈ 1 − (1 − p_voice_fake·p_voice_present)(1 − p_music_fake·p_music_present)`
is a defensible analytic prior, but since `FILE` is scored by its own EER it can also be a
learned head. Worth testing both.

## Evaluation data (평가 데이터 구성)

- **1,200 audio files** total.
- Length: **4 seconds ≤ x ≤ 1 minute**.
- Sample rate: **standardized to 16 kHz**.
- Channels: **mono and stereo both present**, per sample.
- Container/codec: **MP3, WAV, FLAC 등 다양한 확장자**. The model must handle all of them.
- **일부 샘플에는 전화채널 오디오가 포함** — some samples are telephone-channel audio
  (narrowband / codec-degraded).
- Test data is **not released publicly**.

## Provided data (배포용 데이터)

> ※ 본 경진대회에서는 **별도의 학습 데이터셋을 제공하지 않으며**, 참가자는 필요한 경우
> **학습 데이터를 직접 구성하여 활용**해야 합니다.

```
open.zip
├─ baseline_submit.zip   # baseline code + model, a valid leaderboard submission (reference only)
└─ data/
   ├─ test/
   │  ├─ TEST_0000.wav
   │  ├─ TEST_0001.wav
   │  └─ TEST_0002.wav   # 3 dummy files, format-check only
   └─ sample_submission.csv   # 3 rows × 6 cols (ID + 5 predictions)
```

- The 3 dummy WAVs exist only to verify the I/O contract.
- On the eval server, `data/test/` is **replaced by the real 1,200 files at the identical path**.
- Dummy files are WAV, but real data includes MP3/WAV/FLAC — **do not assume `.wav`; glob the
  directory, don't hardcode the extension.**
- Output must have the same number of rows as test IDs and the same column structure as
  `sample_submission.csv`.

**This is effectively a "bring your own dataset" competition.** Dataset construction is a
first-class part of the work and is worth 20 of 100 points in the 2nd stage.

## Schedule (세부일정, KST / UTC+9)

| Date | Event |
|---|---|
| 2026-08-18 10:00 | 참가 신청 시작 (registration opens) |
| 2026-08-26 10:00 | 대회 시작 (competition starts) |
| 2026-09-23 23:59 | 팀 병합 마감 (team merge deadline) — N/A for solo |
| **2026-09-29 10:00** | **리더보드 제출 마감 (final leaderboard submission)** |
| 2026-09-30 10:00 | 대회 종료 |
| 2026-09-30 12:00 – 2026-10-05 10:00 | 2차 평가 자료 제출 (2nd-stage materials) |
| 2026-10-05 12:00 – 2026-10-15 10:00 | 2차 평가 및 검증 |
| 2026-10-16 10:00 | 최종 결과 발표 |
| 2026-11-27 (예정) | 오프라인 시상식 |

> ※ 세부 일정은 대회 운영상황에 따라 변동될 수 있습니다.
> (Site note: the 대회 기간 line contains an obvious typo — "2025년 09월 30일" — the schedule
> list and all other references say 2026.)

**As of 2026-09-04 there are ~25 days until the leaderboard deadline**, and materials for the
2nd stage are due only 6 days after that — so the report and the *actual training data files*
must be kept submission-ready throughout, not assembled at the end.

## Prizes (시상)

| 훈격 | 상 | 팀수 | 상금 |
|---|---|---|---|
| 대상 | 행정안전부 장관상 | 1팀 | 2,000 만원 |
| 최우수상 | 국립과학수사연구원 원장상 / 한국지능정보사회진흥원 원장상 | 2팀 | 각 500 만원 |
| 우수상 | – | 4팀 | 각 300 만원 |
| **합계** | | **7팀** | **총 4,200 만원** |

Original table image: [`assets/prizes.png`](assets/prizes.png)

## Competition format (대회 방식)

- 🔹 **1차 평가**: Private 리더보드 기준 **상위 15팀**을 2차 평가 진출팀으로 선정
- 🔹 **2차 평가**: 진출팀은 **'모델 개발 보고서'**와 **'학습데이터 구성 보고서'**를 작성·제출하고,
  이를 종합 평가하여 **최종 상위 7팀**을 수상팀으로 선정

Details in `03-evaluation.md`.
