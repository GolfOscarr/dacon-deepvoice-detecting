# 05 — Talkboard Q&A

Snapshot of https://dacon.io/competitions/official/236749/talkboard as of **2026-09-06**.
7 posts total (1 pinned notice + 6 questions). Synced with the `dacon-talkboard-sync` skill. Questions marked `[DACON 답변 요청]` are the ones
DACON commits to answering.

---

## 📌 [Pinned] DAKER! 대회 관련 문의 (#417136, DACON.GM, 2026-09-01)

The general DACON FAQ, plus one competition-specific answer in the comments.

### Q) 외국 국적자도 참가 가능한가요?
> **A (DACON):** 네. 해당 대회의 참가자격에서도 안내드리고 있듯이 **'대한민국 국민'만 참가 가능**하며
> 참가 자격에 부합하지 않는 경우에는 **평가 및 수상에 제한**됩니다.

### FAQ — 팀 구성
- 팀 구성 기간 이후에는 팀을 구성할 수 없습니다.
- 팀원 초대는 대회 페이지 팀 탭에서 닉네임/이메일/팀 이름으로 진행, 상대가 수락하면 구성 완료.
- **팀 구성을 하려면 제출 탭에서 최소 1회 제출이 필요합니다** ("최초 1회 제출").
- 아이디어 공유 방지를 위해 **팀 탈퇴 후 재구성은 불가능**합니다.

### FAQ — 제출 파일
- 베이스라인 라이브러리만 써야 하는 것은 아님. 본인 코드를 작성하고 `requirements.txt`에 반영.
- **제출 결과물 개별 삭제 불가.**
- **대회 종료 후 제출 불가.**
- 최고 점수는 리더보드에, 개별 제출 점수는 제출 탭에서 확인.

### FAQ — 리더보드
- 규정 위반 정황 발견 시 치팅 처리되어 점수 숨김. 복구하려면 소명 자료 제출·검증 필요.
- **Public Score는 실시간으로 최고 점수로 업데이트.**
- **동점 시 해당 점수를 먼저 기록한 팀이 상위.**
- 점수 이상이 의심되면 데이콘 계정 메일로 해당 제출물과 설명을 전달.

### FAQ — 2차 평가
- 상세 제출 항목은 규칙 탭의 '2차 평가 자료 제출 규칙' 참고.

### ⏳ Pending comment (2026-09-06) — identical scores from different weights
> **[DACON 답변 요청]** 안녕하세요 제출물 관련해서 문의드립니다. 현재 제가 제출한 2가지 제출물이
> 완전히 동일한 점수를 채점받았습니다. 제출번호는 82980와 83555입니다. 두 제출물을 경우 모델
> 가중치 자체가 달라서 완전히 동일한 점수를 받기는 어렵다고 판단했습니다. 혹시 이상이 있는건지
> 확인해주시면 감사하겠습니다.

*Participant comment, **no official answer yet**.* Two submissions with different model weights
scored **identically**.

⚠️ **Operational warning for us**, regardless of what DACON answers. The most likely benign
explanation is that the submitted code silently failed to use the model — wrong weight path, a
`from_pretrained` falling back to random init, or an exception caught by a defensive handler — so
both runs emitted the same degenerate predictions. Our own `script.py` design has exactly this
hazard: the per-file try/except fallback recommended in
[02-submission.md](02-submission.md) would happily write a full `submission.csv` of 0.5s from a
completely broken model, and it would score **0.5000** without erroring. See the startup-assertion
guard added there.

---

## ⏳ #417333 — [DACON 답변 요청] Real/Fake 라벨 정의와 학습데이터 사용·제출 기준
*(느아, 2026-09-05 16:28) — **0 replies, awaiting official answer***

🔴 **This post asks almost exactly the questions we had queued up.** We do not need to post them
ourselves; we need to watch this thread. Seven questions:

| # | Question | Why it matters to us |
|---|---|---|
| 1 | Do **neural-codec round-trips (EnCodec, DAC)**, **neural noise suppression / speech enhancement**, and **source separation** applied to real audio stay **REAL**? i.e. if a network reconstructs the waveform but generates no new component, is it REAL? Also: can the test set's **telephone-channel audio include neural-codec transforms**? | 🔴 Defines Pool B and the REAL-processed slice. Our [T3 resynthesis twins](../data/05-synthesis-plan.md) assume codec resynthesis = **FAKE**; our [REAL-processed slice](../data/05-synthesis-plan.md) assumes denoising = **REAL**. If answer 1 says codec round-trip is REAL, **S-S1 collapses** |
| 2 | Is an AI song (vocals + backing both generated) labelled `FILE/VOICE/MUSIC_FAKE=1`, `VOICE/MUSIC_PRESENT=1`? Is AI a cappella `VOICE_PRESENT=1, MUSIC_PRESENT=0`? | Confirms our cell 8 and voice-only definitions ([02](../data/02-label-taxonomy.md)) |
| 3 | Is `*_PRESENT=1` when the component occupies only **part** of the file? Any **minimum duration or ratio**? Can files exist with **both PRESENT = 0** (environmental sound / silence only)? If so, how is `FILE_FAKE` decided — is **AI-generated environmental sound** FAKE? | 🔴 Cell 9 currently exists in our taxonomy only as a robustness class. If AI environmental sound counts as FAKE, that is a **new detection target** we have not planned for |
| 4 | Are Voice/Music EER computed over **ground-truth** `*_PRESENT=1` files, not our predictions? | Confirms the masked-EER reading in [03](03-evaluation.md) |
| 5 | 🔴 Can **CC BY-NC-ND** data be used as-is, and may augmentation be applied? If derivative redistribution is barred, does submitting **original + augmentation code + config + seed** satisfy the requirement? | **Directly decides CtrSVDD** (307 h singing @16 kHz) — our open decision [V2](../survey/10-open-questions.md) |
| 6 | 🔴 When public data is **mixed / resampled / format-converted** before training, which stage must be submitted — the generated files, or **originals + code + seed**? | 🔴 Our entire on-the-fly composition design ([06](../data/06-augmentation-spec.md)) rests on the answer being "originals + code + seed". #417280 said augmentation intermediates need not be shipped, but **did not cover mixing two public sources into a new training sample** |
| 7 | Are **YouTube CC-licensed** datasets (e.g. YODAS) usable? | The gray area we identified and chose to avoid ([data/01](../data/01-rules-check.md)) |

**Action**: monitor. Q1, Q5 and Q6 each change work already planned. Q3 could add a detection
target. Nothing here needs a duplicate post from us — the only question still unique to us is the
**AI-Hub / NIA** one ([data/11 §1](../data/11-source-inventory.md)).

---

## #417310 — [DACON 답변 요청] 수상작 독점권리 조항의 외부 데이터·공개 모델 적용 범위
*(과제를열심히, 2026-09-03)*

**Q.** Does the "exclusive rights to winning outputs" clause swallow (1) external public data
and AI-generated training audio submitted for verification, and (2) OSS code and public
pretrained weights embedded in the final model? Can we still use external resources whose
exclusive copyright we have no authority to transfer?

> **A (DACON):** 문의주신 '수상 산출물에 대한 독점권리' 조항은 대회를 통해 참가자가 제출한
> **수상 산출물을 대상으로** 적용됩니다.
> 외부 공개 데이터, 오픈소스 코드, 공개 사전학습 모델·가중치 등 기존에 별도의 권리관계 및 라이선스가
> 존재하는 **외부 자원은 해당 라이선스 및 이용조건을 따르며**, 운영진 검증을 위해 제출한다는 사유만으로
> **기존의 권리관계가 변경되는 것은 아닙니다.**
> 따라서 외부 자원을 활용하는 경우에는 해당 자원의 **라이선스 및 이용조건상 대회 활용과 검증 목적의
> 제출이 가능한지 참가자가 직접 확인**하여 준수해 주시면 됩니다.

**Takeaway:** exclusivity binds only our own deliverable. Third-party assets keep their
licenses. Our obligation: verify each asset's license permits *both* competition use *and*
submission-for-verification.

---

## #417280 — [DACON 답변 요청] 2차 평가 시 외부 학습 데이터 파일 제출 범위 문의
*(과제를열심히, 2026-09-01)*

**Q.** Four-part question about "학습에 활용한 데이터 파일 일체":
1. If a public dataset permits non-commercial training but **restricts redistribution /
   third-party provision**, is it unusable in this competition?
2. For such datasets, may we submit official URL + download script + file list + per-file
   checksums + license evidence instead of the files?
3. For a fine-tuned public pretrained model, is only *our* fine-tuning data required (not the
   upstream pretraining corpus)?
4. For on-the-fly augmentation (codec, noise, reverberation, resampling) with fixed configs and
   seeds, must the augmented intermediates be physically stored and submitted?

> **A (DACON):**
> 1. **운영진 검증 목적의 제출은 일반적인 공개 혹은 재배포와는 구분됩니다.** 다만, 라이선스상
>    **제3자 제공 자체가 제한되는 데이터라면 대회 학습 데이터로 사용할 수 없습니다.**
> 2. 학습에 활용한 데이터는 원칙적으로 **실제 사용한 데이터 파일 일체를 제출**해야 하며,
>    **URL/스크립트/checksum 등으로 대체할 수 없습니다.**
> 3. 공개 사전학습 모델을 fine-tuning한 경우에는 **참가자가 직접 fine-tuning에 사용한 데이터를
>    제출**하면 됩니다. 사전학습 모델 개발에 사용된 **upstream pretraining corpus까지 제출할 필요는
>    없습니다.**
> 4. **augmentation으로 생성되는 중간 파일은 별도로 저장하여 제출할 필요는 없습니다.** 증강 전 원본
>    데이터와 동일한 증강 결과를 재현할 수 있는 **코드, 설정값, random seed** 등을 함께 제출해
>    주시면 됩니다.

**Takeaways — these four answers should shape the data pipeline design:**
- **License gate:** any dataset that forbids third-party provision is *out*, full stop. Vet
  before training, not after.
- Physical files for all training data — plan storage and a Drive-scale delivery path.
- Fine-tuning on public checkpoints keeps the submission burden bounded (a strong argument for
  the pretrained-frontend approach, e.g. wav2vec2/WavLM/AudioMAE + head).
- **On-the-fly augmentation with fixed seeds is the cheapest compliant design** — it keeps the
  shippable corpus small (originals only) while still being reproducible.

---

## #417198 — [DACON 답변 요청] 2차 평가 자료의 "학습에 활용한 데이터 파일 일체" — 참가자가 직접 생성한 학습 데이터의 제출 형식
*(채대언, 2026-08-27)*

**Q.** Since the organizers provide no training data, most participants' training sets will be
largely **self-generated audio**, potentially **hundreds of GB**. We already log the generating
model / version / revision / seed / prompt / post-processing parameters per file, enough to
regenerate identical audio. So: (a) must we submit the generated audio files themselves (and if
so, what is the size cap and is there a large-file channel besides email), or (b) does a
reproducible manifest (generation script + public model source/revision + seed/prompt manifest
+ regeneration procedure) satisfy the requirement?

> **A (DACON):** 문의주신 내용과 관련하여, **직접 생성한 오디오를 포함해 실제 모델 학습에 활용한
> 데이터 파일은 모두 제출**해 주셔야 하며, 해당 데이터의 생성 및 재현이 가능하도록 **생성 스크립트,
> 사용 모델 정보, 버전·리비전, 시드, 프롬프트, 후처리 파라미터 등 관련 정보도 함께 제출**해 주셔야
> 합니다.
> 따라서 **재현 정보만으로 실제 학습 데이터 파일 제출을 대체하는 것은 어렵습니다.**
> 학습 데이터의 용량이 큰 경우에는 **Google Drive 등 대용량 파일을 전달할 수 있는 방법을 활용**하여
> 제출하시면 되며, 최종적으로 모델 학습에 실제 활용한 데이터와 그 재현에 필요한 관련 자료를 제출해야
> 합니다.

**Takeaway:** files **and** provenance metadata, both. No size cap stated; large deliveries go
through Drive-style transfer. Budget disk and organize the corpus for shipping from day one.

---

## #417212 — [DACON 답변 요청] 데이터셋 사용가능여부 문의
*(먹고자진화잠만보, 2026-08-27)*

**Q.** CC-BY-NC-SA처럼 **SA(동일조건변경허락)** 조항이 붙은 데이터셋도 사용 가능할까요?

> **A (DACON):** **네 사용 가능합니다.** 누구나 접근 가능한 공개 자원이고 최소 비영리 목적의 사용이
> 허용되어 있으며 **해당 라이선스 조건을 준수하는 경우** 활용할 수 있습니다.

**Note:** "준수하는 경우" is doing real work here. ShareAlike + the competition's exclusive-rights
clause is a genuine tension if an SA-licensed dataset's terms propagate to derivatives. DACON's
answer treats license compliance as the participant's responsibility. Worth being conservative
about SA-licensed data in anything that ends up *inside the model deliverable*.

---

## #417193 — 사용가능한 데이터셋 문의
*(filot, 2026-08-26)*

**Q.** **CC-BY-NC 계열** 사용이 가능한지 문의를 드립니다.

**A.** The post shows 1 reply; the reply body is not server-rendered so it could not be captured
here. The equivalent question in #417212 (which covers the stricter CC-BY-NC-**SA** case) was
answered **yes**, so CC-BY-NC is usable under the same conditions. Re-check this thread in the
browser if a definitive quote is needed.

---

## Cross-cutting summary — what the Q&A actually settles

| Question | Settled answer |
|---|---|
| Non-Korean nationals? | Cannot be evaluated or win. |
| CC-BY-NC / CC-BY-NC-SA data? | Usable, if license terms are complied with. |
| Redistribution-restricted data? | **Unusable** — cannot even be used for training. |
| Substitute a manifest for data files? | **No.** Files must be submitted. |
| Upstream pretraining corpus of a public checkpoint? | **Not required.** |
| Augmentation intermediates? | **Not required** — originals + code + config + seed suffice. |
| Huge data volume? | Deliver via Google Drive or similar. |
| Does winning transfer rights in external OSS/data/weights? | **No** — only our own deliverable. |

### Open questions — status

⏳ **Items 1–3 below are now pending with DACON via #417333** (asked by another participant on
2026-09-05, not yet answered). Watch that thread rather than duplicating it.

1. Ground-truth labeling of **vocoder-resynthesized real speech** and **voice conversion** —
   the rules say "AI로 생성된" is FAKE and pure post-processing is REAL, but VC/resynthesis sits
   between. Same for AI-**separated** or AI-**upsampled** real audio.
2. Whether the 1,200 test files are **balanced** across the three audio types (voice / music /
   mixed) and across real/fake, since Voice EER and Music EER are computed on different subsets.
3. Whether `FILE` ground truth on a file with, say, real voice + fake music is FAKE (the rule
   implies yes — "하나라도 FAKE이면") — worth confirming explicitly.
4. Whether TTS/music-generation **commercial APIs** (which have terms restricting use of outputs
   for building competing/detection models) are acceptable given the redistribution requirement.
   ⬜ *still unasked*
5. 🔴 Whether **AI-Hub** datasets may be used, given **NIA (한국지능정보사회진흥원) is a 주최기관**
   and operates AI-Hub — and whether submission for 2차 평가 counts as 제3자 제공.
   ⬜ **still unasked, and unique to us** ([data/11 §1](../data/11-source-inventory.md))

Post as `[DACON 답변 요청] ...`. Re-check with the `dacon-talkboard-sync` skill.
