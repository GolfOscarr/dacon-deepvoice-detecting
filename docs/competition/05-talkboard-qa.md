# 05 — Talkboard Q&A

Snapshot of https://dacon.io/competitions/official/236749/talkboard as of **2026-09-08**.
9 posts total (1 pinned notice + 8 questions). Synced with the `dacon-talkboard-sync` skill. Questions marked `[DACON 답변 요청]` are the ones
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

## ✅ #417333 — [DACON 답변 요청] Real/Fake 라벨 정의와 학습데이터 사용·제출 기준
*(느아, 2026-09-05 16:28) — **ANSWERED 2026-09-08, 7 of 7***

🔴 **The single most consequential thread so far.** Another participant asked almost exactly our
queued questions, and DACON answered all seven in one reply. Two answers change work already
planned; two settle open design bets in our favour.

> **A (DACON):**
> 1. 음성·음악 성분을 **새로 생성하지 않는 후처리만 적용된 경우 REAL로 간주**합니다. 대회 개요 ->
>    설명을 확인하세요.
> 2. 네 맞습니다.
> 3. **일부 구간에만 존재하더라도 해당 성분이 있으면 PRESENT=1**입니다.
> 4. Voice/Music EER은 **Ground Truth의 PRESENT 값 기준**으로 평가 대상을 선정합니다. 평가 ->
>    리더보드 산식 설명을 확인하세요.
> 5. **말씀하신 방식도 가능하며**, 이경우에는 **원본 파일과 코드로 재현될 수 있어야합니다.**
> 6. 가공 데이터는 **원본 데이터와 재현 가능한 코드·설정값·seed 등을 제출**하면 됩니다.
> 7. **참가자가 해당 데이터셋 및 원본 콘텐츠의 라이선스·이용조건을 직접 확인하여 판단해야 하며**,
>    운영진에서는 개별 데이터셋의 라이선스 적합 여부를 **별도로 판단하거나 보증하지 않습니다.**

### 🔴 A1 — codec round-trip, enhancement and separation are all REAL. **S-S1 collapses.**

This is the answer we flagged as able to invalidate planned work, and it did. Anything that
reconstructs a waveform **without generating a new voice or music component is REAL** — neural
codec encode/decode (EnCodec, DAC), neural denoise / speech enhancement, and source separation
included.

**Takeaway (against us):** our cheapest planned Pool B multiplier is gone. [`04`](../data/04-sources.md)
lists *"Self vocoder/codec resynthesis of Pool A — ⭐ Cheapest family multiplier; ⭐ gives perfectly
matched real/fake pairs"*, and [`05`](../data/05-synthesis-plan.md)'s **T3 resynthesis twins**
label exactly this as FAKE. Under A1 those files are **REAL**, so training on them as FAKE teaches
the model the inverse of the target.

**Takeaway (for us):** the same answer promotes the **REAL-processed slice** from optional to
required. The evaluation set's REAL class contains codec-round-tripped, denoised and separated
audio, so a detector that has never seen processed REAL will false-positive on it. This is
ArtifactNet's codec-aware training idea applied to the REAL side.

### ✅ A2 — the label taxonomy is confirmed

AI song (vocal + backing generated) is `FILE/VOICE/MUSIC_FAKE=1` with both `PRESENT=1`; AI a
cappella is `VOICE_PRESENT=1, MUSIC_PRESENT=0`. [`02`](../data/02-label-taxonomy.md) stands as
written.

### ✅ A3 — any duration counts, but cell 9 is still open

`PRESENT=1` if the component is there at all, however briefly. No minimum duration or ratio.

**Takeaway:** this is the strongest external support yet for the SED head and the
`0.5·clip + 0.5·frame_max` pooling — a mean pool over a 60 s file cannot represent a 2 s voice
segment, and the ground truth explicitly labels that file `VOICE_PRESENT=1`.

⚠️ **Not settled:** DACON answered the duration half of Q3 and **did not address** whether files
exist with both `PRESENT=0`, nor how `FILE_FAKE` is decided for AI-generated environmental sound.
Cell 9 remains an open question, not a closed one. Do not read A3 as covering it.

### ✅ A4 — masked EER on ground truth, as implemented

Confirms [`03`](03-evaluation.md) and the [`metrics/`](../../metrics/AGENTS.md) implementation.

### 🔴 A5 — **ND data is usable.** The blocker is lifted

CC BY-NC-ND data may be used, augmentation included, provided the work is reproducible from
**원본 파일 + 코드**.

**Takeaway:** this unblocks a large, high-value set that was held out pending exactly this answer —
**Codecfake** (inventory crown jewel #4, 32 GB, and the codec-LM family that vocoder-trained
detectors are blind to), **ST-Codecfake**, **SceneFake**, **CtrSVDD** (260 h of sung fake already
at 16 kHz), and roughly **63,000 ND-licensed tracks** across FMA and MTG-Jamendo. See
[`data/12`](../data/12-acquisition-status.md).

### 🔴 A6 — on-the-fly composition is legal. The design holds

*가공 데이터는 원본 데이터와 재현 가능한 코드·설정값·seed 등을 제출하면 됩니다.*

**Takeaway:** [`data/06`](../data/06-augmentation-spec.md) rested entirely on this answer, and
[PROGRESS](../../PROGRESS.md) recorded it as the open bet. We do **not** ship composed or
augmented output — originals plus code, config and seed suffice. Note how this sits beside
#417280: *source* files cannot be replaced by URLs or checksums, but *derived* data can be
replaced by reproducible code. The two answers are consistent and cover different stages.

This also raises the stakes on `render(spec) == render(spec)`
([`training/`](../../training/AGENTS.md)) — reproducibility stops being an internal nicety and
becomes the thing the 2nd-stage submission is built on.

### ⚠️ A7 — YouTube CC is our call, and nobody will underwrite it

DACON will **not** adjudicate or guarantee any individual dataset's licence. That is neither a yes
nor a no.

**Takeaway:** [`data/01`](../data/01-rules-check.md)'s decision to avoid scraped YouTube audio
stands on its own reasoning (ToS, and no component-level ground truth). More broadly this
validates the **G2 gate** being ours to run: the verdict recorded in
[`scripts/sources.yaml`](../../scripts/sources.yaml) is the only licence check anyone will do.

---

## ✅ #417344 — [규정 문의] 행 독립성 검증 시점 및 사후 규정 위반 판정 가능 여부
*(participant, 2026-09-07) — **ANSWERED***

**Q.** Is row independence actually verified at submission time? Does a surviving leaderboard
score imply the submission is compliant? Can a score be invalidated later?

> **A (DACON):** 행 독립성을 포함한 규칙 준수 여부는 운영진이 대회 기간 중 **불시에 점검**하고 있으며,
> 해당 점검은 **자동으로 이루어지는 방식이 아닙니다.** 참가자는 대회 기간 전체에 걸쳐 규칙을 준수해야
> 하며, 규칙 위반 정황이 확인될 경우 필요한 조치가 이루어질 수 있습니다. 또한 운영진이 규칙 준수 여부
> 확인을 위해 소명을 요청하는 경우 참가자는 이에 응해야 합니다.
> 대회 종료 후에는 **수상 후보 제출물 등을 대상으로 엄밀한 코드 및 규칙 검증**이 진행됩니다.

🔴 **Takeaway:** checks are **spot checks, not automatic**, and a currently-valid leaderboard score
is **not** evidence of compliance. Rigorous verification happens after the competition, on award
candidates — i.e. exactly when a violation is most expensive.

This is the strongest external justification yet for the rule-2.4 work already done: four separate
batch-dependence defects were found and fixed in [`models/`](../../models/AGENTS.md), each of which
would have scored normally on the leaderboard and failed the post-hoc review. "It scored fine" was
never going to be the test.

---

## ✅ #417336 — 2차 평가 학습데이터 제출 방법 문의 (대용량)
*(participant, 2026-09-07) — **ANSWERED***

**Q.** ~20 GB of training data exceeds mail attachment limits. Cloud link acceptable? Is there a
size cap? If so, can a reproducible manifest (filenames, source URLs, licences, md5) substitute?

> **A (DACON):** **상한은 없으며**, **구글 드라이브 등의 방법으로 제출**하시면 되겠습니다.

**Takeaway:** no size limit, and Google Drive delivery is confirmed — [`data/08`](../data/08-build-plan.md)
already assumed this. ⚠️ The third sub-question (manifest instead of files) was **not** answered
here, but #417280 already closed it for source data: URLs and checksums cannot substitute. A6 of
#417333 covers the derived half.

Practical consequence: our raw store is currently ~276 GB of *candidate* downloads. What ships is
only what training actually consumed, and with A6 that is the **source pools plus code**, not the
composed output.

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
| **CC-BY-NC-ND data?** | 🔴 **Usable** (#417333 A5) — augmentation included, if reproducible from 원본 파일 + 코드. |
| Redistribution-restricted data? | **Unusable** — cannot even be used for training. |
| **Codec round-trip / denoise / separation of real audio?** | 🔴 **REAL** (#417333 A1) — reconstruction is not generation. |
| **Composed / augmented / resampled training data?** | 🔴 Ship **originals + code + config + seed** (#417333 A6). Derived files need not be stored. |
| **Source** data files? | Cannot be replaced by URL/script/checksum (#417280). |
| PRESENT label for a short component? | `=1` at any duration; no minimum (#417333 A3). |
| Voice/Music EER population? | Ground-truth `PRESENT=1`, not our predictions (#417333 A4). |
| AI song / AI a cappella labels? | As we had them (#417333 A2). |
| Who checks a dataset's licence? | 🔴 **We do.** DACON neither judges nor guarantees it (#417333 A7). |
| Row-independence enforcement? | 🔴 Spot checks, **not automatic**; rigorous review after the competition (#417344). A valid score is not proof of compliance. |
| 2nd-stage data delivery? | Google Drive, **no size cap** (#417336). |

### Open questions — status

✅ **Items 1–3 of the previous list were answered by #417333 on 2026-09-08** and have moved into
the summary above. What remains:

1. ⚠️ **Cell 9 is still open.** #417333 A3 answered the *duration* half of its question and said
   nothing about files with **both `PRESENT=0`**, nor how `FILE_FAKE` is decided for **AI-generated
   environmental sound**. If AI environmental sound is FAKE, that is a detection target we have not
   planned for ([data/02](../data/02-label-taxonomy.md)). ⬜ *worth a direct ask*
2. Whether the 1,200 test files are **balanced** across audio types and across real/fake, since
   Voice EER and Music EER are computed on different subsets. ⬜ *still unasked*
3. Whether TTS/music-generation **commercial APIs** — whose terms restrict using outputs to build
   detection models — are acceptable. ⚠️ #417333 A7 makes this **our** call: DACON will not
   adjudicate. Treat as ❌ unless a specific vendor's terms clearly permit it.
4. 🔴 Whether **AI-Hub** datasets may be used, given **NIA is a 주최기관** and operates AI-Hub — and
   whether submission for 2차 평가 counts as 제3자 제공.
   ⬜ **still unasked, and still unique to us** ([data/11 §1](../data/11-source-inventory.md))

⚠️ Note on #417136: a participant's comment reporting **two different model weights scoring
identically** (submissions 82980, 83555) is still unanswered. Worth watching — if the scorer is
insensitive in some regime, that affects how much we trust small leaderboard deltas.

Post as `[DACON 답변 요청] ...`. Re-check with the `dacon-talkboard-sync` skill.
