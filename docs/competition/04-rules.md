# 04 — Rules

Source: 규칙 탭 + 참가 동의사항. Korean originals quoted where precision matters.
(Team-formation rules are included for completeness but are not relevant to solo participation.)

## 1. Participation

- 개인 또는 팀 참여 가능. **개인 참가**: 팀 신청 없이 제출탭에서 자유롭게 제출.
- 팀 최대 인원 5명. 동일인이 개인 또는 복수팀에 **중복 등록 불가**.
- **참가 자격: 대한민국 국민 누구나.** DACON's official talkboard answer confirms non-Korean
  nationals are **restricted from evaluation and awards**:
  > 해당 대회의 참가자격에서도 안내드리고 있듯이 '대한민국 국민'만 참가 가능하며 참가 자격에
  > 부합하지 않는 경우에는 평가 및 수상에 제한됩니다.
- One dacon.io account only. Multiple accounts / proxy accounts → disqualification.
- Minors must submit a guardian consent form.
- Employees/interns/contractors/officers/directors of the hosting organizations and DACON
  cannot participate.
- Time zone for all deadlines: **KST (UTC+9)**.

## 2. Competition rules (대회 규칙)

### 2.1 사용에 법적 제한이 없는 모든 방법론 가능

> 아래 규정을 준수하는 범위 내에서, **누구나 접근 가능한 공개 자원**이며 **최소 비영리 목적으로의
> 사용이 허용된 경우** 사전학습 모델, API, 외부 데이터 수집·생성 등 **모든 방법론을 활용**할 수
> 있습니다. 단, 사용하려는 데이터, 사전학습 모델, API 등의 **라이선스 및 이용조건을 참가자가 직접
> 확인하고 준수**해야 합니다.

So: **pretrained models, APIs, external data collection, and self-generated data are all
allowed**, subject to (a) publicly accessible, (b) at minimum non-commercial use permitted,
(c) license compliance is our responsibility.

Confirmed by official answer on the talkboard: **CC-BY-NC and CC-BY-NC-SA datasets are usable.**
See `05-talkboard-qa.md`.

⚠️ **Critical license caveat** (also from the talkboard): because 2nd-stage submission requires
handing over the *actual training data files*, a dataset whose license **forbids third-party
provision / redistribution cannot be used at all** in this competition — even if training on it
would be permitted. **Vet licenses for redistribution-to-organizer up front, before training.**

### 2.2 출처 명시 의무

> 2차 평가 자료 제출 시 본 규정에 따라 사용된 **모든 요소에 대한 각각의 출처를 명확하게 기술**할 수
> 있어야 합니다.

→ Maintain a provenance ledger from day one: every dataset, pretrained checkpoint, API, and
generated-audio batch with source, version/revision, license, and how it was used.

### 2.3 비공개 평가 데이터를 활용한 추가 학습 금지

> 제출하는 추론 코드에서는 예측을 위한 **전처리·후처리 및 추론 과정은 자유롭게 구성**할 수 있습니다.
> 다만, **비공개 평가 데이터셋을 활용한 추가 학습, 모델 튜닝, Pseudo-Labeling 등 모델을 갱신하거나
> 학습에 활용하는 행위는 허용되지 않습니다.**

→ No test-time training, no transductive learning, no pseudo-labeling on the eval set.

### 2.4 파일 단위 독립 예측 원칙

> 비공개 평가 데이터의 각 파일 샘플은 **서로 독립적으로 예측**해야 합니다.
> - **허용**: 하나의 파일 샘플을 **세그먼트 단위로 분할하여 추론하고, 결과를 종합**하는 방식
> - **금지**: **다른 파일 샘플의 정보·예측값·통계 등을 활용**하여 예측값을 생성하거나 보정하는 방식

→ Segment-and-pool within a file: **allowed**. Cross-file normalization, batch statistics over
the test set, score standardization across the 1,200 files, clustering, rank calibration:
**forbidden**. Note this also rules out common EER-boosting tricks like score normalization
over the eval cohort.

⚠️ Watch out for accidental violations: `BatchNorm` in train mode at inference, any global
statistic computed over the test set (e.g. per-dataset mean/var normalization fitted on test),
or adaptive thresholding across files. Keep the model in `eval()` and make inference a pure
function of a single file.

## 3. 2차 평가 자료 제출 규칙

Finalists email code + two reports to **dacon@dacon.io** within the deadline
(2026-09-30 12:00 – 2026-10-05 10:00).

**Report form**: HWP template provided by DACON.
- 구성 항목 변경 및 추가 가능
- 보고서 분량 제한 없음
- 단, 2차 평가 항목의 내용이 모두 반영될 수 있도록 반드시 구성

**제출 파일 목록:**

1. 🔹 **Private Score 재현이 가능한 학습 코드** (training code that reproduces the score)
2. 🔹 **모델 개발 보고서 (HWP)** — must be HWP format
3. 🔹 **학습데이터 구성 보고서 (HWP)** — must be HWP format
   - **학습에 활용한 데이터 파일 일체 포함** (all data files actually used for training)
4. 🔹 **팀 구성원 정보**: 성명 / 생년월일 / 성별 / 현재 소속

The scope of item 3 is the single most-clarified point on the talkboard — see
`05-talkboard-qa.md`. Summary of official answers:

- Actual data files must be submitted; **URL / download script / checksum manifests are NOT an
  acceptable substitute**.
- This includes **audio we generated ourselves**, plus generation scripts, model names,
  versions/revisions, seeds, prompts, and post-processing parameters for reproducibility.
- For a fine-tuned public pretrained model, only **our** fine-tuning data is required — not the
  upstream pretraining corpus.
- **On-the-fly augmentation intermediates need not be stored**; submit the pre-augmentation
  originals + augmentation code + config + random seed.
- Large volumes may be delivered via Google Drive or similar.

**Practical consequence for planning**: our data pipeline must produce a *persisted, shippable*
training corpus with a per-file provenance manifest. Anything generated non-deterministically
without being saved is a liability. Prefer: persist source/generated audio, do augmentation
on-the-fly with fixed seeds.

## 4. 유의 사항

- **1일 최대 제출 횟수: 3회**
- 사용 가능 언어: **Python**
- 모든 CSV 데이터와 제출 파일은 **UTF-8 인코딩**
- 대회 종료 후 공개되는 **Private 리더보드는 최종 순위가 아님** — 2차 평가·검증 후 최종 수상자 결정
- **코드 제출 기능을 악용한 평가 데이터셋 유출 시도 등이 발견되는 경우 즉시 실격**
  (→ do not attempt to exfiltrate, print, hash-dump, or otherwise leak test data through logs
  or the output CSV; any such probing is an instant DQ)
- 데이콘 대회 **부정 제출 이력**이 있는 경우 평가가 제한됨
- 규정 위반 정황 발견 시 치팅 처리되어 리더보드 점수가 숨김 처리되며, 복구하려면 소명 자료 제출 필요
- 제출물에는 **검증/테스트 데이터셋에 대한 자료가 포함되어서는 안 되며**, 사람이 예측한 정보의 사용·
  통합도 금지

## 5. 토론(질문) 규칙

> 대회 운영 및 데이터 이상에 관련된 질문 외에는 답변을 드리지 않고 있습니다. 데이콘 답변을 희망하는
> 경우 토크 게시글 댓글로 질문을 올려 주시기 바랍니다. 예) **[DACON 답변 요청]** 시상식은 언제 열리나요?

→ To get an official answer, prefix the talkboard post title with `[DACON 답변 요청]`.

## 6. 참가 동의사항 — IP terms

### 아이디어에 대한 권리 및 사용
Rights to submitted ideas remain with the participant. The host/operator may use submitted
ideas and materials only as needed for review/selection, plus incidental uses customary to
running the competition (promotion, exhibition of selected ideas, printing, PR).

### 수상 산출물에 대한 독점권리 (exclusive rights to winning outputs)

> 주최 기관이 수상 산출물에 대한 상장 혹은 상금을 참여자에게 지급하는 경우, 대회 **수상작에 대한
> 저작권(2차적 저작물작성권 포함)은 주최 측에 독점적으로 귀속**되며, 참가자는 주최 측의 사전 서면
> 동의 없이 해당 저작물을 사용하거나 제3자에게 제공할 수 없습니다.

**If we win, copyright in the winning work (including derivative-work rights) transfers
exclusively to the host, and we cannot use or share it without their prior written consent.**

Official clarification (talkboard 417310) on how this interacts with external resources:

> 문의주신 '수상 산출물에 대한 독점권리' 조항은 대회를 통해 참가자가 제출한 **수상 산출물을 대상으로**
> 적용됩니다. 외부 공개 데이터, 오픈소스 코드, 공개 사전학습 모델·가중치 등 **기존에 별도의 권리관계
> 및 라이선스가 존재하는 외부 자원은 해당 라이선스 및 이용조건을 따르며**, 운영진 검증을 위해
> 제출한다는 사유만으로 기존의 권리관계가 변경되는 것은 아닙니다.

→ The exclusivity clause applies to *our own* deliverable. Third-party open data, OSS code, and
public pretrained weights keep their own licenses; submitting them for verification does not
transfer their rights. It remains our duty to confirm the license permits both competition use
and submission-for-verification.

### Other general terms
- Binding contract on entry; false information about identity/residence/contact/rights
  ownership → immediate disqualification.
- Entries that are unreadable, incomplete, corrupted, forged/altered, late, or non-conforming
  are void and may be disqualified.
- DACON acts as operator/independent contractor, not a party to the prize award.
