# 01 — Rules Check: Can We Crawl?

**Question asked**: is crawling YouTube (or the open internet) for AI-generated audio permitted
as a source of training data for this competition?

## 🔴 Verdict: No. It is blocked by the competition rules, independently of copyright.

This is not a copyright judgment — it is a mechanical consequence of two competition rules plus
one official DACON answer.

### The chain of reasoning

1. **2차 평가 자료 제출 규칙** requires submitting *"학습에 활용한 데이터 파일 일체"* — the actual
   audio files used for training.
2. **DACON answer, talkboard #417198**: reproducibility information alone cannot substitute.
   > 재현 정보만으로 실제 학습 데이터 파일 제출을 대체하는 것은 어렵습니다.
   And #417280:
   > 학습에 활용한 데이터는 원칙적으로 **실제 사용한 데이터 파일 일체를 제출**해야 하며,
   > **URL/스크립트/checksum 등으로 대체할 수 없습니다.**
3. **DACON answer, talkboard #417280** — the decisive sentence:
   > 라이선스상 **제3자 제공 자체가 제한되는 데이터라면 대회 학습 데이터로 사용할 수 없습니다.**

YouTube audio carries no license permitting us to provide the files to a third party. Therefore
we cannot ship it. Therefore, by DACON's own words, **we cannot train on it.**

There is no workaround inside the rules: the manifest route is explicitly closed, and the
"we'll just not mention it" route fails at the 2nd-stage verification stage, which is a written
review by 국립과학수사연구원 — a forensic institute — with 20 of 100 points allocated to
데이터 구성 and an explicit **출처 명시 의무** for every element used.

### Cost of getting this wrong

Rule violation → 치팅 처리, leaderboard score hidden, and 부정 제출 이력 restricts future DACON
evaluation. This would surface at 2차 평가, i.e. *after* we've done all the work.

### Two gray areas, both of which I'd still avoid

| Case | Assessment |
|---|---|
| **YouTube's CC-BY filtered videos** | The uploader grants CC-BY, which *does* permit redistribution. But downloading still violates YouTube's own ToS (a contract question, separate from the content license). The same CC-BY audio is available from sources that host it directly — Free Music Archive, Jamendo, archive.org — so there is no reason to take the risk. |
| **Crawled data for evaluation only, never training** | Arguably not "학습에 활용한 데이터". But if it drives model selection it shapes training in substance, and we'd be arguing a technicality in front of a forensics panel. Also: scraped clips have **no reliable ground-truth labels** — we'd be guessing which are AI-generated. |

## What we do instead

The alternative is not a downgrade. For this specific task it is **technically better**, for four
concrete reasons:

| | Crawling | Self-generation + licensed corpora |
|---|---|---|
| Labels | Guessed from titles/descriptions. Unverifiable. | **Exact by construction**, including component-level voice/music fake labels |
| Decoupled cells (real voice + fake music) | Essentially unobtainable | **Trivially constructible** from stems |
| Generator-disjoint validation splits | Impossible — we don't know the generator | **Designed in**: we assign generators to splits before generating |
| Confound control | None. Genre, language, loudness, codec all correlate with source | **Controllable**: matched prompts, identical signal chain |

We cannot label a scraped clip's `VOICE_FAKE` vs `MUSIC_FAKE` separately at all — and that is
half of our metric. Crawling cannot produce our label structure even in principle.

### The one real capability we lose

Coverage of **commercial generators** (Suno, Udio, ElevenLabs) that plausibly appear in the test
set. Three mitigations:

1. Open models occupy the same design space (ACE-Step, YuE for song generation; the open TTS
   family for speech). Breadth of families beats fidelity to one vendor.
2. 🔴 **16 kHz standardization strips the production fingerprints that distinguish Suno/Udio
   anyway** — fixed 192/320 kbps, 48 kHz output, upsampling traces ([survey 02](../survey/02-sota-music.md)).
   What remains is generator-family structure, which open models share.
3. Commercial API ToS typically forbid using outputs to build detection or competing models, so
   this route was likely closed regardless — see [V6](../survey/10-open-questions.md).

## Allowed source classes (the actual working set)

| Class | Permitted | Notes |
|---|---|---|
| **Self-generated** with open-weight models | ✅ | We own the files. The backbone of the plan. |
| **CC-BY / CC0 / public-domain** corpora | ✅ | Redistributable |
| **CC-BY-NC / CC-BY-NC-SA** corpora | ✅ | Explicitly confirmed by DACON (#417212) |
| **CC-\*-ND** corpora | ⚠️ | NoDerivatives may bar augmented copies — legal review ([V2](../survey/10-open-questions.md)) |
| Research datasets with permissive terms | ✅ | Verify each ([06-datasets](../survey/06-datasets.md)) |
| **Scraped web/YouTube audio** | ❌ | This document |
| **Commercial generative API outputs** | ⚠️→❌ | ToS + redistribution; assume no until verified |

## Verification duty

Every item entering the corpus gets a row in the provenance ledger with an explicit
**redistribution verdict** before a single byte is downloaded. See
[08-build-plan.md](08-build-plan.md#provenance-ledger).
