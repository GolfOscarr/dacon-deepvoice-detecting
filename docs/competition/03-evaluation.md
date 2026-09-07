# 03 — How We Are Judged

## 1. Leaderboard formula

```
Score (총점, ↑ higher is better) = 0.9 × ADS + 0.1 × CPS
```

### ADS — AI-Generated Audio Detection Score

```
ADS = 0.5 × (1 − File  EER)   # 파일 단위 탐지
    + 0.2 × (1 − Voice EER)   # 음성 성분 단위 탐지
    + 0.3 × (1 − Music EER)   # 음악 성분 단위 탐지
```

### CPS — Component Presence Score

```
CPS = 0.5 × (Voice Presence ROC-AUC)   # 음성 성분 검출
    + 0.5 × (Music Presence ROC-AUC)   # 음악 성분 검출
```

### EER definition (official code)

```python
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
fnr = 1 - tpr
idx = np.argmin(np.abs(fpr - fnr))
EER = (fpr[idx] + fnr[idx]) / 2
```

### Official notes

- ※ **FAKE를 양성 클래스(1)로 정의합니다.** (FAKE is the positive class.)
- ※ **Voice EER은 음성이 존재하는 샘플에서만, Music EER은 음악이 존재하는 샘플에서만 계산됩니다.**
  (Voice EER is computed only over samples where voice is present; Music EER only where music
  is present.)
- ※ 최종 평가는 총점(Score)을 기준으로 평가됩니다.

## 2. What the formula implies (strategy-relevant)

**Effective weights on the total score:**

| Component | Metric | Weight in Score |
|---|---|---|
| File fake | 1 − EER | 0.9 × 0.5 = **0.45** |
| Music fake | 1 − EER | 0.9 × 0.3 = **0.27** |
| Voice fake | 1 − EER | 0.9 × 0.2 = **0.18** |
| Voice presence | ROC-AUC | 0.1 × 0.5 = **0.05** |
| Music presence | ROC-AUC | 0.1 × 0.5 = **0.05** |

Three consequences:

1. **Music-fake detection is weighted higher than voice-fake (0.27 vs 0.18)** even though the
   competition is branded as deepvoice. AI-music detection (Suno/Udio-class generators) is
   *not* a side quest — it is 60% more valuable than the voice head, and it also feeds the
   file head. This is the least-obvious, highest-leverage fact in the whole rule set.
2. **EER and ROC-AUC are both ranking metrics** — only the *ordering* of scores matters, not
   calibration. No threshold tuning is needed; monotone transforms of a score are free.
   (Still, output within [0,1] as required.)
3. **Presence heads are worth only 0.10 total, and ROC-AUC is an easier metric than 1−EER.**
   A competent audio-tagging model (PANNs, or a VAD + music detector) should get these close
   to saturation cheaply. Do them early, correctly, and then stop investing there. But note
   they *also* gate which samples enter the Voice/Music EER pools for the organizers'
   ground truth — that's their labels, not our predictions, so our presence errors don't
   corrupt the fake EERs.

**On the masked EERs**: since Voice EER is computed only on voice-present samples, our
`VOICE_FAKE_PROB` on music-only files is ignored. Same for `MUSIC_FAKE_PROB` on
voice-only files. There is no penalty for arbitrary values there — but no benefit either, so
don't waste effort. What matters is the *ranking quality within the present-component pool*.

**On `FILE_FAKE_PROB`**: worth the most (0.45). Ground truth is `voice_fake OR music_fake`
over present components. Both a dedicated head and an analytic combination of the other
heads are viable; test both, they will differ mostly on mixed audio.

## 3. Public vs Private

- **Public Score : 전체 테스트 데이터 100%**
- **Private Score : 대회 종료 시점의 Public Score**

There is **no held-out private split**. Private == Public at close of the competition.

Implications:
- **No leaderboard shakeup.** Rank at the deadline is rank.
- **But the leaderboard is a 1,200-sample public test set that we can query 3×/day.**
  Overfitting it via repeated probing is a real risk *and* explicitly bounded — it is also
  what the 2nd-stage rubric is designed to counteract. A robust local CV built on held-out
  *generators* (not held-out files) is what actually transfers.
- Ties are broken in favor of whoever hit the score first.

## 4. First-stage judging

**1차 평가**: Private Score (= Public at close). Top **15 teams** advance, conditional on
submitting 2nd-stage materials and passing verification.

## 5. Second-stage judging (종합 평가, 100 points)

Written review by an expert judging panel (전문 심사위원단의 서면 평가).

| 항목 (Item) | 기준 (Criteria) | 점수 |
|---|---|---|
| **모델 성능** | Private 리더보드 환산 점수 | **30** |
| **학습데이터 구성** | 공개·외부 데이터의 선정 및 활용 전략 / 데이터 증강, 전처리 및 학습 데이터 구성의 적절성 | **20** |
| **모델 개발** | 문제에 적합한 모델 설계·선정 / 학습 및 성능 개선 과정의 체계성 | **25** |
| **결과 해석 및 일반화** | 탐지 결과와 판단 근거의 해석·시각화 / 다양한 생성방식과 음성환경에 대한 일반화 가능성 | **15** |
| **실무 활용성** | 실무 적용 가능성 / 모델 운영 및 지속적 고도화 방안 | **10** |

Original table image: [`assets/second-stage-rubric.png`](assets/second-stage-rubric.png)

**Leaderboard performance conversion formula:**

```
모델 성능 점수 = 30 × ( (our Public score) / (best Public score among the 15 finalists) ) ^ N
```
where `N` is an undisclosed adjustment coefficient set between 1 and 5.

### Why this matters more than the leaderboard

- **Only 30 of 100 points are the leaderboard.** 70 points are the reports.
- The ratio is over the *finalists*, not the whole field. If scores among the top 15 are
  tightly bunched (likely, since Private=Public and everyone converges), the ratio is near 1
  and even `N=5` compresses the gap. E.g. our score at 97% of the best, with N=5:
  `0.97^5 = 0.859` → 25.8 vs 30 — a 4.2-point deficit that a stronger report easily erases.
- Therefore: **making the top 15 is the hard gate; after that, the reports decide the prize.**
- The rubric rewards exactly what a leaderboard-only push neglects: documented data strategy,
  systematic ablations, interpretability/visualization of *why* the model flags something, and
  generalization to unseen generators and acoustic environments.
- **Write the reports as we go.** They are due 6 days after the leaderboard closes, in HWP,
  and must ship with the actual training data files (see `04-rules.md`).

> ※ 대회 종료 후 공개되는 Private 리더보드는 **최종 순위가 아니며** 2차 평가와 검증 이후 최종
> 수상자가 결정됩니다.
