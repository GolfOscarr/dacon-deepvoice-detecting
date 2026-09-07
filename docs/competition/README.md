# DACON — 딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회

> Deep Voice Crime Response AI Detection Model Competition
> https://dacon.io/competitions/official/236749/overview/description

Documentation set for competition **236749**. Compiled 2026-09-04 from the official
overview / data / evaluation / rules tabs and the talkboard (Q&A).

| Doc | Contents |
|---|---|
| [01-overview.md](01-overview.md) | Objective, background, task definition, labels, data, schedule, prizes, organizers |
| [02-submission.md](02-submission.md) | What exactly we submit: `submit.zip` layout, eval server spec, preinstalled packages, limits, error types |
| [03-evaluation.md](03-evaluation.md) | Scoring formula (Score / ADS / CPS), EER code, Public vs Private, 1st & 2nd stage judging rubric |
| [04-rules.md](04-rules.md) | Eligibility, allowed resources & licenses, prohibited actions, 2nd-stage material submission, IP terms |
| [05-talkboard-qa.md](05-talkboard-qa.md) | All talkboard posts + official DACON answers, with practical implications |

## 30-second summary

- **Task**: given an audio file, output 5 probabilities — is the file fake, is the *voice*
  component fake, is the *music* component fake, is voice present, is music present.
- **No training data is provided.** We must build our own training set. This is the single
  biggest strategic constraint.
- **Code submission**: upload `submit.zip` (model weights + `script.py` + `requirements.txt`).
  It runs **offline** on an L4 GPU, ≤60 min for 1,200 files, ≤10 min pip install.
- **Metric**: `Score = 0.9 × ADS + 0.1 × CPS`, where ADS is built from **EER** (lower better)
  on file / voice / music, and CPS from **ROC-AUC** on voice-presence / music-presence.
- **Private = Public** (100% of test data). No hidden split, no shakeup — but also no
  protection against overfitting the leaderboard.
- **Two stages**: leaderboard top-15 advance → written report review (100 pts, of which
  leaderboard is only 30) → top 7 win.
- **Deadline for leaderboard: 2026-09-29 10:00 KST.** 3 submissions/day.
- **Eligibility: Korean nationals only.** Solo participation is fine.
