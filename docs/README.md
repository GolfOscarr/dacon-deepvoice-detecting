# Documentation Index

Working documentation for **DACON 236749 — 딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회**.

| Set | Purpose | Start at |
|---|---|---|
| **[competition/](competition/)** | What the competition is, what we submit, how we're scored, the rules, and every official Q&A | [README](competition/README.md) |
| **[survey/](survey/)** | Academic prior art: SOTA per head, model & dataset catalogs, augmentation, challenge playbooks | [README](survey/README.md) |
| **[kaggle/](kaggle/)** | Practical competition intelligence: winners' EDA, augmentation, synthesis, validation and inference tricks | [README](kaggle/README.md) |
| **[data/](data/)** | Our data strategy: rules check, label taxonomy, sources, synthesis, augmentation spec, build plan | [README](data/README.md) |
| **[validation/](validation/)** | How we measure: split scheme, metric harness, decision protocol, gates, LB decomposition | [README](validation/README.md) |

## Reading order for someone new

1. `competition/README.md` — 30-second summary of the task and constraints
2. `competition/03-evaluation.md` — the scoring formula; everything else follows from it
3. `survey/README.md` — the 10 findings that drive design
4. `kaggle/05-transferable-playbook.md` — the practical checklist
5. `data/README.md` — what we're actually building
6. `validation/README.md` — how we decide whether any of it worked

## Confidence marks used throughout

★ verified from a primary source · ☆ secondary source, re-verify before relying on it ·
⚠️ risk or caveat · 🔴 decision-changing · ❌ forbidden by competition rules

## Status

| | |
|---|---|
| Competition docs | ✅ complete (as of 2026-09-04) |
| Prior-art survey | ✅ complete; open items in [survey/10](survey/10-open-questions.md) |
| Kaggle intelligence | ✅ complete (2026-09-05) |
| Data strategy | ✅ planned; **not yet executed** — see [data/08](data/08-build-plan.md) |
| Validation design | ✅ drafted (2026-09-07); gates not yet implemented |
| Modeling | ⬜ not started |

**Next action**: Phase 0 of [data/08-build-plan.md](data/08-build-plan.md) — license audit and
dummy-file forensics, both blocking.
