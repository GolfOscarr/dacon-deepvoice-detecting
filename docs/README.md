# Documentation Index

Working documentation for **DACON 236749 — 딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회**.

| Set | Purpose | Start at |
|---|---|---|
| **[competition/](competition/)** | What the competition is, what we submit, how we're scored, the rules, and every official Q&A | [README](competition/README.md) |
| **[survey/](survey/)** | Academic prior art: SOTA per head, model & dataset catalogs, augmentation, challenge playbooks | [README](survey/README.md) |
| **[kaggle/](kaggle/)** | Practical competition intelligence: winners' EDA, augmentation, synthesis, validation and inference tricks | [README](kaggle/README.md) |
| **[data/](data/)** | Our data strategy: rules check, label taxonomy, sources, synthesis, augmentation spec, build plan | [README](data/README.md) |
| **[EDA/](EDA/)** | The per-pool analysis plan: what to measure in each pool, why, and which processing knob each answer sets | [README](EDA/README.md) |
| **[processing/](processing/)** | From EDA results to a processing strategy: the stream harness, the decision threads, the per-slice strategies, the acceptance gates | [README](processing/README.md) |
| **[validation/](validation/)** | How we measure: split scheme, metric harness, decision protocol, gates, LB decomposition | [README](validation/README.md) |
| **[architecture/](architecture/)** | How we build the model: the design envelope, pretrained candidates, ensembling, runtime budget | [README](architecture/README.md) |
| **[pipelines/](pipelines/)** | How data reaches the model: the sample contract, sampler, transform registries, collation, invariants | [README](pipelines/README.md) |
| **[training/](training/)** | What we optimize: the objective derived from the metric, what the evidence rules out, the schedule | [README](training/README.md) |

## Reading order for someone new

1. `competition/README.md` — 30-second summary of the task and constraints
2. `competition/03-evaluation.md` — the scoring formula; everything else follows from it
3. `survey/README.md` — the 10 findings that drive design
4. `kaggle/05-transferable-playbook.md` — the practical checklist
5. `data/README.md` — what we're actually building
6. `EDA/README.md` — what is actually in the corpus, and how we find out
7. `validation/README.md` — how we decide whether any of it worked
8. `architecture/README.md` — what we build, and why that and not something else
9. `pipelines/README.md` — the code contract between the corpus and the model
10. `training/README.md` — what the loss does with the batch, and why it is so short

## Confidence marks used throughout

★ verified from a primary source · ☆ secondary source, re-verify before relying on it ·
⚠️ risk or caveat · 🔴 decision-changing · ❌ forbidden by competition rules ·
🔷 our own inference, untested (used in [architecture/](architecture/README.md))

## Status (2026-09-29, end of competition)

| | |
|---|---|
| Data | ✅ built through strategy-v6c (652,165 rows) — [training/15](training/15-data-plan.md), [training/17](training/17-final-runs.md) §4 |
| Training | ✅ 300M runs 1–3, XLS-R-1B DDP main run, fine-tunes A/C (D stopped) — [training/16](training/16-lessons-learned.md), [training/17](training/17-final-runs.md) |
| Best LB | **0.85441** (`1b-v6bA-p2-max3`); final packages: soup/ensemble of runs A and C ([training/17](training/17-final-runs.md) §6) |
| Handoffs | [HANDOFF_TRAINING](HANDOFF_TRAINING.md), [HANDOFF_RUN3](HANDOFF_RUN3.md), [HANDOFF_RUN4](HANDOFF_RUN4.md) (historical snapshots) |

The planning-phase status table below is kept for history.

### Planning-phase status (2026-09-08)

| | |
|---|---|
| Competition docs | ✅ complete (as of 2026-09-04) |
| Prior-art survey | ✅ complete; open items in [survey/10](survey/10-open-questions.md) |
| Kaggle intelligence | ✅ complete (2026-09-05) |
| Data strategy | ✅ planned; **not yet executed** — see [data/08](data/08-build-plan.md) |
| Validation design | ✅ drafted (2026-09-07); gates not yet implemented |
| Architecture design | ✅ candidates ranked (2026-09-07); **nothing trained yet** |
| Data pipeline | ✅ designed (2026-09-08); ⬜ **no code** — see [pipelines/README](pipelines/README.md) |
| Training objective | ✅ designed (2026-09-08), deep-read 4 axes; ⬜ **corrections not applied** — see [training/05](training/05-corrections.md) |
| Modeling | ⬜ not started |

**Next action**: Phase 0 of [data/08-build-plan.md](data/08-build-plan.md) — license audit and
dummy-file forensics, both blocking.
