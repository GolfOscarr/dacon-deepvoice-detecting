---
name: dacon-talkboard-sync
description: Fetch the latest Q&A from the DACON 236749 talkboard, diff it against the last snapshot, and update docs/competition/05-talkboard-qa.md. Use when asked to check for new competition questions/answers, sync the talkboard, or see what changed on the DACON site.
---

# DACON talkboard sync

Keeps [`docs/competition/05-talkboard-qa.md`](../../../docs/competition/05-talkboard-qa.md) current
with https://dacon.io/competitions/official/236749/talkboard.

## Why a script

DACON is a Nuxt SSR app. `WebFetch` and plain `curl` return only a ~6 KB shell — the post list,
bodies and replies live in an embedded `window.__NUXT__=(function(a,b,...){...}(args))` blob.
`scripts/fetch_talkboard.py` parses that state directly. **Do not try to WebFetch these URLs.**

## Run

```bash
python3 .claude/skills/dacon-talkboard-sync/scripts/fetch_talkboard.py \
  --out /tmp/talkboard \
  --snapshot docs/competition/talkboard-snapshot.json
```

Stdin-free, no auth, ~10 s. Prints a change report and writes `/tmp/talkboard/posts.json`:

```
posts: 7 | new: 1 | reply-count changed: 1
  NEW      #417333  r=0  [DACON 답변 요청] Real/Fake 라벨 정의와 ...
  CHANGED  #417198  replies 1 -> 2  ...
```

Flags: `--no-bodies` for a fast list-only check.

## Procedure

1. **Run the script.** If it errors with `no __NUXT__ state`, the site layout changed — report that
   and stop; do not guess at the content.
2. **If `no changes`** — say so in one line and stop. Do not touch the docs.
3. **For each NEW post**: read its `content` array in `posts.json`. Blocks are ordered; typically
   the question body first, then any official replies. The reply from DACON is usually the block
   that reads as an answer (starts 문의주신 / 안녕하세요 / 네 …). If the reply count is > 0 but no
   reply text appears, the reply was not server-rendered — say so explicitly rather than inventing
   it (post #417193 has this behaviour).
4. **For each CHANGED post**: a reply-count increase usually means DACON answered. Re-read its
   content and extract the new reply.
5. **Update `docs/competition/05-talkboard-qa.md`**:
   - add a section per post using the existing format: `## #<id> — <title>` , *(author, date)*,
     the question summarized, then the official answer **quoted in Korean** as a blockquote,
     then a short **Takeaway** in English
   - keep posts in the file's existing order (pinned notice first, then newest first)
   - update the "snapshot as of" date at the top
   - update the **Cross-cutting summary** table if an answer settles a new question
   - move anything now answered out of the **Open questions** list at the bottom
6. **Propagate consequences.** An answer often invalidates a plan assumption. Check and update:
   - [`docs/competition/04-rules.md`](../../../docs/competition/04-rules.md) — if a rule is clarified
   - [`docs/survey/10-open-questions.md`](../../../docs/survey/10-open-questions.md) — V/G items
   - [`docs/data/`](../../../docs/data/README.md) — especially `01-rules-check.md` (license gate),
     `02-label-taxonomy.md` (label semantics), `06-augmentation-spec.md` (what must be shipped)
   - [`PROGRESS.md`](../../../PROGRESS.md) — open decisions
7. **Commit the snapshot**: copy `/tmp/talkboard/posts.json` to
   `docs/competition/talkboard-snapshot.json` so the next run diffs against this state.
8. **Report** to the user: what's new, what it changes, and what it does *not* settle. Flag
   unanswered `[DACON 답변 요청]` posts — questions asked by other participants that we benefit
   from watching.

## Rules

- **Quote official answers verbatim in Korean.** Paraphrase only in the Takeaway line.
- **Never invent an answer.** `replies > 0` with no rendered reply text means unknown, not absent.
- Distinguish **official DACON replies** (author `DACON.GM` or similar staff account) from
  participant comments. Only the former settle a rule.
- If an answer contradicts something in our docs, **say so loudly** rather than quietly editing —
  it may invalidate work already done.
