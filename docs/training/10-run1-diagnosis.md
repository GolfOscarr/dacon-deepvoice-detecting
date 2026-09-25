# Run 1 diagnosis — where the errors are and what fixes them

*2026-09-26, from `first-v3` fold VAL predictions. Tool: `scripts/diag/run_breakdown.py`
(outputs `runs/first-v3/diag/{summary.json,slice_rates.csv,joined.parquet}`). "err" is the error
rate of a slice at the fold's pooled EER threshold: false-alarm rate for a real slice, miss rate
for a fake slice. That says which side carries the error, which a one-vs-all slice EER hides.*

## 1 · Findings

| # | finding | numbers | reading |
|---|---|---|---|
| F1 | **Held-out clean read speech is called fake.** The fold-0 voice error is almost all real LJSpeech false alarms, not missed vocoders | fold 0: real `ljspeech` err **0.52** (p̄ 0.77, n 215); WaveFake families err 0.04–0.07; real `libritts-r` err 0.00 | D1's matched-pair gap is this: the model learned "LJ-like studio read speech ⇒ fake". English fakes are 181.5 h of WaveFake (one speaker, LJ) vs 134 h of English real from three corpora |
| F2 | **Fold 1 errs on both sides** | real `musan-speech` err **0.49** (p̄ 0.60, n 656); fake `ko-synth/xtts` err **0.31** (p̄ 0.71); `mlaad/optispeech` 0.30, `Soprano11` 0.25, `parler_tts_mini_v1` 0.14 | musan-speech (LibriVox / gov read speech) is another unseen clean read-speech corpus → the F1 shortcut again. XTTS is the unseen-cloner problem (D2) |
| F3 | **Korean real is fine where it is Zeroth; crowd-sourced Korean is not** | `zeroth-korean` err 0.00–0.01 all folds; `common-voice-ko` err 0.05 / 0.08 / **0.25** (n 611 / 160 / 61) | real Korean from consumer mics / lossy uploads leans fake. The test's real side (voice-phishing context) is closer to CV than to Zeroth. Emilia-ko (88 h in-the-wild) is the fix already built |
| F4 | **Validation has no augmentation or test-chain transforms** | `transforms == []` in all 24,000 VAL specs; `normalize` = wav/mono | VAL measures clean composed audio only — part of why it reads 0.95 (D4). A codec/telephony VAL view is needed to judge robustness |
| F5 | **Music ranks well but is poorly scaled on unseen families; the file head ignores it** | music err ≤ 0.06 on every fold, but held-out fake p̄ 0.38 (`stable_audio_open`), 0.56 (`musicldm`). File head misses **0.36** of fold-0 cell 6 (real voice + fake music) | the music head is not the weak part; FILE_FAKE is. See F6 |
| F6 | **A post-hoc stacker on the five outputs lowers file EER** | leave-one-fold-out logistic on logits: file EER 0.1005→0.077, 0.1005→0.078, 0.0113→0.023, 0.0438→0.029 (mean 0.064→0.052, ≈ +0.005 Score); `max(v·vp, m·mp)` alone: mean 0.060 | **not shipped.** Built as an opt-in (`processing/file_stack.py`, `scripts/diag/fit_file_stack.py`, `package_submission.py --file-stack`). The gain rides on the *ungated* music logit (coef 4.8): on music-free VAL clips it sits at ≈ −1.6, but on the competition's own TEST_0000 (speech only) the music head reads 0.23 and the stack moved FILE 0.07 → **0.58**. The test's music-free behaviour differs from VAL, so the stack is a transfer risk. The presence-free, gated-only variant gains nothing (0.074) |
| F7 | **Longer training hurt fold 0** | fold 0 mid-run score 0.949 at step 24,000 (EMA) vs 0.928 final at 54,000; fold 3 0.968 → 0.966 | consistent with shortcut learning growing with steps. Learning curves for folds 0/1 (passes 1…17) are running (`scripts/diag/learning_curve.sbatch`, jobs 220484/220485) — **PENDING** |

What is **not** the problem: known vocoder/TTS families (err ≤ 0.07 almost everywhere), Zeroth
Korean real, presence (AUC ≈ 1), music ranking.

## 2 · What to change for run 2, by expected gain

1. **Break the "clean read speech = fake" shortcut (F1, F2).**
   - Real English diversity: add **Emilia-EN** (+YODAS) through the emilia-ko reader, 40–80 h, many speakers.
   - English fakes from many speakers: run the ko-synth2 cloners (chatterbox, cosyvoice, fishspeech,
     maskgct, seedvc) on **English** prompts from those real speakers → same-speaker real/fake pairs.
   - Down-weight WaveFake (cap its domain share) so English fakes are not "the LJ speaker".
   - Keep S1 processing on both labels (already built).
2. **Unseen Korean cloners (F2, D2):** ko-synth2's five families, prompts from Emilia-ko / Zeroth
   speakers (verified in `ko-synth2/*/metadata.csv`: `prompt_speaker` ∈ {emilia-ko, zeroth-korean}).
   Keep each prompt speaker's reals and clones in the same fold.
3. **In-the-wild Korean real (F3):** Emilia-ko — built.
4. **Training length (F7):** pick from the learning curves; option B (init from run 1, ~20k steps)
   is favoured if the curves peak early.
5. **FILE_FAKE (F6):** the stacker is off (transfer risk, see F6). Better fix in training: make the file head see the music evidence (loss weight on cell 4/6, or `file_head.mode` comparison on VAL).
6. **A robustness VAL view (F4):** score the fold models on the same VAL specs with the test-chain
   normalize menu + codec augments on, to see how much the clean-VAL score overstates.
