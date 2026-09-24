# ko-synth — synthesised Korean fake voice (track K, plan 07 §1 D-a)

Root: `/data/project/private/dacon-corpus/interim/ko-synth/<family>/` — `*.wav` (mono PCM16 at the
generator's native rate) + `metadata.csv` (columns: file, family, model, model_revision, licence,
text, text_source_utts, prompt_speaker, prompt_files, seed, duration_s, sample_rate).
Scripts: `scripts/synth/` in the repo (`synth_common.py`, `synth_<family>.py`, `run_family.sbatch`,
`verify_ko_synth.py`). `_index/zeroth_index.csv` is the cached Zeroth index (22,720 utterances,
115 speakers, 52.87 h); `_smoke/` holds the 3-4-file smoke outputs per family (not corpus data);
`_logs/` holds Slurm and install logs.

## Families

| family | architecture | model (revision) | licence | sample rate | speakers | text / prompt |
|---|---|---|---|---|---|---|
| mms | VITS (end-to-end) | facebook/mms-tts-kor @ 1b64913 | CC-BY-NC-4.0 (model card) | 16 kHz | 1 built-in (`mms/kor`) | 1-2 Zeroth sentences, uroman-romanised; seeded noise/duration/rate draws |
| melo | VITS + BERT prosody | myshell-ai/MeloTTS-Korean @ 0207e5a, code MeloTTS @ 2091453 | MIT (repo LICENSE + model card) | 44.1 kHz | 1 built-in (`melo/KR`) | 1-2 Zeroth sentences; seeded speed/noise/sdp draws |
| knnvc | WavLM-Large kNN matching + HiFi-GAN (voice conversion) | bshall/knn-vc @ c616845, release v0.1 weights (prematched) | MIT (repo LICENSE) | 16 kHz | 115 Zeroth targets | source = 1-2 Zeroth utterances of another speaker; matching set = 8 utterances of the target |
| xtts | GPT codec LM + HiFi-GAN decoder, zero-shot cloning | coqui/XTTS-v2 @ 6c2b0d7, coqui-tts 0.27.5 | CPML 1.0.0 — non-commercial only (LICENSE.txt in the HF repo) | 24 kHz | 115 Zeroth prompts | 3 prompt utterances (≤ 30 s); 1-2 sentences from other utterances, one inference per sentence |
| openvoice | tone-colour conversion (flow-based VC) over MeloTTS-KR | myshell-ai/OpenVoiceV2 @ f36e7ed (code OpenVoice @ 74a1d14) + MeloTTS-Korean | MIT (both) | 22.05 kHz | 115 Zeroth targets | MeloTTS-KR speech (own seed, own texts) converted with tgt_se = mean ref-encoder embedding of 3 prompt utterances; watermark disabled |
| cosyvoice | Qwen speech-token LM + flow matching + HiFT, zero-shot cloning | FunAudioLLM/CosyVoice2-0.5B @ eec1ae6, code CosyVoice @ 074ca6d | Apache-2.0 (model card + repo LICENSE) | 24 kHz | 115 Zeroth prompts | 1 prompt utterance (3-10 s) with its transcript; 1-2 sentences from other utterances, text front end off |
| bark | semantic/coarse/fine GPT LMs + EnCodec | suno/bark @ 70a8a7d (transformers BarkModel, fp16) | MIT (model card, repo) | 24 kHz | 10 presets `v2/ko_speaker_0..9` | 1 Zeroth sentence per file (Bark caps output at ~13 s) |

Licence notes (read from the repos / model cards, not assumed):
- mms (CC-BY-NC-4.0) and xtts (CPML) are non-commercial. CPML also says outputs are for
  non-commercial use; it does not forbid sharing the outputs with a third party for non-commercial
  purposes. Both are recorded per file in `licence`; exclude them if the rules reading changes.
- MeloTTS-Korean's text front end pulls `kykim/bert-kor-base` (no licence tag on the card). Only
  its hidden states feed prosody; it does not appear in the output data.
- kNN-VC downloads WavLM-Large (MIT, microsoft/unilm) via the bshall release.

## Text and speaker draw (`synth_common.py`)

- Texts: Zeroth transcripts. Each file = 1 sentence (source ≥ 4 s, 55 %) or 2 sentences drawn
  uniformly over the corpus, capped at 15 s of source speech. Bark uses single sentences (4-11 s
  source, ≤ 90 chars).
- Cloning / VC: prompt speakers cycle through all 115 Zeroth speakers (order shuffled by the family
  seed), so every family covers every speaker roughly evenly; the text never comes from a prompt
  utterance. kNN-VC's source speaker is always ≠ target speaker.
- Seeds: one fixed seed per family (mms 101, knnvc 202, xtts 303, melo 404, cosyvoice 505,
  openvoice 606, bark 707); job list = f(family, seed, n_files), so re-running with the same
  arguments regenerates the same job list; per-file `seed` = seed·1000003 + idx is used for the
  sampling RNGs. Files < 3.0 s or with RMS < 1e-3 after edge-silence trimming are dropped (counted
  in the logs).
- Resume: `metadata.csv` rows are appended with a file lock; a re-run skips files already listed.
  `--shard i/n` splits a family over n GPUs.

## Environment / install notes

One uv venv per family under `/data/project/private/dacon-venvs/synth-<family>/` (python 3.11),
weights under `/data/project/private/dacon-weights/synth/` (`HF_HOME=.../hf`, `TORCH_HOME=.../torch`,
CosyVoice repo clone at `.../cosyvoice-repo`). GPU runs via `sbatch scripts/synth/run_family.sbatch
<family> [args]` (1 GPU, 6 CPUs per job). Traps hit:

- `coqui-tts` needs `torch`, `torchaudio`, `torchcodec` (torch ≥ 2.9 audio IO) and `transformers<5`
  installed explicitly.
- torchaudio 2.14 `load` needs torchcodec; the kNN-VC script feeds tensors instead of paths.
- MeloTTS / OpenVoice: `setuptools<70` (librosa 0.9 imports pkg_resources), `python-mecab-ko`
  (g2pkk), `python -m unidic download`. OpenVoice installed `--no-deps`; the converter is built
  without `wavmark` (watermark off).
- CosyVoice: pinned `openai-whisper==20231117` fails to build under uv → `>=20240930`; deepspeed,
  tensorrt, gradio left out; `setuptools<70`. onnxruntime-gpu 1.18 cannot load its CUDA provider on
  this node (libcublasLt.so.11), so the speech tokenizer / campplus ONNX models run on CPU — works,
  RTF ≈ 0.5.
- Everything else installed from PyPI with default torch (2.14 + cu130) and ran on the H200s.

## Results
(filled in after the runs — see the end of this file)
