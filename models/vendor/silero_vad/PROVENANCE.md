# Silero VAD — provenance

| | |
|---|---|
| Upstream | https://github.com/snakers4/silero-vad |
| File | `src/silero_vad/data/silero_vad.jit` |
| Fetched | 2026-09-14 from `master` |
| Bytes | 2,272,526 |
| sha256 | `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720` |
| Licence | MIT — `LICENSE`, copied from the same ref |

```bash
curl -sL -o models/vendor/silero_vad/silero_vad.jit \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit
curl -sL -o models/vendor/silero_vad/LICENSE \
  https://raw.githubusercontent.com/snakers4/silero-vad/master/LICENSE
sha256sum models/vendor/silero_vad/silero_vad.jit
```

The distribution is the TorchScript archive itself, so there is no source to
keep byte-for-byte and no upstream edit to record. `__init__.py` beside it is
**ours**: a loader that verifies the digest and a chunker that resets the
model's recurrent state per file.

⚠️ The digest is checked at load (`MODEL_SHA256`). Re-vendoring a newer upstream
means updating it here *and* in `__init__.py`, and re-running the content tier —
a different model produces different numbers under the same column names.

## Validation at vendor time

Run on three files each from six sources, chain plane, threshold 0.5
(docs/EDA/05 and 03 carry the full result):

| source | expected | measured speech ratio |
|---|---|---|
| `ljspeech` | speech | 0.910, 0.966, 0.944 |
| `zeroth-korean` | speech | 0.783, 0.651, 0.773 |
| `fma` | instrumental | 0.890, 0.332, 0.000 |
| `fakemusiccaps` | instrumental | 0.000, 0.000, 0.000 |
| `musan-noise` | noise | 0.002, 0.074, 0.000 |
| `sonics` | AI song with vocals | 0.364, 0.311, 0.152 |

It fires on speech and stays quiet on generated instrumental and on noise, which
is the check that a VAD is doing anything at all. 🔴 The `fma` row is not a
calibration failure — it is [C2](../../../docs/EDA/03-pool-c-real-instrumental.md)'s
question answering itself on the third file tried.
