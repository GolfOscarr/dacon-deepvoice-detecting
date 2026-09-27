#!/bin/bash
# venvs for the English round-3 families (en-synth3). usage: en_install.sh <family>
# All under /data/project/private (never /home). Logs: interim/en-synth3/_logs/install-<family>.log
set -eu
F=${1:?family}
V=/data/project/private/dacon-venvs/synth3en-$F
R=/data/project/private/dacon-weights/synth3/repos
export UV_CACHE_DIR=/data/project/private/dacon-venvs/_uv-cache
IDX="--extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match"
uv venv -q --python 3.11 "$V"
P="uv pip install -q --python $V/bin/python"
case $F in
  f5tts)
    $P $IDX torch==2.5.1 torchaudio==2.5.1
    $P $IDX "$R/F5-TTS" torch==2.5.1 torchaudio==2.5.1 espeakng-loader scipy ;;
  zonos)
    $P $IDX torch==2.5.1 torchaudio==2.5.1
    $P $IDX -e "$R/Zonos" torch==2.5.1 torchaudio==2.5.1 espeakng-loader scipy ;;
  sparktts)
    $P $IDX -r "$R/Spark-TTS/requirements.txt" scipy huggingface_hub ;;
  styletts2)
    $P $IDX torch==2.5.1 torchaudio==2.5.1
    $P $IDX SoundFile munch pydub pyyaml librosa nltk matplotlib accelerate "transformers<4.47" einops einops-exts tqdm \
       "git+https://github.com/resemble-ai/monotonic_align.git" phonemizer espeakng-loader scipy torch==2.5.1 torchaudio==2.5.1 ;;
  *) echo "unknown family $F"; exit 2 ;;
esac
"$V/bin/python" -c "import torch, torchaudio; print('ok', torch.__version__, torchaudio.__version__)"
