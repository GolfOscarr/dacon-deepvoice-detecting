#!/usr/bin/env bash
# D-c: real voice from S3 -- Common Voice ko (whole, CC0) and LibriTTS-R
# train-clean-100 (+doc, CC BY 4.0). Rerunnable: `aws s3 sync` skips what is
# already local and the extract step overwrites in place.
#
# Layout mirrors the other sources: raw/<name>/<version>/{_meta,payload} and
# interim/<name>/<version>/<unpacked tree>, which is what configs/eda.yaml's
# `root:` entries (common-voice-ko/cv-26.0, libritts-r/openslr-141) expect.
set -euo pipefail
REPO=${REPO:-/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting}
CORPUS=${CORPUS:-/data/project/private/dacon-corpus}
PY=${PY:-/data/project/private/dacon-venvs/dacon311/bin/python}
cd "$REPO"
"$PY" scripts/fetch_from_s3.py common-voice-ko --dest "$CORPUS/raw" --extract "$CORPUS/interim"
"$PY" scripts/fetch_from_s3.py libritts-r --only 'train_clean_100.tar.gz' --only 'doc.tar.gz' \
      --dest "$CORPUS/raw" --extract "$CORPUS/interim"
