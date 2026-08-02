#!/usr/bin/env bash
# Download MELD.Raw and extract per-utterance WAV so the wav2vec2 prosody pipeline can run.
#
# MELD ships as ~10 GB of .mp4 clips named dia<D>_utt<U>.mp4 per split. This script
# fetches the official tarball, extracts it, and transcodes each clip to 16 kHz mono WAV
# named dia<D>_utt<U>.wav under <split>_wav/, which _resolve_meld_audio_path() expects.
#
# Prereqs: curl, tar, ffmpeg  (macOS: brew install ffmpeg)
# Usage:   bash scripts/download_meld_raw.sh datasets/MELD
set -euo pipefail

DEST="${1:-datasets/MELD}"
URL="https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
mkdir -p "$DEST"

command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not found (brew install ffmpeg)"; exit 1; }

TARBALL="$DEST/MELD.Raw.tar.gz"
EXPECTED_BYTES=10878146150   # full MELD.Raw.tar.gz size; used to gate extraction

# Resumable download: -C - continues a partial file; --retry rides out the connection
# resets this host throws on long transfers. Skip entirely if already complete (a resume
# attempt on a finished file returns HTTP 416 and would fail with --fail).
have=$( [ -f "$TARBALL" ] && wc -c < "$TARBALL" | tr -d ' ' || echo 0 )
if [ "$have" = "$EXPECTED_BYTES" ]; then
  echo "Tarball already complete ($EXPECTED_BYTES bytes); skipping download."
else
  echo "Downloading MELD.Raw (~10 GB, resumable) -> $TARBALL"
  curl -L --fail -C - --retry 10 --retry-delay 5 --retry-all-errors -o "$TARBALL" "$URL"
  actual=$(wc -c < "$TARBALL" | tr -d ' ')
  if [ "$actual" != "$EXPECTED_BYTES" ]; then
    echo "ERROR: $TARBALL is $actual bytes, expected $EXPECTED_BYTES (incomplete). Re-run to resume."
    exit 1
  fi
fi

# Canonical MELD split -> source clip dir mapping.
RAW="$DEST/MELD.Raw"
TRAIN_SRC="$RAW/train_splits"
DEV_SRC="$RAW/dev_splits_complete"
TEST_SRC="$RAW/output_repeated_splits_test"

# Extract only if the split dirs aren't already present (idempotent; avoids re-untarring 10 GB).
if [ ! -d "$TRAIN_SRC" ] || [ ! -d "$DEV_SRC" ] || [ ! -d "$TEST_SRC" ]; then
  echo "Extracting $TARBALL"
  tar -xzf "$TARBALL" -C "$DEST"
  for inner in "$RAW"/*.tar.gz; do
    [ -f "$inner" ] && tar -xzf "$inner" -C "$RAW"
  done
else
  echo "Split dirs already present; skipping extraction."
fi

# Transcode each clip to 16 kHz mono WAV under <split>_wav/, named dia<D>_utt<U>.wav.
# Fault-tolerant: MELD ships a few corrupt clips (e.g. train dia125_utt3, 'moov atom not
# found'); a bad file is logged and skipped rather than aborting the whole run.
transcode_dir() {
  local src="$1"
  local split="$2"
  local out="$DEST/${split}_wav"
  [ -d "$src" ] || { echo "  WARN: $src missing; skipping $split"; return 0; }
  mkdir -p "$out"
  local ok=0 skip=0
  while IFS= read -r f; do
    local target="$out/$(basename "${f%.mp4}").wav"
    [ -f "$target" ] && { ok=$((ok+1)); continue; }
    if ffmpeg -nostdin -loglevel error -y -i "$f" -ac 1 -ar 16000 "$target" 2>/dev/null; then
      ok=$((ok+1))
    else
      rm -f "$target"; skip=$((skip+1))
      echo "  skip corrupt: $(basename "$f")"
    fi
  done < <(find "$src" -name 'dia*_utt*.mp4')
  echo "  $split: $ok wav ok, $skip skipped -> $out"
}

echo "Transcoding clips to 16 kHz mono WAV"
set +e            # individual ffmpeg failures must not abort the batch
transcode_dir "$TRAIN_SRC" train
transcode_dir "$DEV_SRC"   dev
transcode_dir "$TEST_SRC"  test
set -e

echo "Done. Point --audio_root at $DEST when running build_meld."
