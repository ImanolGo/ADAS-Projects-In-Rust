#!/usr/bin/env bash
# Download the sample dashcam clips used by Activities 1 and 7 into data/.
#
# The clips come from Udacity's CarND-LaneLines-P1 project (MIT licence,
# Copyright (c) 2016-2019 Udacity, Inc.).
#
# GTSRB: 51 test images and their labels (data/Test, data/Test.csv) are already
# in the repository. For the full 12,630-image test set, download the Kaggle
# "GTSRB - German Traffic Sign Recognition Benchmark" dataset and put its
# Test/ folder and Test.csv into data/.
set -euo pipefail
cd "$(dirname "$0")/../data"

BASE="https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos"
declare -A CLIPS=(
  [video.mp4]=solidYellowLeft.mp4   # the default for lane_detection and adas_drive
  [solidWhiteRight.mp4]=solidWhiteRight.mp4
  [challenge.mp4]=challenge.mp4     # curves, shadows and a bright concrete bridge
)

fetch_with_curl() {
  for out in "${!CLIPS[@]}"; do
    [[ -s "$out" ]] && { echo "✓ $out (already there)"; continue; }
    curl -fsSL -o "$out" "$BASE/${CLIPS[$out]}" && echo "✓ $out"
  done
}

fetch_with_git() {
  local tmp
  tmp="$(mktemp -d)"
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/udacity/CarND-LaneLines-P1 "$tmp"
  git -C "$tmp" sparse-checkout set test_videos
  for out in "${!CLIPS[@]}"; do
    [[ -s "$out" ]] || cp "$tmp/test_videos/${CLIPS[$out]}" "$out"
    echo "✓ $out"
  done
  rm -rf "$tmp"
}

fetch_with_curl 2>/dev/null || { echo "Direct download failed, trying git…"; fetch_with_git; }

command -v ffmpeg >/dev/null || echo "Note: install ffmpeg to read and write video (apt install ffmpeg / brew install ffmpeg)."
