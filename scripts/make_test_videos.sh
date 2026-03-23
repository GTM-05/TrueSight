#!/usr/bin/env bash
# Synthetic MP4s for pipeline smoke tests (not representative of human vs deepfake accuracy).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/samples"
mkdir -p "$OUT"

# A: SMPTE bars + 1 kHz tone (5 s)
ffmpeg -y -f lavfi -i "smptebars=duration=5:size=640x480:rate=24" \
  -f lavfi -i "sine=frequency=1000:sample_rate=48000:duration=5" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
  "$OUT/Synthetic_Bars_Tone.mp4"

# B: test pattern + 220 Hz (5 s)
ffmpeg -y -f lavfi -i "testsrc=duration=5:size=640x480:rate=24" \
  -f lavfi -i "sine=frequency=220:sample_rate=48000:duration=5" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
  "$OUT/Synthetic_TestPattern_Tone.mp4"

echo "Wrote:"
ls -la "$OUT"/*.mp4
