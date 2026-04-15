#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation"
SRC_DIR="${PROJECT_ROOT}/data/raw_video_sources/werewolf_among_us/Youtube/videos"
DEST_DIR="${PROJECT_ROOT}/data/raw_video_sources/werewolf_among_us/Youtube/videos_mmsi_named"
BENCH_DIR="${PROJECT_ROOT}/data/mmsi/benchmark/youtube/transcripts/anonymized"

mkdir -p "${DEST_DIR}"

python - <<'PY'
from pathlib import Path
import os
import re

project_root = Path("/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation")
src_dir = project_root / "data/raw_video_sources/werewolf_among_us/Youtube/videos"
dest_dir = project_root / "data/raw_video_sources/werewolf_among_us/Youtube/videos_mmsi_named"
bench_dir = project_root / "data/mmsi/benchmark/youtube/transcripts/anonymized"

def normalize(text: str) -> str:
    text = text.replace("#", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

bench_names = [p.stem for p in bench_dir.glob("*.txt")]
video_names = [p.stem for p in src_dir.glob("*.mp4")]

video_lookup = {normalize(name): name for name in video_names}
created = 0
missing = []

for bench_name in bench_names:
    key = normalize(bench_name)
    match = video_lookup.get(key)
    if not match:
        missing.append(bench_name)
        continue
    src = src_dir / f"{match}.mp4"
    dst = dest_dir / f"{bench_name}.mp4"
    if not dst.exists():
        os.symlink(src, dst)
        created += 1

print(f"created_symlinks={created}")
print(f"total_expected={len(bench_names)}")
print(f"missing={len(missing)}")
if missing:
    print("missing_examples=")
    print("\n".join(missing[:10]))
PY
