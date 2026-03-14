#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/mmsi}"
DOWNLOAD_ROOT="${PROJECT_ROOT}/data/downloads"

BENCHMARK_URL="https://www.dropbox.com/scl/fo/fbv6njzu1ynbgv9wgtrwo/ANPk2TKqK2rl44MqKu05ogk?rlkey=yx7bmzmmiymauvz99q2rvjajg&st=305631zj&dl=1"
KEYPOINT_URL="https://www.dropbox.com/scl/fo/01rp8c126kc9014kbhvkg/AO2JvbsFuMd4WkkwzzOR06U?rlkey=910f1sf90zm6piii0krepikzi&st=u36zodh8&dl=1"

mkdir -p "${DATA_ROOT}/benchmark" "${DATA_ROOT}/keypoints" "${DOWNLOAD_ROOT}"

curl -L "${BENCHMARK_URL}" -o "${DOWNLOAD_ROOT}/datasets.zip"
curl -L "${KEYPOINT_URL}" -o "${DOWNLOAD_ROOT}/keypoints.zip"

# The zip files contain a stray root entry, but the actual data extracts correctly.
unzip -q -o "${DOWNLOAD_ROOT}/datasets.zip" -d "${DATA_ROOT}/benchmark" || true
unzip -q -o "${DOWNLOAD_ROOT}/keypoints.zip" -d "${DATA_ROOT}/keypoints" || true

echo "Data ready under ${DATA_ROOT}"
