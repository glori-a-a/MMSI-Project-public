#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-MMSI-Project-public}"

git add .
git commit -m "Initial public project export"

gh repo create "${REPO_NAME}" --public --source=. --remote=origin --push
