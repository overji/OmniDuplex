#!/usr/bin/env bash
# Fetch public native source only; no models, services, or inference are started.
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
BENCH_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$BENCH_REPO"
mkdir -p benchmark/cache/downloads benchmark/cache/official
if [ ! -f benchmark/cache/official/Omni-Interaction-Agent-main/minicpm_ft/mcpmft/infer/online.py ]; then
  curl --noproxy '*' --connect-timeout 10 --max-time 300 -fsSL https://codeload.github.com/Omni-Interaction-Gander/Omni-Interaction-Agent/tar.gz/refs/heads/main -o benchmark/cache/downloads/gander.tar.gz
  tar xzf benchmark/cache/downloads/gander.tar.gz -C benchmark/cache/official Omni-Interaction-Agent-main/minicpm_ft Omni-Interaction-Agent-main/gander_runtime
fi
if [ ! -f benchmark/cache/official/Ming/modeling_bailingmm.py ]; then
  curl --noproxy '*' --connect-timeout 10 --max-time 300 -fsSL https://codeload.github.com/inclusionAI/Ming/tar.gz/refs/tags/v1.5 -o benchmark/cache/downloads/ming-v15.tar.gz
  mkdir -p benchmark/cache/official/Ming
  tar xzf benchmark/cache/downloads/ming-v15.tar.gz --strip-components=1 -C benchmark/cache/official/Ming --exclude='*.whl' --exclude='*.mp4' --exclude='*.wav'
fi
