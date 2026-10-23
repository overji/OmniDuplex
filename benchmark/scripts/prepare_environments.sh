#!/usr/bin/env bash
# Install dependencies only. Never load weights, start services, or run inference.
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=''
BENCH_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BENCH_BASE_PYTHON=${BENCHMARK_BASE_PYTHON:-$BENCH_REPO/benchmark/.venv/bin/python}
cd "$BENCH_REPO"
if [ "$#" -eq 0 ]; then set -- probe gander baichuan omnivinci ming joyai; fi
for backend in "$@"; do
    env_dir="benchmark/.venv-$backend"
    if [ ! -x "$env_dir/bin/python" ]; then
        "$BENCH_BASE_PYTHON" -m venv --system-site-packages "$env_dir"
    fi
    py="$env_dir/bin/python"
    echo "Preparing dependency environment: $backend"
    case "$backend" in
      probe)
        "$py" -m pip install --disable-pip-version-check 'transformers==4.57.3' 'qwen-omni-utils[decord]>=0.0.8' qwen-vl-utils 'accelerate>=1.10,<2' 'numpy<2'
        ;;
      gander)
        "$py" -m pip install --disable-pip-version-check --no-deps -e benchmark/cache/official/Omni-Interaction-Agent-main/minicpm_ft
        "$py" -m pip install --disable-pip-version-check 'transformers==4.51.0' 'accelerate>=1.10,<2' 'av>=14,<18' 'numpy<2' 'librosa>=0.9,<0.11' pyyaml soundfile
        ;;
      baichuan)
        "$py" -m pip install --disable-pip-version-check 'transformers==4.45.2' 'numpy<2' easydict ujson decord cairosvg omegaconf 'diffusers==0.33.0' fire imagesize 'opencv-python-headless==4.8.0.76'
        DS_BUILD_OPS=0 "$py" -m pip install --disable-pip-version-check --no-build-isolation 'deepspeed==0.16.9'
        ;;
      omnivinci)
        "$py" -m pip install --disable-pip-version-check 'transformers==4.46.0' 'numpy<2' 'timm==0.9.12' einops-exts openai-whisper soundfile librosa ftfy hydra-core loguru 'opencv-python-headless==4.8.0.76' beartype kaldiio
        "$py" benchmark/scripts/prepare_omnivinci_transformers.py
        if ! "$py" -c 'import s2wrapper' >/dev/null 2>&1; then
            mkdir -p benchmark/cache/downloads
            curl --noproxy '*' --connect-timeout 10 --max-time 180 -fsSL https://codeload.github.com/bfshi/scaling_on_scales/zip/refs/heads/master -o benchmark/cache/downloads/s2wrapper.zip
            "$py" -m pip install --disable-pip-version-check --no-deps benchmark/cache/downloads/s2wrapper.zip
        fi
        ;;
      joyai)
        uv pip install --python "$py" --torch-backend cu128 'vllm==0.19.1' requests pillow fastapi uvicorn aiohttp
        ;;
      ming)
        "$BENCH_BASE_PYTHON" benchmark/scripts/prepare_ming_text.py
        "$py" -m pip install --disable-pip-version-check 'setuptools<81' wheel
        if [ ! -f benchmark/cache/official/Ming/modeling_bailingmm.py ]; then
            echo 'Ming v1.5 source is missing; see preparation_report.json' >&2
            exit 1
        fi
        "$py" -m pip install --disable-pip-version-check --no-build-isolation 'transformers==4.52.4' 'funasr==1.1.14' hyperpyyaml 'peft==0.13.2' 'diffusers==0.33.0' 'av==12.3.0' 'soundfile==0.12.1' 'librosa<0.11' decord 'openai-whisper==20240930'
        ;;
      *) echo "Unknown backend: $backend" >&2; exit 2 ;;
    esac
done
