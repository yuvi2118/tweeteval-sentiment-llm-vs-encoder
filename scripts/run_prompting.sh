#!/bin/bash
# Step 7: zero-shot, then few-shot, on one GPU. Usage: scripts/run_prompting.sh <gpu_id>
# Each run is its own process so peak GPU memory is measured separately.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${1:-1}
.venv/bin/python -u src/prompt_llm.py --run_name qwen7b-zeroshot --n_shots 0 > logs/qwen7b-zeroshot.log 2>&1
.venv/bin/python -u src/prompt_llm.py --run_name qwen7b-fewshot  --n_shots 9 > logs/qwen7b-fewshot.log 2>&1
