#!/bin/bash
# OPTIONAL data-size experiment (NOT run in our final results: it was dropped for time). Usage: scripts/run_all.sh <gpu_id>
# Each run is its own process (so peak memory is measured per run) and has its own log in logs/.
# A failed run does not stop the queue; look at logs/<run>.log.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${1:-1}
PY=.venv/bin/python
run() { name=$1; shift; echo "$(date +%T) start $name"; "$@" > logs/$name.log 2>&1; echo "$(date +%T) end   $name (exit $?)"; }

# data-size experiment: 1% and 10% of the 45,615 training tweets (the 100% points are the runs above)
run roberta-large_1pct  $PY -u src/finetune_encoder.py --run_name roberta-large_1pct  --max_train 456
run roberta-large_10pct $PY -u src/finetune_encoder.py --run_name roberta-large_10pct --max_train 4562
run qwen7b-lora_1pct    $PY -u src/finetune_llm_lora.py --run_name qwen7b-lora_1pct  --max_train 456
run qwen7b-lora_10pct   $PY -u src/finetune_llm_lora.py --run_name qwen7b-lora_10pct --max_train 4562

echo "$(date +%T) ALL DONE"
