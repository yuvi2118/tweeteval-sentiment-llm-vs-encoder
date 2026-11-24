#!/bin/bash
# Runs every remaining experiment one after another on ONE GPU. Usage: scripts/run_all.sh <gpu_id>
# Each run is its own process (so peak memory is measured per run) and has its own log in logs/.
# A failed run does not stop the queue; look at logs/<run>.log.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${1:-1}
PY=.venv/bin/python
run() { name=$1; shift; echo "$(date +%T) start $name"; "$@" > logs/$name.log 2>&1; echo "$(date +%T) end   $name (exit $?)"; }

# --- 1. main systems, now also scored on our own test set
run tfidf-lr          $PY src/baseline_tfidf.py --run_name tfidf-lr
run bilstm-glove      $PY -u src/bilstm.py --run_name bilstm-glove
run qwen7b-zeroshot-own $PY -u src/prompt_llm.py --run_name qwen7b-zeroshot --n_shots 0 --own_only
run qwen7b-fewshot-own  $PY -u src/prompt_llm.py --run_name qwen7b-fewshot  --n_shots 9 --own_only
run roberta-large     $PY -u src/finetune_encoder.py --run_name roberta-large
run qwen7b-lora       $PY -u src/finetune_llm_lora.py --run_name qwen7b-lora

# --- 2. optional extras
run bertweet-large    $PY -u src/finetune_encoder.py --run_name bertweet-large --model vinai/bertweet-large
run qwen7b-qlora      $PY -u src/finetune_llm_lora.py --run_name qwen7b-qlora --qlora
echo "$(date +%T) ALL DONE"
