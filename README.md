# Tweet sentiment: PEFT of LLMs vs encoder transformers

Term project for *Natural Language Processing with Deep Learning*.

**Question:** does a LoRA-fine-tuned 7B LLM beat a fully fine-tuned encoder transformer
(RoBERTa-large) on tweet sentiment classification, is the extra computational cost worth it,
and where do all the models fail?

Data: [TweetEval, sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (negative / neutral / positive),
official train (45,615), validation (2,000) and test (12,284) splits, plus a small own test set
(`data/own_test.csv`).

## Systems

| run name | system | what is trained |
|---|---|---|
| `tfidf-lr` | TF-IDF + Logistic Regression | all (286k weights) |
| `bilstm-glove` | 2-layer BiLSTM, GloVe Twitter 100d | all (2.6M) |
| `roberta-large` | RoBERTa-large, full fine-tuning | all (355M) |
| `qwen7b-lora` | Qwen2.5-7B-Instruct + LoRA (r=16) as a classifier | 40M (0.57%) |
| `qwen7b-zeroshot` | Qwen2.5-7B-Instruct, prompt only | nothing |
| `qwen7b-fewshot` | same + 9 solved examples in the prompt | nothing |
| extras | `bertweet-large` (full FT), `qwen7b-qlora` (4-bit base) | |

All runs use seed 42. Every system is scored by the same function (`src/common.py`), and speed
and memory are measured with the same inference batch size (32), each run in its own process.

## Project layout

```
src/
  common.py            load data, metrics, save results, speed measurement
  explore_data.py      look at the dataset (sizes, class balance, examples)
  baseline_tfidf.py    system: TF-IDF + logistic regression
  bilstm.py            system: BiLSTM + GloVe
  finetune_encoder.py  system: full fine-tuning of an encoder (RoBERTa-large, BERTweet-large)
  finetune_llm_lora.py system: LoRA / QLoRA fine-tuning of an LLM as a classifier
  prompt_llm.py        systems: zero-shot and few-shot prompting
  make_report.py       tables, figures and the hard-examples file from the saved results
scripts/run_all.sh     runs the main systems and the two extras in order on one GPU
scripts/run_ablation.sh optional data-size experiment (not run for our results)
annotation/            hard tweets with their causes (first reading, assigned by an LLM)
data/own_test.csv      own test set (180 posts, 60 per class)
results/<run>/         meta.json (metrics and cost) for each run; prediction CSVs are not committed
results/figures/       PNG figures for the slides
docs/                  RESULTS.md, EXPLAINER.md
logs/                  one log per run (not committed)
```

## Setup

Python 3.12 and an NVIDIA GPU (the 7B runs need about 15-20 GB of GPU memory).

```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
```

Model and dataset downloads go into `.hf_cache/` inside the project (set in `src/common.py`).
Qwen2.5-7B-Instruct is not gated, so no Hugging Face token is needed.

## Run

Quick look at the data:

```bash
python src/explore_data.py
```

One command per system (run from the project root; add `--max_train 500 --results_root results_smoke`
for a quick test that does not touch the real results):

```bash
python src/baseline_tfidf.py --run_name tfidf-lr
python src/bilstm.py --run_name bilstm-glove
python src/finetune_encoder.py --run_name roberta-large
python src/finetune_llm_lora.py --run_name qwen7b-lora
python src/prompt_llm.py --run_name qwen7b-zeroshot --n_shots 0
python src/prompt_llm.py --run_name qwen7b-fewshot --n_shots 9
```

Data-size experiment (1% and 10% of the training tweets; the 100% points are the runs above).
**Not run for our final results** (dropped for time); the commands are in `scripts/run_ablation.sh`:

```bash
python src/finetune_encoder.py --run_name roberta-large_1pct  --max_train 456
python src/finetune_encoder.py --run_name roberta-large_10pct --max_train 4562
python src/finetune_llm_lora.py --run_name qwen7b-lora_1pct   --max_train 456
python src/finetune_llm_lora.py --run_name qwen7b-lora_10pct  --max_train 4562
```

Optional extras:

```bash
python src/finetune_encoder.py --run_name bertweet-large --model vinai/bertweet-large
python src/finetune_llm_lora.py --run_name qwen7b-qlora --qlora
```

The main systems and extras in order on one GPU, then the report:

```bash
scripts/run_all.sh 1          # GPU id; logs go to logs/
python src/make_report.py     # tables, figures, hard examples
```

Pick the GPU with `CUDA_VISIBLE_DEVICES`. On a shared server use only an idle GPU.

## Own test set

`data/own_test.csv` has the columns `text,label` (`negative`, `neutral`, `positive`): 180 short
posts, 60 per class. The labels were assigned with an LLM. If the file is missing or empty,
the own-set evaluation is skipped and everything else still runs.

## Results

See `docs/RESULTS.md` for the tables, figures and the discussion, and `results/summary.md` for the raw table.
