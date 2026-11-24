"""Step 5, system 3: full fine-tuning of an encoder transformer (RoBERTa-large).

RoBERTa is already pre-trained on huge amounts of English text. We add a small
classification layer on top and keep training, so that ALL ~355 million weights adapt to
tweet sentiment. That is what "full" fine-tuning means (LoRA in Step 6 trains far fewer).
"""
import time

# `common` must be imported BEFORE transformers/datasets: it sets HF_HOME so that model
# downloads go into the project folder instead of the (almost full) home disk.
from common import (INFER_BATCH_SIZE, compute_metrics, load_tweeteval, make_parser, predict,
                    save_results, score_own_set, set_seed, ROOT)

import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

MAX_LEN = 64  # tweets are short, so 64 tokens is enough


def main():
    parser = make_parser("Full fine-tuning of an encoder")
    parser.add_argument("--model", default="roberta-large")
    parser.add_argument("--epochs", type=float, default=3)
    args = parser.parse_args()
    set_seed(args.seed)
    data = load_tweeteval(args.max_train, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)

    def to_dataset(df):
        # No padding here: the collator pads each batch only up to its longest tweet.
        ds = Dataset.from_pandas(df[["text", "label"]])
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN),
                      batched=True, remove_columns=["text"])

    def validation_scores(eval_pred):
        logits, labels = eval_pred
        m = compute_metrics(labels, logits.argmax(axis=-1))
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{args.run_name}",
        learning_rate=1e-5,                 # small steps: we only want to nudge a pre-trained model
        num_train_epochs=args.epochs,       # 3 passes over the training tweets
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=0.06,                   # a float below 1 = share of all steps: ramp up over the first 6%
        weight_decay=0.01,
        bf16=True,                          # 16-bit maths: faster and uses less memory
        seed=args.seed,
        eval_strategy="epoch",              # validation scores are only for watching progress;
        save_strategy="no",                 # we keep the LAST epoch, no checkpoints on disk
        report_to="none",
        logging_steps=50,
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=to_dataset(data["train"]),
                      eval_dataset=to_dataset(data["validation"]),
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=validation_scores, processing_class=tokenizer)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    peak_train_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    model.eval()

    def scores(texts):
        batch = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True,
                          return_tensors="pt").to(model.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(**batch).logits.float().cpu().numpy()

    torch.cuda.reset_peak_memory_stats()
    preds, speed = predict(scores, data["test"]["text"].tolist())
    peak_infer_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    n_params = sum(p.numel() for p in model.parameters())
    cost = {"trainable_params": n_params, "total_params": n_params,  # full fine-tuning: all train
            "train_time_s": train_time, "peak_train_mem_gb": peak_train_gb,
            "peak_infer_mem_gb": peak_infer_gb, "throughput_tweets_per_s": speed,
            "infer_batch_size": INFER_BATCH_SIZE, "device": "cuda"}
    meta = save_results(args.run_name, data["test"], preds, cost=cost, args=vars(args),
                        results_root=args.results_root)
    m = meta["metrics"]["tweeteval"]
    print(f"{args.run_name}: accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
          f"macro-recall={m['macro_recall']:.4f}  (train {train_time:.0f}s, {speed:.0f} tweets/s)")
    score_own_set(args.run_name, scores, args.results_root)
    # keep the trained model so it can be scored again later without retraining (git-ignored)
    model.save_pretrained(ROOT / "checkpoints" / args.run_name)
    tokenizer.save_pretrained(ROOT / "checkpoints" / args.run_name)


if __name__ == "__main__":
    main()
