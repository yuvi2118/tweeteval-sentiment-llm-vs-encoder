"""Step 6, system 4: Qwen2.5-7B-Instruct fine-tuned with LoRA as a sentiment classifier.

LoRA idea: freeze the 7 billion original weights. Next to each big weight matrix W we add
two thin matrices A and B (rank r=16), and the layer computes  W x + (alpha/r) * B A x.
Only A and B are trained, so well under 1% of the numbers are updated and the memory for
gradients and optimizer state stays small.

The model is used as a *sequence classifier*: a new 3-way linear layer reads the hidden
state of the last token. There is no text generation. (Zero-shot prompting is Step 7.)

--qlora loads the frozen 7B weights in 4 bits instead of 16 (QLoRA): much less GPU memory.
"""
import time

# `common` must be imported BEFORE transformers/datasets: it sets HF_HOME so that model
# downloads go into the project folder
from common import (INFER_BATCH_SIZE, compute_metrics, load_tweeteval, make_parser, predict,
                    save_results, score_own_set, set_seed, ROOT)

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MAX_LEN = 96  # a bit longer than for RoBERTa: Qwen's tokenizer splits tweets into more pieces


def main():
    parser = make_parser("LoRA fine-tuning of an LLM for classification")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--qlora", action="store_true", help="load the frozen model in 4-bit")
    args = parser.parse_args()
    set_seed(args.seed)
    data = load_tweeteval(args.max_train, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if args.qlora:  # NF4 = a 4-bit number format designed for neural-network weights
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    # device_map puts the weights straight onto the (single visible) GPU, so they never
    # have to fit in CPU memory first
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=quant)
    model.config.pad_token_id = tokenizer.pad_token_id  # needed to find each tweet's last real token
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    # Start the new 3-way layer at zero. With the default random start, Qwen's large hidden
    # states gave huge, wrong, over-confident scores (loss 24 and stuck near 2.5 in our first
    # run). At zero all three classes start equally likely (loss ~1.10) and learning is stable.
    # Must happen before get_peft_model, which copies this layer.
    model.score.weight.data.zero_()

    lora = LoraConfig(
        task_type="SEQ_CLS",  # also makes the new 3-way classification layer trainable
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    def to_dataset(df):
        ds = Dataset.from_pandas(df[["text", "label"]])
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN),
                      batched=True, remove_columns=["text"])

    def validation_scores(eval_pred):
        logits, labels = eval_pred
        m = compute_metrics(labels, logits.argmax(axis=-1))
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{args.run_name}",
        learning_rate=1e-4,                  # LoRA can use a larger rate than full fine-tuning
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,       # 8 x 4 = effective batch of 32
        per_device_eval_batch_size=16,
        warmup_steps=0.03,                   # float below 1 = share of all steps
        bf16=True,
        gradient_checkpointing=True,         # recompute activations to save GPU memory
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        eval_strategy="epoch", save_strategy="no", report_to="none", logging_steps=20,
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
        with torch.no_grad():
            return model(**batch).logits.float().cpu().numpy()

    torch.cuda.reset_peak_memory_stats()
    preds, speed = predict(scores, data["test"]["text"].tolist())
    peak_infer_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    cost = {"trainable_params": trainable, "total_params": total, "train_time_s": train_time,
            "peak_train_mem_gb": peak_train_gb, "peak_infer_mem_gb": peak_infer_gb,
            "throughput_tweets_per_s": speed, "infer_batch_size": INFER_BATCH_SIZE,
            "device": "cuda", "qlora": args.qlora}
    meta = save_results(args.run_name, data["test"], preds, cost=cost, args=vars(args),
                        results_root=args.results_root)
    m = meta["metrics"]["tweeteval"]
    print(f"{args.run_name}: accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
          f"macro-recall={m['macro_recall']:.4f}  (train {train_time:.0f}s, {speed:.0f} tweets/s)")
    score_own_set(args.run_name, scores, args.results_root)
    model.save_pretrained(ROOT / "checkpoints" / args.run_name)  # only the small LoRA adapter + head


if __name__ == "__main__":
    main()
