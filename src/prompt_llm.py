"""Step 7, systems 5a and 5b: Qwen2.5-7B-Instruct with prompting only (no training).

We ask the model "is this tweet negative, neutral or positive?" and look at which of the
three label words it considers most likely as the FIRST word of its answer. We do not let
it write free text, so there is nothing to parse and no invalid answers.

    --n_shots 0   zero-shot: only the instruction
    --n_shots 9   few-shot:  the instruction plus 9 solved examples (3 per class) from train
"""
import time

# `common` must be imported BEFORE transformers/datasets: it sets HF_HOME so that model
# downloads go into the project folder instead of the (almost full) home disk.
from common import (INFER_BATCH_SIZE, LABELS, load_tweeteval, make_parser, predict, save_results,
                    score_own_set, set_seed)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are a sentiment classifier for tweets."
# Qwen answers with a capital letter ("Positive"), so these are the words we score. With the
# lowercase words only 3% of its probability landed on the labels (checked on validation tweets).
ANSWER_WORDS = [word.capitalize() for word in LABELS]
INSTRUCTION = ("Classify the sentiment of the following tweet as negative, neutral or positive. "
               "Answer with exactly one word.\n\nTweet: ")


def pick_examples(train, n_shots, seed):
    """n_shots examples from train, n_shots/3 per class, in shuffled order (fixed seed)."""
    per_class = n_shots // 3
    examples = train.groupby("label", group_keys=False).sample(per_class, random_state=seed)
    return examples.sample(frac=1, random_state=seed)  # shuffle so the classes are mixed


def build_messages(tweet, examples):
    """Chat messages: system text, then each solved example as a user/assistant pair, then the tweet."""
    messages = [{"role": "system", "content": SYSTEM}]
    if examples is not None:  # zero-shot has no examples
        for text, label in zip(examples["text"], examples["label"]):
            messages.append({"role": "user", "content": INSTRUCTION + text})
            messages.append({"role": "assistant", "content": ANSWER_WORDS[label]})
    messages.append({"role": "user", "content": INSTRUCTION + tweet})
    return messages


def main():
    parser = make_parser("Prompting an LLM (zero-shot / few-shot)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_shots", type=int, default=0, choices=[0, 9])
    parser.add_argument("--min_mass", type=float, default=0.8,
                        help="stop if the 3 label words get less than this share of the probability")
    parser.add_argument("--own_only", action="store_true",
                        help="score only our own test set and add it to existing results "
                             "(prompting is deterministic, so the TweetEval result stays valid)")
    args = parser.parse_args()
    set_seed(args.seed)
    data = load_tweeteval(args.max_train, args.seed)
    examples = pick_examples(data["train"], args.n_shots, args.seed) if args.n_shots else None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # so the LAST position of every row is the end of its prompt
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    # With the chat template the answer starts right after "<|im_start|>assistant\n", so the
    # answer words are used WITHOUT a leading space. We score only their first token.
    label_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ANSWER_WORDS]
    assert len(set(label_ids)) == 3, f"label first-tokens are not all different: {label_ids}"
    print("first token of each answer word:", dict(zip(ANSWER_WORDS, label_ids)))

    def encode(texts):
        prompts = [tokenizer.apply_chat_template(build_messages(t, examples), tokenize=False,
                                                 add_generation_prompt=True) for t in texts]
        return tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    def last_logits(texts):
        """Logits (over the whole vocabulary) for the first answer token, shape [batch, vocab]."""
        with torch.no_grad():
            return model(**encode(texts), logits_to_keep=1).logits[:, -1, :].float()

    def scores(texts):
        return last_logits(texts)[:, label_ids].cpu().numpy()  # keep only the 3 label words

    # ---- check the label-word scoring BEFORE the full run (uses validation tweets, no labels)
    check_texts = data["validation"]["text"].head(20).tolist()
    probs = torch.softmax(last_logits(check_texts), dim=-1)
    mass = probs[:, label_ids].sum(dim=1)                       # share of probability on the 3 words
    best_of_three = [label_ids[i] for i in probs[:, label_ids].argmax(dim=1).tolist()]
    with torch.no_grad():
        generated = model.generate(**encode(check_texts), max_new_tokens=1, do_sample=False)
    greedy = generated[:, -1].tolist()                          # what the model really writes first
    check = {"mean_probability_mass_on_labels": float(mass.mean()),
             "min_probability_mass_on_labels": float(mass.min()),
             "argmax_matches_greedy_generation": f"{sum(a == b for a, b in zip(best_of_three, greedy))}/20",
             "greedy_token_is_a_label": f"{sum(g in label_ids for g in greedy)}/20"}
    print("label-token check:", check)
    if check["mean_probability_mass_on_labels"] < args.min_mass:
        raise SystemExit(f"Only {check['mean_probability_mass_on_labels']:.2f} of the probability is on "
                         f"the three label words (need {args.min_mass}). Fix the prompt first.")

    if not args.own_only:
        torch.cuda.reset_peak_memory_stats()
        preds, speed = predict(scores, data["test"]["text"].tolist())
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

        total = sum(p.numel() for p in model.parameters())
        cost = {"trainable_params": 0, "total_params": total, "train_time_s": 0.0,
                "peak_train_mem_gb": None, "peak_infer_mem_gb": peak_gb,
                "throughput_tweets_per_s": speed, "infer_batch_size": INFER_BATCH_SIZE,
                "device": "cuda", "label_token_check": check}
        meta = save_results(args.run_name, data["test"], preds, cost=cost, args=vars(args),
                            results_root=args.results_root)
        m = meta["metrics"]["tweeteval"]
        print(f"{args.run_name}: accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
              f"macro-recall={m['macro_recall']:.4f}  ({speed:.0f} tweets/s)")

    score_own_set(args.run_name, scores, args.results_root)


if __name__ == "__main__":
    main()
