"""Step 8: turn the saved results of all systems into tables, figures and the hard-examples file.

Run:  python src/make_report.py
It only READS results/<system>/ (written by the training scripts) and needs no GPU.

Outputs
  results/summary.csv, results/summary.md        main comparison + cost table
  results/figures/comparison.png                 bar chart of the scores
  results/figures/cost_vs_score.png             cost against macro-F1
  results/figures/confusion_matrices.png        one confusion matrix per system
  results/hard_examples.csv                     test tweets that (almost) every system got wrong
  results/ablation.csv, figures/data_size.png    macro-F1 with 1%, 10% and 100% of the training data
  results/run_to_run.csv                         the same system trained twice (run 1 vs run 2)
  results/bootstrap_ci.csv                       is the gap between two systems bigger than test-set noise?
"""
import json
import re

import matplotlib
matplotlib.use("Agg")  # draw to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import LABELS, ROOT, SEED, compute_metrics

# run folder name -> name shown in tables and plots (in the order we want them)
SYSTEMS = {
    "tfidf-lr": "TF-IDF + LR",
    "bilstm-glove": "BiLSTM + GloVe",
    "roberta-large": "RoBERTa-large (full FT)",
    "qwen7b-zeroshot": "Qwen 7B zero-shot",
    "qwen7b-fewshot": "Qwen 7B few-shot",
    "qwen7b-lora": "Qwen 7B + LoRA",
    "bertweet-large": "BERTweet-large (full FT)",   # optional extras: shown only if they ran
    "qwen7b-qlora": "Qwen 7B + QLoRA (4-bit)",
}
MAIN = list(SYSTEMS)[:6]  # the five systems of the proposal (Qwen has 3 variants); hard examples use these
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def load_all():
    """Read meta.json and the test predictions of every system that has finished."""
    metas, preds = {}, {}
    for run in SYSTEMS:
        folder = RESULTS / run
        if not (folder / "meta.json").exists():
            print(f"[skip] {run}: no results yet")
            continue
        metas[run] = json.loads((folder / "meta.json").read_text())
        preds[run] = pd.read_csv(folder / "preds_tweeteval.csv")
    return metas, preds


# ------------------------------------------------------------------ summary table
def build_summary(metas, preds):
    first = next(iter(preds.values()))
    rows = []
    # the "always neutral" guess: the floor every real system has to beat
    floor = compute_metrics(first["label"].values, np.ones(len(first), dtype=int))
    rows.append({"system": "always-neutral (floor)", "accuracy": floor["accuracy"],
                 "macro_f1": floor["macro_f1"], "macro_recall": floor["macro_recall"]})
    for run, meta in metas.items():
        m, c = meta["metrics"]["tweeteval"], meta["cost"]
        row = {"system": SYSTEMS[run], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
               "macro_recall": m["macro_recall"]}
        if "own" in meta["metrics"]:  # only when data/own_test.csv existed at run time
            o = meta["metrics"]["own"]
            row.update({"own_accuracy": o["accuracy"], "own_macro_f1": o["macro_f1"],
                        "own_macro_recall": o["macro_recall"]})
        row.update({"trainable_params": c["trainable_params"], "total_params": c["total_params"],
                    "train_time_s": c["train_time_s"], "peak_train_mem_gb": c.get("peak_train_mem_gb"),
                    "peak_infer_mem_gb": c.get("peak_infer_mem_gb"),
                    "tweets_per_s": c["throughput_tweets_per_s"]})
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(df):
    df.to_csv(RESULTS / "summary.csv", index=False)

    def fmt(col, value):
        if pd.isna(value):
            return "-"
        if col in ("trainable_params", "total_params"):
            return f"{value:,.0f}"
        if col in ("accuracy", "macro_f1", "macro_recall") or col.startswith("own_"):
            return f"{value:.4f}"
        return f"{value:.1f}"

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(row["system"] if c == "system" else fmt(c, row[c]) for c in cols) + " |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ figures
def plot_comparison(df):
    d = df.reset_index(drop=True)
    bars = [("macro-F1 (TweetEval test)", "macro_f1"), ("macro-recall (TweetEval test)", "macro_recall")]
    if "own_macro_f1" in d and d["own_macro_f1"].notna().any():  # only when the own set was scored
        bars.append(("macro-F1 (own set)", "own_macro_f1"))
    x = np.arange(len(d))
    width = 0.8 / len(bars)
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, (label, col) in enumerate(bars):
        ax.bar(x + (k - (len(bars) - 1) / 2) * width, d[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(d["system"], rotation=25, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Sentiment classification: all systems")
    fig.tight_layout()
    fig.savefig(FIGURES / "comparison.png", dpi=150)
    plt.close(fig)


def plot_cost_vs_score(df):
    d = df[df["system"] != "always-neutral (floor)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panels = [("tweets_per_s", "prediction speed (tweets per second, log scale)", "log"),
              ("train_time_s", "training time (seconds, 0 = no training)", "symlog")]
    for ax, (col, label, scale) in zip(axes, panels):
        ax.scatter(d[col], d["macro_f1"], s=70)
        for _, r in d.iterrows():
            ax.annotate(r["system"], (r[col], r["macro_f1"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
        ax.set_xscale(scale, **({"linthresh": 10} if scale == "symlog" else {}))
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1 on the test set")
    fig.suptitle("Cost against score (GPU numbers measured on shared GPUs, so they are noisy)")
    fig.tight_layout()
    fig.savefig(FIGURES / "cost_vs_score.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(metas):
    rows = -(-len(metas) // 3)  # 3 plots per row, round up
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4 * rows), squeeze=False)
    for ax in axes.flat[len(metas):]:
        ax.axis("off")  # hide unused boxes
    for ax, (run, meta) in zip(axes.flat, metas.items()):
        cm = np.array(meta["metrics"]["tweeteval"]["confusion_matrix"])
        share = cm / cm.sum(axis=1, keepdims=True)  # each row sums to 100%: that class's recall
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                        color="white" if share[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3))
        ax.set_xticklabels(LABELS)
        ax.set_yticks(range(3))
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(SYSTEMS[run])
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrices.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ hard examples
NEGATION = re.compile(r"\b(?:not|no|never|none|nobody|nothing|neither|nor|cannot)\b|n't\b", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def build_hard_examples(preds, n=40):
    """Test tweets that every (or almost every) system got wrong, with simple automatic flags.

    The flags are only a help for sorting by eye; the causes (sarcasm, mixed sentiment, ...)
    are for a person to decide after reading the tweets.
    """
    runs = [r for r in MAIN if r in preds]
    base = preds[runs[0]][["text", "label"]].copy()
    for run in runs:  # all systems must have been scored on exactly the same tweets in the same order
        assert (preds[run]["text"] == base["text"]).all(), f"{run}: test tweets differ in order"
    wrong = np.column_stack([preds[r]["pred"].values != base["label"].values for r in runs])
    base["n_wrong"] = wrong.sum(axis=1)

    # all systems wrong if there are enough such tweets, otherwise all but one, and so on
    for threshold in range(len(runs), 0, -1):
        pool = base[base["n_wrong"] >= threshold]
        if len(pool) >= n:
            break
    print(f"hard examples: {len(pool)} tweets wrong for at least {threshold} of {len(runs)} systems; "
          f"sampling {n} of them at random (seed {SEED})")
    hard = pool.sample(n=n, random_state=SEED)

    out = pd.DataFrame({"text": hard["text"], "true_label": hard["label"].map(dict(enumerate(LABELS)))})
    for run in runs:
        out[SYSTEMS[run]] = preds[run].loc[hard.index, "pred"].map(dict(enumerate(LABELS)))
    out["n_systems_wrong"] = hard["n_wrong"]
    out["has_negation"] = hard["text"].str.contains(NEGATION)
    out["has_emoji"] = hard["text"].str.contains(EMOJI)
    out["has_question_mark"] = hard["text"].str.contains(r"\?")
    out["has_hashtag"] = hard["text"].str.contains("#")
    out["very_short"] = hard["text"].str.split().str.len() <= 5
    out.to_csv(RESULTS / "hard_examples.csv", index=False)
    print("true labels of the hard examples:", out["true_label"].value_counts().to_dict())


# ------------------------------------------------------------------ data-size experiment
ABLATION_MODELS = {"RoBERTa-large (full FT)": "roberta-large", "Qwen 7B + LoRA": "qwen7b-lora"}
SIZES = [("1%", "_1pct", 456), ("10%", "_10pct", 4562), ("100%", "", 45615)]  # 100% = the main runs


def build_ablation():
    rows = []
    for label, base in ABLATION_MODELS.items():
        for pct, suffix, n in SIZES:
            folder = RESULTS / (base + suffix)
            if not (folder / "meta.json").exists():
                continue
            m = json.loads((folder / "meta.json").read_text())["metrics"]["tweeteval"]
            share = pd.read_csv(folder / "preds_tweeteval.csv")["pred"].value_counts(normalize=True)
            rows.append({"model": label, "train_share": pct, "train_tweets": n,
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "macro_recall": m["macro_recall"],
                         **{f"predicted_{LABELS[k]}": share.get(k, 0.0) for k in range(3)},
                         # a model that outputs one class for >95% of tweets has collapsed
                         "collapsed_to_one_class": bool(share.max() > 0.95),
                         "confusion_matrix": m["confusion_matrix"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    if df["train_share"].nunique() < 2:  # only the 100% runs exist: nothing to compare
        return
    df.to_csv(RESULTS / "ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group in df.groupby("model", sort=False):
        ax.plot(group["train_tweets"], group["macro_f1"], marker="o", label=label)
        for _, r in group.iterrows():
            ax.annotate(f"{r['macro_f1']:.3f}" + (" (collapsed)" if r["collapsed_to_one_class"] else ""),
                        (r["train_tweets"], r["macro_f1"]), textcoords="offset points", xytext=(5, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([n for _, _, n in SIZES])
    ax.set_xticklabels([f"{p}\n({n:,})" for p, _, n in SIZES])
    ax.set_xlabel("share of the training tweets used (number of tweets)")
    ax.set_ylabel("macro-F1 on the TweetEval test set")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Does the LLM help when there is little data?")
    fig.tight_layout()
    fig.savefig(FIGURES / "data_size.png", dpi=150)
    plt.close(fig)
    print("\ndata-size experiment:\n" + df.drop(columns="confusion_matrix").round(4).to_string(index=False))


# ------------------------------------------------------------------ run 1 vs run 2
def build_run_to_run():
    """The four systems trained in both runs: how much does the score move between runs?"""
    rows = []
    for run in ["tfidf-lr", "bilstm-glove", "roberta-large", "qwen7b-lora"]:
        old_path, new_path = ROOT / "results_run1" / run / "meta.json", RESULTS / run / "meta.json"
        if not (old_path.exists() and new_path.exists()):
            continue
        old = json.loads(old_path.read_text())["metrics"]["tweeteval"]
        new = json.loads(new_path.read_text())["metrics"]["tweeteval"]
        rows.append({"system": SYSTEMS[run], "macro_f1_run1": old["macro_f1"], "macro_f1_run2": new["macro_f1"],
                     "difference": new["macro_f1"] - old["macro_f1"],
                     "accuracy_run1": old["accuracy"], "accuracy_run2": new["accuracy"]})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / "run_to_run.csv", index=False)
        print("\nrun 1 vs run 2 (same code, same seed):\n" + df.round(4).to_string(index=False))


# ------------------------------------------------------------------ is a gap real or noise?
def macro_f1_fast(y_true, y_pred):
    """Macro-F1 from a 3x3 confusion table (much faster than calling sklearn 4000 times)."""
    cm = np.bincount(y_true * 3 + y_pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    return float(np.mean(2 * precision * recall / np.maximum(precision + recall, 1e-12)))


def build_bootstrap(preds, pairs, n_resamples=2000):
    """Paired bootstrap over the TEST TWEETS: resample the tweets with replacement and
    recompute the macro-F1 difference of two systems each time.

    This measures how much the gap moves because of WHICH test tweets we happened to have.
    It does NOT measure training randomness (different seeds), which we did not run.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for a, b in pairs:
        if a not in preds or b not in preds:
            continue
        y = preds[a]["label"].values
        pa, pb = preds[a]["pred"].values, preds[b]["pred"].values
        diffs = []
        for _ in range(n_resamples):
            idx = rng.integers(0, len(y), len(y))
            diffs.append(macro_f1_fast(y[idx], pa[idx]) - macro_f1_fast(y[idx], pb[idx]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"system_a": SYSTEMS[a], "system_b": SYSTEMS[b],
                     "macro_f1_a_minus_b": macro_f1_fast(y, pa) - macro_f1_fast(y, pb),
                     "ci95_low": lo, "ci95_high": hi, "gap_larger_than_noise": bool(lo > 0 or hi < 0)})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / "bootstrap_ci.csv", index=False)
        print("\npaired bootstrap of the macro-F1 gap (test tweets resampled):\n" + df.round(4).to_string(index=False))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    metas, preds = load_all()
    summary = build_summary(metas, preds)
    write_summary(summary)
    plot_comparison(summary)
    plot_cost_vs_score(summary)
    plot_confusion_matrices(metas)
    build_hard_examples(preds)
    build_ablation()
    build_run_to_run()
    build_bootstrap(preds, [("qwen7b-lora", "roberta-large"), ("qwen7b-lora", "bertweet-large"),
                            ("bertweet-large", "roberta-large"), ("qwen7b-zeroshot", "roberta-large"),
                            ("qwen7b-zeroshot", "qwen7b-fewshot"),
                            ("qwen7b-qlora", "qwen7b-lora"), ("qwen7b-qlora", "roberta-large")])
    print("\n" + (RESULTS / "summary.md").read_text())


if __name__ == "__main__":
    main()
