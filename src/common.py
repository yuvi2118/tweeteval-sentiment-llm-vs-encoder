"""Shared helpers used by every system. We add to this file step by step.

Step 1: load the TweetEval sentiment dataset.
Step 2: compute metrics and save results in one standard format.
Step 3: shared command-line options and a speed (throughput) measurement.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Keep Hugging Face downloads inside the project folder (the home disk is almost full).
# This must run before `datasets` is imported.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("GENSIM_DATA_DIR", str(ROOT / ".gensim_data"))  # GloVe vectors (Step 4)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

LABELS = ["negative", "neutral", "positive"]  # label ids 0, 1, 2
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
SEED = 42
# Same inference batch size for every system, so the speed numbers are comparable.
INFER_BATCH_SIZE = 32


def load_tweeteval(max_train=None, seed=SEED):
    """Return {'train', 'validation', 'test'}: three tables with columns text, label.

    We use the official splits and never re-split them. `max_train` takes a random
    subset of the training tweets (for quick tests and the data-size experiment).
    """
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    data = {split: ds[split].to_pandas()[["text", "label"]]
            for split in ["train", "validation", "test"]}
    if max_train:
        data["train"] = data["train"].sample(n=min(max_train, len(data["train"])),
                                             random_state=seed).reset_index(drop=True)
    return data


def compute_metrics(y_true, y_pred):
    """Score predictions. Every system is scored by this one function.

    - accuracy:     share of tweets with the right label (misleading when classes are unbalanced)
    - macro_f1:     F1 per class, then the plain average over the 3 classes
    - macro_recall: recall per class, then the plain average (the official TweetEval metric)
    - confusion_matrix: rows = true label, columns = predicted label
    """
    kwargs = {"labels": [0, 1, 2], "average": "macro", "zero_division": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, **kwargs)),
        "macro_recall": float(recall_score(y_true, y_pred, **kwargs)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "n": int(len(y_true)),
    }


def save_results(run_name, df, preds, name="tweeteval", cost=None, args=None,
                 results_root="results"):
    """Save one system's predictions and scores in a standard layout.

    results/<run_name>/preds_<name>.csv   columns text, label, pred
    results/<run_name>/meta.json          metrics (+ cost and settings when given)
    `name` is the test set: "tweeteval" now, "own" later. Calling this twice for the same
    run adds the second test set to the same meta.json.
    """
    out_dir = ROOT / results_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": df["text"], "label": df["label"], "pred": preds}).to_csv(
        out_dir / f"preds_{name}.csv", index=False)

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"run_name": run_name,
                                                                         "metrics": {}}
    meta["metrics"][name] = compute_metrics(df["label"], preds)
    if cost:
        meta["cost"] = cost
    if args:
        meta["args"] = args
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def set_seed(seed=SEED):
    """Fix the random numbers so a run can be repeated and gives the same result."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_parser(description):
    """Command-line options that every training script shares."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--run_name", required=True, help="results go to results/<run_name>/")
    p.add_argument("--max_train", type=int, default=None,
                   help="use only N random training tweets (quick tests, data-size experiment)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--results_root", default="results",
                   help="use results_smoke for quick tests so the real results stay clean")
    return p


def _sync_gpu():
    """GPU work is asynchronous: wait for it to finish before reading the clock."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def predict(scores_fn, texts, batch_size=INFER_BATCH_SIZE):
    """Predict label ids batch by batch. Returns (predictions, tweets_per_second).

    scores_fn(list_of_texts) must return a numpy array of shape [len(list), 3].
    The first batch is a warm-up and is not timed; loading the model is not timed either.
    """
    preds, seconds, tweets = [], 0.0, 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        _sync_gpu()
        start = time.perf_counter()
        scores = scores_fn(batch)
        _sync_gpu()
        if i > 0:
            seconds += time.perf_counter() - start
            tweets += len(batch)
        preds.extend(scores.argmax(axis=1).tolist())
    return np.array(preds), tweets / seconds


def load_own_set():
    """Load data/own_test.csv (columns text,label). Returns None when the file is missing or empty."""
    path = ROOT / "data" / "own_test.csv"
    if not path.exists():
        print("[own set] data/own_test.csv not found -> own-set evaluation skipped")
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        print("[own set] data/own_test.csv is empty -> own-set evaluation skipped")
        return None
    df["label"] = df["label"].str.strip().str.lower().map(LABEL_TO_ID)
    if df["label"].isna().any():
        raise ValueError("own_test.csv: every label must be negative, neutral or positive")
    return df[["text", "label"]]


def score_own_set(run_name, scores_fn, results_root="results"):
    """Score one system on our own test set and add it to that system's results.

    Every script calls this after the TweetEval test, with the same scores_fn.
    """
    own = load_own_set()
    if own is None:
        return
    preds, _ = predict(scores_fn, own["text"].tolist())
    meta = save_results(run_name, own, preds, name="own", results_root=results_root)
    m = meta["metrics"]["own"]
    print(f"{run_name}: OWN SET accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
          f"macro-recall={m['macro_recall']:.4f}  (n={m['n']})")
