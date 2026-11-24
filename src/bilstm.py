"""Step 4, system 2: BiLSTM with GloVe word vectors.

Idea: every word becomes a 100-number vector (GloVe, pre-trained on 2 billion tweets).
A 2-layer bidirectional LSTM reads the tweet left-to-right and right-to-left, giving one
vector per word. Mean-pooling and max-pooling over the words give one vector per tweet,
and a linear layer turns it into 3 class scores.
"""
import copy
import re
import time
from collections import Counter

import torch
import torch.nn as nn

from common import (INFER_BATCH_SIZE, compute_metrics, load_tweeteval, make_parser, predict,
                    save_results, score_own_set, set_seed)

MAX_LEN = 64      # tweets are short (max 35 words); anything longer is cut
PAD, UNK = 0, 1   # word id 0 = padding, 1 = unknown word


def tokenize(text):
    """Lowercase, then split into words and single punctuation / emoji characters."""
    return re.findall(r"[a-z0-9_@']+|[^\sa-z0-9_@']", text.lower())


def build_vocab(texts, min_count=2):
    """Word -> id, for words seen at least `min_count` times in the TRAINING tweets only."""
    counts = Counter(tok for text in texts for tok in tokenize(text))
    words = [w for w, c in counts.items() if c >= min_count]
    return {w: i + 2 for i, w in enumerate(words)}  # ids 0 and 1 are PAD and UNK


def encode(texts, vocab):
    """Turn tweets into a padded table of word ids, shape [number of tweets, MAX_LEN]."""
    ids = torch.zeros(len(texts), MAX_LEN, dtype=torch.long)
    for row, text in enumerate(texts):
        toks = [vocab.get(t, UNK) for t in tokenize(text)][:MAX_LEN]
        ids[row, :len(toks)] = torch.tensor(toks, dtype=torch.long)
    return ids


def glove_matrix(vocab, dim=100):
    """Start-up values for the embedding layer: GloVe vectors for the words we know.

    Words GloVe does not have keep small random values. If the GloVe download fails we
    fall back to all-random vectors and say so (it is recorded in meta.json).
    """
    matrix = torch.randn(len(vocab) + 2, dim) * 0.1
    matrix[PAD] = 0
    try:
        import gensim.downloader as api
        glove = api.load("glove-twitter-100")
        found = 0
        for word, i in vocab.items():
            if word in glove:
                matrix[i] = torch.tensor(glove[word])
                found += 1
        print(f"GloVe: {found} of {len(vocab)} vocabulary words found")
        return matrix, True
    except Exception as err:
        print(f"WARNING: GloVe download failed ({err}). Using RANDOM embeddings instead.")
        return matrix, False


class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden=128):
        super().__init__()
        # the GloVe vectors are only a starting point; training may adjust them
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden, num_layers=2,
                            bidirectional=True, batch_first=True)
        self.out = nn.Linear(4 * hidden, 3)  # mean-pool (2*hidden) + max-pool (2*hidden) -> 3 classes

    def forward(self, ids):
        real = (ids != PAD).unsqueeze(-1)                    # True where there is a real word
        states, _ = self.lstm(self.embed(ids))               # one vector per word
        mean = (states * real).sum(1) / real.sum(1).clamp(min=1)
        biggest = states.masked_fill(~real, -1e9).max(1).values  # padding must not win the max
        return self.out(torch.cat([mean, biggest], dim=1))


def main():
    parser = make_parser("BiLSTM + GloVe")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_tweeteval(args.max_train, args.seed)

    vocab = build_vocab(data["train"]["text"])
    matrix, glove_ok = glove_matrix(vocab)
    model = BiLSTM(matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x_train = encode(data["train"]["text"].tolist(), vocab)
    y_train = torch.tensor(data["train"]["label"].values)
    x_val = encode(data["validation"]["text"].tolist(), vocab)

    def validation_macro_f1():
        model.eval()
        with torch.no_grad():
            preds = [model(x_val[i:i + 256].to(device)).argmax(1).cpu() for i in range(0, len(x_val), 256)]
        return compute_metrics(data["validation"]["label"].values, torch.cat(preds).numpy())["macro_f1"]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    best_f1, best_epoch, best_weights = -1, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        for i in range(0, len(order), 64):               # batch size 64
            batch = order[i:i + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch].to(device)), y_train[batch].to(device))
            loss.backward()
            optimizer.step()
        f1 = validation_macro_f1()
        print(f"epoch {epoch}: validation macro-F1 = {f1:.4f}")
        if f1 > best_f1:  # keep the best epoch, chosen on validation (never on test)
            best_f1, best_epoch, best_weights = f1, epoch, copy.deepcopy(model.state_dict())
    train_time = time.time() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else None
    model.load_state_dict(best_weights)
    model.eval()

    def scores(texts):
        with torch.no_grad():
            return model(encode(texts, vocab).to(device)).cpu().numpy()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    preds, speed = predict(scores, data["test"]["text"].tolist())

    n_params = sum(p.numel() for p in model.parameters())
    cost = {"trainable_params": n_params, "total_params": n_params,
            "train_time_s": train_time, "peak_train_mem_gb": peak_gb,
            "peak_infer_mem_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else None,
            "throughput_tweets_per_s": speed, "infer_batch_size": INFER_BATCH_SIZE,
            "device": device, "best_epoch": best_epoch, "glove_used": glove_ok}
    meta = save_results(args.run_name, data["test"], preds, cost=cost, args=vars(args),
                        results_root=args.results_root)
    m = meta["metrics"]["tweeteval"]
    print(f"{args.run_name}: accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
          f"macro-recall={m['macro_recall']:.4f}  (best epoch {best_epoch}, train {train_time:.0f}s)")
    score_own_set(args.run_name, scores, args.results_root)


if __name__ == "__main__":
    main()
