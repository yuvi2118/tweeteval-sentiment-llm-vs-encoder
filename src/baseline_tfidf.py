"""Step 3, system 1: TF-IDF + Logistic Regression.

The simplest baseline. Every tweet becomes a vector of word / word-pair weights (TF-IDF),
and a linear classifier learns which weights point to negative, neutral or positive.
No neural network, runs on the CPU.
"""
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import INFER_BATCH_SIZE, load_tweeteval, make_parser, predict, save_results, score_own_set


def main():
    args = make_parser("TF-IDF + Logistic Regression").parse_args()
    data = load_tweeteval(args.max_train, args.seed)

    # word 1-2 grams ("not good" is one feature); min_df=2 drops words seen only once;
    # sublinear_tf uses 1+log(count) so a word repeated many times does not dominate.
    # Lowercasing is the only cleaning, as for every non-transformer system.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    classifier = LogisticRegression(max_iter=1000)

    start = time.time()
    x_train = vectorizer.fit_transform(data["train"]["text"])
    classifier.fit(x_train, data["train"]["label"])
    train_time = time.time() - start

    def scores(texts):
        return classifier.decision_function(vectorizer.transform(texts))  # shape [batch, 3]

    preds, speed = predict(scores, data["test"]["text"].tolist())

    # a logistic regression has one weight per feature per class, plus one bias per class
    n_params = int(classifier.coef_.size + classifier.intercept_.size)
    cost = {"trainable_params": n_params, "total_params": n_params,
            "train_time_s": train_time, "peak_train_mem_gb": None,
            "throughput_tweets_per_s": speed, "infer_batch_size": INFER_BATCH_SIZE,
            "device": "cpu"}
    meta = save_results(args.run_name, data["test"], preds, cost=cost, args=vars(args),
                        results_root=args.results_root)
    m = meta["metrics"]["tweeteval"]
    print(f"{args.run_name}: accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
          f"macro-recall={m['macro_recall']:.4f}  (train {train_time:.0f}s, {speed:.0f} tweets/s)")
    score_own_set(args.run_name, scores, args.results_root)


if __name__ == "__main__":
    main()
