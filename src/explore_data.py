"""Step 1: look at the data before building any model.

Run:  python src/explore_data.py
It prints how big each split is, how the three classes are balanced, how long the
tweets are, and a few example tweets.
"""
from common import LABELS, load_tweeteval

data = load_tweeteval()

print("Split sizes:")
for split, df in data.items():
    print(f"  {split:10s} {len(df):6d} tweets")

print("\nClass balance (share of each label):")
for split, df in data.items():
    share = df["label"].value_counts(normalize=True).sort_index()
    print(f"  {split:10s}", "  ".join(f"{LABELS[i]}={share[i]:.1%}" for i in range(3)))

words = data["train"]["text"].str.split().str.len()
print(f"\nTweet length in words (train): mean={words.mean():.1f}, max={words.max()}")

print("\nThree example tweets per class (train):")
for i, name in enumerate(LABELS):
    print(f"  [{name}]")
    for text in data["train"][data["train"]["label"] == i]["text"].head(3):
        print("    -", text)
