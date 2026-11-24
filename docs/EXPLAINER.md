# Explainer: how every part works, in plain language

Use this to defend the project. Numbers are the real ones from `docs/RESULTS.md`.

## The task and the data

Tweet sentiment classification: each tweet is **negative, neutral or positive**. We use TweetEval
(sentiment). It comes with fixed splits: 45,615 training tweets, 2,000 validation, 12,284 test. We never
re-split them, so our scores can be compared with other work. User mentions are already replaced by
`@user` and links by `http`; we add no cleaning for the transformers and only lowercase for TF-IDF and the BiLSTM.

Two facts about the data matter for everything else:
- **Neutral is the biggest class** (about 45-48%). Always answering "neutral" gives 48% accuracy and
  teaches nothing. That is our floor.
- **The test set has a different class mix than train** (negative 32% in test, 15.5% in train; positive 19%
  in test, 39% in train). A model that learned "negative is rare" gets punished on test.

## The metrics

- **Accuracy:** share of tweets with the correct label. Misleading when classes are unbalanced.
- **Macro-F1:** compute F1 (balance of precision and recall) for each class, then average the three. Every
  class counts the same, however small.
- **Macro-recall:** compute recall (share of the true tweets of that class that we found) for each class,
  then average. This is the official TweetEval metric.
- **Confusion matrix:** rows are the true label, columns the predicted label. The diagonal is correct.
  Each row shows where one true class goes wrong.

Always answering "neutral" gets macro-recall exactly 1/3 (finds 100% of neutral, 0% of the others).

## The systems

**1. TF-IDF + Logistic Regression.** Each tweet becomes a long vector: one number per word or word pair
(1-2 grams), larger when the word is frequent in the tweet but rare overall (TF-IDF). A logistic regression
learns one weight per feature per class and picks the class with the highest score. 285,819 weights, trains
in 23 seconds on a CPU. It cannot see word order beyond pairs, which is why it misses negation and sarcasm.

**2. BiLSTM with GloVe.** Each word becomes a 100-number vector from GloVe (pre-trained on 2 billion
tweets). A 2-layer bidirectional LSTM reads the tweet left-to-right and right-to-left and builds a
representation for each word. We take the average and the maximum over all words (mean + max pooling) and a
linear layer gives the 3 scores. It trains for 5 epochs and we keep the best epoch on the validation split.
2.6 million weights.

**3. RoBERTa-large, full fine-tuning.** RoBERTa is a transformer encoder pre-trained on a lot of English
text (a better-trained BERT). We add a small classification layer and keep training, so **all 355 million
weights** adapt to sentiment. Settings: learning rate 1e-5 (small, to nudge not destroy what it learned),
3 epochs, batch 32, 6% warm-up, weight decay 0.01, bf16. Dropout 0.1 is active in training and off at test time.

**4. Qwen2.5-7B + LoRA.** LoRA freezes the 7 billion original weights. Next to each big weight matrix W it
adds two thin matrices A and B (rank r=16) and computes W·x + (alpha/r)·B·A·x. Only A and B and the new
classifier layer are trained: **40 million weights, 0.57%**. That saves memory for gradients and optimizer
state. The 7B model is used as a classifier (a 3-way linear layer on the last token), not for generation.
Settings: r=16, alpha=32, dropout 0.05, all attention and MLP projections, learning rate 1e-4, 1 epoch,
effective batch 32 (8 x 4 gradient accumulation).

**QLoRA** (extra): the same, but the frozen 7B weights are stored in 4 bits (NF4) and only dequantized
when used. Much less memory, a bit slower.

**5a. Zero-shot prompting.** No training. We ask Qwen (chat template) to classify the tweet with one word.
Instead of letting it write text, we read its probability for the first token of "Negative", "Neutral",
"Positive" and take the highest. Nothing to parse, and no invalid answers. **5b. Few-shot** adds 9 solved
examples (3 per class from the training data, seed 42) to the prompt.

## What each experiment shows (the findings)

- Better models cost more, and the gains shrink: from TF-IDF (0.554 macro-F1) to BiLSTM (0.643) to RoBERTa
  (0.714) to LoRA (0.741).
- LoRA beats full fine-tuning of RoBERTa-large by 2.8 macro-F1 points (interval 2.0 to 3.5) at about 4x the
  training time and 9x slower prediction. BERTweet-large (pre-trained on tweets) closes some of the gap.
- QLoRA (4-bit frozen weights) matches 16-bit LoRA (macro-F1 0.742 vs 0.741) with 6.9 GB instead of 14.3 GB of GPU memory, but trains about 1.4x slower.
- Prompting alone gets surprisingly far (0.684) but few-shot with 9 examples did not help (0.673).
- All systems fail on sarcasm, implicit opinions and noisy text, and about a third of the hard tweets
  have labels that look wrong.

## 20 likely questions, with short answers

1. **Why this dataset?** It is a standard benchmark with official splits, so results are comparable, and the
   tweets are short and noisy, which is a real challenge.
2. **Why not re-split or tune on the test set?** Re-splitting breaks comparability; tuning on test leaks
   information and inflates scores. We chose settings from the proposal, used validation only for the BiLSTM
   epoch and the prompt check, and looked at test results once.
3. **Why macro-F1 and macro-recall, not just accuracy?** Classes are unbalanced and the test mix differs
   from train. Always answering neutral gives 48% accuracy but macro-recall 33%.
4. **Why does macro-recall equal 1/3 for the "always neutral" guess?** Recall per class is 0%, 100%, 0%,
   average 33.3%.
5. **What is full fine-tuning and what is LoRA?** Full: all weights are updated (355M for RoBERTa). LoRA:
   the original weights are frozen and a small rank-16 add-on is trained (40M of 7.1B, 0.57%).
6. **Why does LoRA save memory?** Gradients and optimizer state are only kept for the small add-on, not for 7
   billion weights.
7. **What are rank and alpha?** Rank r is the width of the thin matrices (capacity of the add-on); alpha scales
   the add-on's output by alpha/r (here 32/16 = 2).
8. **What does QLoRA change?** The frozen base weights are stored in 4 bits (NF4), cutting memory further at some
   cost in speed. The trained add-on stays in higher precision.
9. **How does zero-shot scoring work?** We take the model's probabilities for the first token of the three
   label words right after the assistant tag, and pick the highest. We checked on 20 validation tweets that the
   three tokens differ, that they get about 97% of the probability, and that the pick matches what the model
   would write.
10. **Why did you have to change the zero-shot prompt?** The first check showed only 3% of the probability on
    lowercase "negative/neutral/positive", because Qwen writes a capital first letter. The capitalized words
    got 97%. We changed both prompts after looking at validation tweets only.
11. **Why did few-shot not beat zero-shot?** With the 9 examples the model moved many positive tweets to
    neutral (positive recall 72% to 47%). We used one draw of 9 examples; a different draw could change the
    result, so we cannot say few-shot is generally worse.
12. **Is the LoRA win real or noise?** The paired bootstrap over test tweets gives a 95% interval of 2.0 to 3.5
    macro-F1 points, so it is not due to which test tweets we had. We did not run several seeds, so training
    randomness is not covered.
13. **Why do the same-seed runs give identical numbers?** Seed 42 fixes initialization and data order, and
    the computation was deterministic here. That shows reproducibility, not stability across seeds.
14. **Why is the BiLSTM epoch chosen on validation and not on test?** Choosing on test would be leakage. The
    best epoch was epoch 2; later epochs overfit.
15. **Why is the class shift a problem?** Test has twice the share of negative tweets that train has, so a
    model that learned "negative is rare" under-predicts it. TF-IDF sent 59% of negative tweets to neutral.
16. **Why are the own-set scores so high?** The 180 posts were written and labeled with LLM assistance, with
    clear wording, so they are much easier than real tweets. Qwen + LoRA gets all 180 right, so the set is at its
    ceiling and cannot rank the strong systems. We compare systems on the TweetEval test.
17. **Is the extra cost of the LLM worth it?** About 3 macro-F1 points for 4x the training time, 2x the memory
    and 9x slower prediction compared with RoBERTa-large. Worth it when accuracy matters and volume is low;
    not for a high-volume service.
18. **Where do the models fail?** On tweets where no sentiment word appears (implicit opinion, sarcasm,
    rhetorical questions), on noisy or truncated text, on the neutral/sentiment boundary, and on tweets
    whose label itself looks wrong. 720 test tweets were wrong for all six systems.
19. **What would you do next?** Run several seeds, run the data-size experiment, try several few-shot draws,
    and collect real tweets with human labels for the own set.
20. **What went wrong in the project?** A misread training loss (logged loss is multiplied by the gradient
    accumulation of 4) led us to stop a healthy run, and the first zero-shot prompt scored the wrong word
    case until the safety check caught it. Both are described in `docs/RESULTS.md`.
