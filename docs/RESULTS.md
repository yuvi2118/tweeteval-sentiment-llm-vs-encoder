# Results

All numbers come from finished runs in `results/` (one run per system, seed 42). Test set: TweetEval
sentiment, official test split, 12,284 tweets (32.3% negative, 48.3% neutral, 19.3% positive).
The training split is 15.5% / 45.3% / 39.1%, so the class mix shifts between train and test.

## Main comparison

| System | Accuracy | Macro-F1 | Macro-recall | Own set macro-F1 |
|---|---|---|---|---|
| always "neutral" (floor) | 0.4833 | 0.2172 | 0.3333 | - |
| TF-IDF + Logistic Regression | 0.5891 | 0.5540 | 0.5609 | 0.8214 |
| BiLSTM + GloVe | 0.6462 | 0.6427 | 0.6502 | 0.9289 |
| Qwen2.5-7B zero-shot | 0.6847 | 0.6840 | 0.7087 | 0.9719 |
| Qwen2.5-7B few-shot (9 examples) | 0.6922 | 0.6730 | 0.6612 | 0.9889 |
| RoBERTa-large, full fine-tuning | 0.7125 | 0.7139 | 0.7381 | 0.9779 |
| BERTweet-large, full fine-tuning (extra) | 0.7227 | 0.7239 | 0.7423 | 0.9944 |
| **Qwen2.5-7B + LoRA** | **0.7407** | **0.7414** | **0.7538** | **1.0000** |
| Qwen2.5-7B + QLoRA, 4-bit (extra) | 0.7410 | 0.7417 | 0.7524 | 0.9944 |

Macro-recall is the official TweetEval metric. Figures: `results/figures/comparison.png`.

## Cost

| System | Trainable parameters | Total parameters | Training time (s) | Peak GPU memory, training / inference (GB) | Speed (tweets/s) |
|---|---|---|---|---|---|
| TF-IDF + LR | 285,819 | 285,819 | 23 (CPU) | - | 27,254 (CPU) |
| BiLSTM + GloVe | 2,649,723 | 2,649,723 | 46 | 0.4 / 0.1 | 6,225 |
| RoBERTa-large | 355,362,819 | 355,362,819 | 809 | 6.6 / 4.6 | 594 |
| BERTweet-large | 355,362,819 | 355,362,819 | 878 | 6.6 / 4.6 | 591 |
| Qwen 7B zero-shot | 0 | 7,615,616,512 | 0 | - / 14.8 | 48 |
| Qwen 7B few-shot | 0 | 7,615,616,512 | 0 | - / 18.2 | 6 |
| Qwen 7B + LoRA | 40,380,928 (0.57%) | 7,111,010,816 | 3,218 | 14.3 / 14.1 | 66 |
| Qwen 7B + QLoRA (4-bit base) | 40,380,928 | 7.11B in 4-bit storage (the counter shows 3,848,366,592 because two 4-bit weights are packed in one stored number) | 4,583 | 6.9 / 6.6 | 49 |

Speed is measured on the test set with the same inference batch size (32) for every system, after one
warm-up batch, each system in its own process. The GPUs were shared with other users' jobs, so the
timings and speeds are noisy and should be read as orders of magnitude. Figure:
`results/figures/cost_vs_score.png`. Full table: `results/summary.md`.

## Summary

**Does a LoRA-fine-tuned 7B LLM beat a fully fine-tuned encoder?** On this benchmark, yes. Qwen + LoRA
reaches macro-F1 0.7414 against 0.7139 for RoBERTa-large, a gap of 2.8 points. When the test tweets are
resampled (paired bootstrap, 2,000 resamples, `results/bootstrap_ci.csv`), the 95% interval of the gap is
2.0 to 3.5 points, so the gap is larger than the noise from the choice of test tweets. BERTweet-large, an
encoder pre-trained on tweets, is 1.0 point better than RoBERTa-large at the same cost (interval 0.4 to
1.6) and narrows the gap to LoRA to 1.8 points (interval 1.1 to 2.4).

**Is the extra cost worth it?** It depends on what matters. Compared with RoBERTa-large, LoRA needs
about 4 times the training time (3,218 s against 809 s), more than twice the GPU memory (14.3 GB against
6.6 GB) and predicts about 9 times slower (66 against 594 tweets per second). For roughly 3 points of
macro-F1 that can be a good trade when accuracy matters and volume is low, and a poor one for a
high-volume service. Only 0.57% of the LLM's weights were trained.

**QLoRA (4-bit base) matches LoRA.** Macro-F1 0.7417 against 0.7414 for 16-bit LoRA; the gap of 0.0003 is far inside the noise (interval -0.003 to +0.004). It needs less than half the GPU memory (6.9 GB against 14.3 GB), but trains about 1.4 times slower (4,583 s against 3,218 s) and predicts a bit slower (49 against 66 tweets per second). In this task, quantizing the frozen weights to 4 bits cost no accuracy.

**Prompting without training.** Zero-shot Qwen (macro-F1 0.6840) beats TF-IDF and the BiLSTM and is 3.0
points behind RoBERTa-large, with no training at all, but it predicts about 12 times slower than
RoBERTa. Giving it 9 examples did not help: accuracy rose slightly (0.6922) but macro-F1 fell by 1.1
points and macro-recall by 4.8 points, because the model moved many positive tweets to neutral (positive
recall 72% to 47%). This is one draw of 9 examples with one prompt, and few-shot results are known to depend
on which examples are chosen, so it shows that few-shot did not help here, not that it never helps.

**Where all the models fail.** Neutral is the hardest class for the neural systems (recall 57% to 70%).
720 of the 12,284 test tweets (5.9%) were wrong for all six main systems. Of 40 of these (random sample,
seed 42; `results/hard_examples.csv`), the causes assigned in `annotation/hard_examples_annotated.csv` are:
sarcasm or irony 9, implicit stance with no sentiment words 8, noisy text (very short, truncated,
hashtags or links only) 7, the boundary between neutral and sentiment 6, mixed sentiment 4, labels that look
wrong 3, negation 1, missing context 1, emoji with non-literal meaning 1. For 13 of the 40 the label looks
wrong or debatable, so part of the remaining error on this benchmark is label noise. These causes were
assigned by an LLM and not checked by a human; the table is a first reading, not a measurement.

## Per-class recall (TweetEval test)

| System | negative | neutral | positive |
|---|---|---|---|
| TF-IDF + LR | 31.7% | 76.7% | 59.9% |
| BiLSTM + GloVe | 66.1% | 63.2% | 65.8% |
| Qwen zero-shot | 83.2% | 57.1% | 72.3% |
| Qwen few-shot | 80.8% | 70.2% | 47.3% |
| RoBERTa-large | 81.2% | 61.7% | 78.6% |
| Qwen + LoRA | 81.6% | 68.1% | 76.5% |

Confusion matrices for all systems: `results/figures/confusion_matrices.png`.
TF-IDF sends 59% of the negative tweets to neutral, which fits the shift: negative is 15.5% of training
tweets but 32.3% of test tweets.

## Limitations (please read)

- **One training run per system, one seed (42).** Re-running the same code with the same seed gave
  identical results for all four retrained systems (`results/run_to_run.csv`), so the runs are
  reproducible, but we did not measure how much the scores change with different seeds. The bootstrap
  interval above only covers the choice of test tweets.
- **The own test set is easy and does not separate the systems.** `data/own_test.csv` has 180 short
  posts (60 per class) written and labeled with LLM assistance, not collected from real users. Scores
  are 0.82 to 1.00 macro-F1, much higher than on TweetEval, and Qwen + LoRA gets all 180 right, so the set
  is at its ceiling. Use the TweetEval test for the comparison.
- **The data-size ablation was not run.** The proposal planned a comparison with 1%, 10% and 100% of the
  training data. It was dropped for time. The scripts exist (`scripts/run_ablation.sh`) but have no results.
- **Prompting results depend on prompt wording.** One instruction, one set of 9 examples. The answer
  words were capitalized ("Positive") after a check on validation tweets showed that Qwen writes them that
  way; test data was not used to choose the prompt.
- **Timings come from shared GPUs** (see above).
- **TweetEval labels are noisy** (see the hard examples), which limits the best achievable score.
- No test data was used to choose hyperparameters, epochs or prompts. The BiLSTM epoch was chosen on the
  validation split; the transformers use the last epoch.

## What went wrong on the way

- The first Qwen + LoRA run was stopped at step 785 because the logged training loss looked stuck above
  chance. It was a misreading: with gradient accumulation of 4 the logged loss appears to be 4 times the
  real per-sample loss (our evidence is that the validation loss was about a quarter of it), so the run was
  probably fine. The second run, with a zero-initialized classification layer, followed the same loss
  plateau (logged about 2.5) over the steps where the two runs overlap, and trained to the end. The
  final numbers are from the second run.
- The first zero-shot run stopped at its safety check: only 3% of the probability was on the lowercase
  label words, because Qwen answers with a capital letter. After switching both prompts to the capitalized
  words, 97% (zero-shot) and 99.98% (few-shot) of the probability was on the label words.
