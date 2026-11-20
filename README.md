# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/nv9mdfwfbw/badge.svg)](https://ask.thili.ai/b/nv9mdfwfbw)

> **AI assignment** for the **thiliAI course track project on Language Models from scratch — drill 4: fine-tune
> GPT-2 to tag legal contract clauses**, applied here to **customer feedback categorization**.
> By **Nikita Ravindra, USN 1MS24CI079**.

## What it does

A retail manager pastes a customer comment into a Streamlit app and sees which category it belongs to, with a
confidence bar across all categories. The classifier is a **fine-tuned GPT-2 (`distilgpt2`) that has been taught five
new special label tokens** such as `<staff_praise>`. At inference there is no text generation and no parsing:
one forward pass, take the logits at the last position, keep only the five category-token ids, softmax, argmax.
The answer can never be anything but a defined category, and it is deterministic.

The model is trained and run locally. **There is no external API call.**

## Required technique 1: extending the tokenizer and model (`app/tokens.py`)

Five new special tokens are added to the GPT-2 tokenizer, and the model's embedding matrix is resized to hold them.

```python
tokenizer.add_special_tokens({"additional_special_tokens": LABEL_TOKENS})   # 1. new ids at the end of the vocab
model.resize_token_embeddings(len(tokenizer))                               # 2. embedding matrix grows to match
emb[new_rows] = mean_of_existing_embeddings(label words) + tiny noise       # 3. mean-initialised, not random
```

| token | id |
|---|---:|
| `<product_issue>` | 50257 |
| `<staff_praise>` | 50258 |
| `<staff_complaint>` | 50259 |
| `<pricing_value>` | 50260 |
| `<store_experience>` | 50261 |

Vocab size / embedding rows: **50,257 before -> 50,262 after** (`distilgpt2`). Run `python -m app.cli milestone 2` to
print this live, including `embeddings_resized_to_match: true`.

**Mean initialisation and why it helps.** `resize_token_embeddings` adds rows that would otherwise be random noise.
Each new row is instead set to the mean embedding of the words in its label (`<staff_praise>` -> "staff praise");
`init="mean"` uses the mean of all existing embeddings. GPT-2 ties input and output embeddings, so this also gives
each label a sensible starting output logit. The model starts from distinct, meaningful points rather than
arbitrary ones, which matters with only ~270 training examples (see the observation under "How the special tokens
were added" below).

## Required technique 2: the Hugging Face `Trainer` (`app/finetune.py`)

Fine-tuning uses `transformers.Trainer` with `TrainingArguments` (`training_arguments()` and `build_trainer()` in
`app/finetune.py`; `train()` runs it). The loss is masked to the label token only. These are the arguments used for
the real run:

```python
TrainingArguments(
    num_train_epochs=8, per_device_train_batch_size=8, learning_rate=1e-4, weight_decay=0.01,
    warmup_steps=0.1, lr_scheduler_type="linear", logging_steps=5, save_strategy="no",
    eval_strategy="no", report_to="none", seed=42, remove_unused_columns=False,
    use_cpu=False,  # True when the device is cpu
)
trainer = Trainer(model=model, args=args, train_dataset=FeedbackDataset(...), data_collator=PadCollator(...))
trainer.train(); trainer.save_model(...)
```

Run it with `python -m app.cli train` (or `python -m app.cli milestone 3`); the summary (arguments used, epochs, final
loss, save location) is printed and written to `models/feedback-gpt2/training_info.json`.

## Categories (and why)

| token | meaning |
|---|---|
| `<product_issue>` | defective, damaged, wrong, spoiled or poorly made goods |
| `<staff_praise>` | positive remarks about the behaviour or helpfulness of staff |
| `<staff_complaint>` | rude, unhelpful, dishonest or untrained staff |
| `<pricing_value>` | prices, fees, discounts, billing, value for money |
| `<store_experience>` | the place itself: cleanliness, layout, crowds, queues, facilities, parking |

These are the five things a retail manager acts on, and they route to five different owners (merchandising,
store-floor management/training, pricing, facilities). Praise and complaint are separate categories on purpose
so one tells you what to reward and the other what to fix. `staff_praise` vs `staff_complaint` and `pricing_value`
vs `staff_complaint` (billing disputes) are the deliberately hard boundaries.

## Dataset

Every comment was **invented for this project** (no scraped or copied reviews). They live in
`dataset/raw/<category>.txt`, one per line, and vary in length, tone, politeness, typos and phrasing. Keyword
shortcuts were avoided on purpose (e.g. words like "broken" also appear in non-product comments) and each class
contains ambiguous comments that mention more than one topic (labelled by the main complaint/compliment).

**337 comments**, split 80/20 with a **stratified** split (seed 42): 269 train, 68 test.

| category | total | train | test |
|---|---:|---:|---:|
| product_issue | 72 | 58 | 14 |
| staff_praise | 68 | 54 | 14 |
| staff_complaint | 65 | 52 | 13 |
| pricing_value | 68 | 54 | 14 |
| store_experience | 64 | 51 | 13 |

The classes are close to balanced (19-21% each), so accuracy is meaningful; the majority-class baseline below is
~20%, as expected for five balanced classes.

## How the special tokens were added (milestone 2)

`app/tokens.py`:

1. `tokenizer.add_special_tokens({"additional_special_tokens": [...]})` adds `<product_issue>`, ... Each encodes
   to exactly one id.
2. `model.resize_token_embeddings(len(tokenizer))` — without it the new ids are out of range of the embedding
   matrix. (A test fails with an `IndexError` if this is skipped.)
3. The new embedding rows are **not left random**. The default `init="words"` sets each new row to the mean
   embedding of the words in the label (`<staff_praise>` -> "staff praise"); `init="mean"` sets them to the mean of
   all existing embeddings (plus 1e-3 noise). GPT-2 ties input and output embeddings, so this also initialises the
   output logit of each label. Starting from meaningful vectors converges much faster on ~270 examples.
   **Observed here:** with `init="mean"` (same hyperparameters, 8 epochs) the loss fell from 14 to ~1.63 and stayed
   there — about ln(5)=1.61, i.e. chance level, the model had only learned that the labels are equally likely.
   With `init="words"` the loss reached ~0. I did not run a controlled ablation over seeds, so treat this as one
   observation, not a measured effect size.

## Training (milestone 3)

- Template: `Comment: <text>\nCategory:<label_token><eos>`; identical at inference (`encode_prompt` is shared).
- **Loss is masked on everything except the label token**: `labels = -100` for the comment, separator, EOS and
  padding.
- Hugging Face `Trainer` (`app/finetune.py`), base `distilgpt2` (82M params; `--base gpt2` also works), run on
  Apple `mps`, about 70 seconds.

```python
TrainingArguments(
    num_train_epochs=8, per_device_train_batch_size=8, learning_rate=1e-4, weight_decay=0.01,
    warmup_steps=0.1, lr_scheduler_type="linear", logging_steps=5, save_strategy="no",
    eval_strategy="no", report_to="none", seed=42, remove_unused_columns=False,
    use_cpu=False,  # True when the device is cpu
)
```

Training loss (mean over 5-step windows, first/last logged): **10.48 at step 5 -> 0.00 by the end of epoch ~6**;
final reported training loss averaged over the whole run: 0.69 (it includes the high early loss).
Max sequence length 128 tokens; comments are truncated so the separator always survives.

> **Dependency note:** `accelerate` was **not** in the starter `requirements.txt`; I added it because the brief
> requires the HF `Trainer`, which raises `ImportError` without `accelerate`. It is the only dependency added.
> Evaluation metrics are computed by hand (no scikit-learn).

## Results on the held-out test split (68 comments)

Single run, seed 42, `distilgpt2`. Reproduce with `python -m app.cli evaluate` (also saved to
`results/evaluation.json`).

**Accuracy: 94.1%** (64/68) vs a **majority-class baseline of 20.6%** (always predicting `product_issue`).

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| product_issue | 1.000 | 1.000 | 1.000 | 14 |
| staff_praise | 0.923 | 0.857 | 0.889 | 14 |
| staff_complaint | 0.800 | 0.923 | 0.857 | 13 |
| pricing_value | 1.000 | 0.929 | 0.963 | 14 |
| store_experience | 1.000 | 1.000 | 1.000 | 13 |
| **macro avg** | | | 0.942 | 68 |

Confusion matrix (rows = true class, columns = predicted):

| true \ predicted | product_issue | staff_praise | staff_complaint | pricing_value | store_experience |
|---|---:|---:|---:|---:|---:|
| **product_issue** | 14 | 0 | 0 | 0 | 0 |
| **staff_praise** | 0 | 12 | 2 | 0 | 0 |
| **staff_complaint** | 0 | 1 | 12 | 0 | 0 |
| **pricing_value** | 0 | 0 | 1 | 13 | 0 |
| **store_experience** | 0 | 0 | 0 | 0 | 13 |

### Error analysis (all 4 mistakes)

| comment | true -> predicted | confidence |
|---|---|---:|
| "The young man in the shoe section went above and beyond for my mother." | staff_praise -> staff_complaint | 0.71 |
| "The pharmacist gave me a lecture instead of an answer. I felt judged." | staff_complaint -> staff_praise | 0.50 |
| "I was very confused about which laptop to buy and the assistant patiently explained everything without any pressure." | staff_praise -> staff_complaint | 0.99 |
| "I was double charged for one item and had to chase the refund for a week." | pricing_value -> staff_complaint | 1.00 |

What I think went wrong (these are my hypotheses, not tested): three of four errors are the *polarity* boundary
between the two staff classes. Neither praise example contains an obvious praise word ("went above and beyond"
is idiomatic; the laptop comment opens with negative-sounding words "confused", "pressure"), so the model likely
keyed on surface sentiment. The "lecture / judged" comment is genuinely close to a polite-advice reading and the
model was only 50% sure. The double-charge comment is a billing error about money that also describes a hassle
chasing a refund; the model chose the "service failure" reading with near-total confidence, which shows that the
confidence scores are **overconfident** and should not be read as calibrated probabilities.

### Example classifications (real output, written after training)

| comment | prediction | confidence |
|---|---|---:|
| "The kettle I bought leaked from day one and nobody at the counter would even look at me." | staff_complaint | 1.00 |
| "Nice shop, but way too pricey for what it is." | pricing_value | 0.99 |
| "Sunita was so patient with my dad. The queue was long but she made it fine." | staff_praise | 1.00 |
| "The self checkout is down again and the floor is filthy." | store_experience | 1.00 |
| "my order arived late and the box was crushed" | product_issue | 0.93 |
| "Great prices and the cashier was friendly" | pricing_value | 0.54 |
| "The staff were polite but the shoes fell apart in a week." | staff_praise | 0.77 |
| "asdf qwerty" | store_experience | 0.86 |

The mixed-topic comments (rows 1, 6, 7) show the single-label design: the model must pick one category, and for
the kettle and shoe comments it picked the *less* important one (arguably `product_issue`). Gibberish still gets
a confident label because the softmax is only over the five categories — there is no "none of the above".

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.cli dataset                       # validate raw comments, write dataset/*.csv|jsonl
python -m app.cli train                         # fine-tune (~1 min on Apple Silicon); saves models/feedback-gpt2
python -m app.cli evaluate                      # accuracy, per-class P/R/F1, confusion matrix, baseline
python -m app.cli classify "The cashier was rude to me"
python -m app.cli milestone N                   # run milestone N (1-5) and print its result, see below
streamlit run app/main.py                       # the UI (loads the model once; never trains)
pytest                                          # offline, ~1s, no downloads
```

`train` downloads `distilgpt2` from the Hugging Face Hub on first use. Device selection: `mps` when available,
else CPU; override with `--device cpu` or `CFA_DEVICE=cpu`. `python -m app.cli train --base gpt2` uses the full
124M model. If no trained model exists, the Streamlit app shows a message telling you to run the train command.

### Why the checkpoint is not committed

The fine-tuned model is ~316 MB (`distilgpt2`; ~500 MB for `gpt2`), too large to commit, so `models/` is in
`.gitignore`. Regenerate it with `python -m app.cli train` (the dataset and split are committed and seeded, so the
run is reproducible up to hardware nondeterminism on `mps`). `dataset/` (not `data/`, which the starter ignores) is
committed.

### Running each milestone

`app/core.py` has one function per milestone; each runs with no arguments and prints a real result.

| command | what it does / returns |
|---|---|
| `python -m app.cli milestone 1` | builds the dataset; stats with class balance (overall, train, test) |
| `python -m app.cli milestone 2` | downloads `distilgpt2`, adds the category tokens, resizes embeddings; token ids, vocab size before/after, resize confirmation |
| `python -m app.cli milestone 3` | fine-tunes with the `Trainer` (~1 min); TrainingArguments used, epochs, final loss, save path |
| `python -m app.cli milestone 4 ["comment"]` | label, confidence and the full score distribution (needs a trained model) |
| `python -m app.cli milestone 5 ["comment"]` | label, all scores, display fields and the launch command `streamlit run app/main.py` (needs a trained model) |

## Tests

`tests/test_core.py` has one test per milestone plus metric checks, all offline: a tiny `GPT2Config` model and a
stub word-level tokenizer (`tests/conftest.py`); nothing is downloaded and Streamlit is never launched.
They cover the template and label-only masking, single-id tokens within the embedding range with a decode round
trip and the resize, a 25-step `Trainer` run on the toy model reducing loss, a classifier that always returns a
defined category and is deterministic, metrics against a hand-checked confusion matrix, and the Streamlit logic
layer. `tests/test_smoke.py` is untouched.

## Project layout

```
app/
  dataset.py   milestone 1: categories, loading, stratified split, template + loss masking
  tokens.py    milestone 2: add category tokens, resize embeddings, initialise new rows
  finetune.py  milestone 3: TrainingArguments, Trainer, save
  classify.py  milestone 4: score category-token logits (no generate)
  evaluate.py  hand-written metrics: accuracy, P/R/F1, confusion matrix, majority baseline
  ui_logic.py  milestone 5: pure logic behind the page
  main.py      milestone 5: Streamlit page
  core.py      one function per milestone, orchestrating the modules above
  cli.py       dataset | train | classify | evaluate | milestone N
  device.py    mps / cpu selection
  llm.py, storage.py   unused starter files (see below)
dataset/raw/   hand-written comments, one file per category; dataset/*.jsonl|csv are generated
results/evaluation.json   last evaluation run
tests/         offline tests
```

## Unused parts of the starter

`app/llm.py` and the LLM settings in `.env.example` belong to the starter template and are **not used**: the model
is trained and run locally with no external API call. Never commit `.env`; it is gitignored.

## Limitations

- **Small, self-written data.** 337 invented comments; the 68-comment test set means one mistake moves accuracy by
  1.5 points, so 94% carries wide uncertainty (roughly +/-6 points at 95%). One seed, one run, no cross-validation.
- The comments were written by one person, so they share style; real customer text (other languages, code-mixing,
  very long reviews, sarcasm) will be harder. Not evaluated on any real reviews.
- **Single label.** Mixed comments are forced into one category; there is no "other/unclear" class, so
  off-topic or gibberish input still gets a confident label.
- Confidence is **overconfident** (softmax of a fine-tuned LM with a near-zero training loss); do not treat it as
  calibrated.
- `distilgpt2` was used for speed; the full `gpt2` option exists but was not evaluated.
- Training is not bit-reproducible on `mps`; expect results to vary by a few comments across runs.
