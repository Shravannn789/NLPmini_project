# Fake News Detection

An NLP web application that classifies news as **REAL** or **FAKE**, served through
a Flask web interface with an explainability view showing which words drove each
decision. It ships **two models** you can switch between in the UI:

- **Article model** — TF-IDF + Logistic Regression over full news articles (ISOT),
  ~97% accuracy.
- **Claim model** — a short fact-checking-statement classifier (LIAR dataset) built
  with a classical bag-of-words vs TF-IDF baseline comparison.

## Dataset

Two datasets are used, one per model.

**ISOT Fake News dataset** ([GonzaloA/fake_news](https://huggingface.co/datasets/GonzaloA/fake_news) on Hugging Face, downloaded to `data/train.csv`):

| | |
|---|---|
| Articles | 24,353 (11,107 fake / 13,246 real) |
| Columns | `title`, `text`, `label` |
| Label convention here | `1` = FAKE, `0` = REAL (the raw file uses the opposite mapping; we remap on load) |

### LIAR dataset (claim model)

**LIAR** (Wang, ACL 2017), downloaded from the author's
[liar_dataset.zip](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip), lives in
`data/liar/`:

| | |
|---|---|
| Splits | 10,269 train / 1,284 valid / 1,283 test statements |
| Input | single fact-checking claims (`statement` column only) |
| Original labels | 6-way: true, mostly-true, half-true, barely-true, false, pants-fire |
| Mapped to binary | true/mostly-true/half-true → **REAL**; barely-true/false/pants-fire → **FAKE** |

### Data-leakage note (article dataset)

95.8% of REAL articles in this dataset carry a `(Reuters)` source marker and **0%**
of FAKE articles do. A model that latches onto that single token reports ~98.4%
accuracy while learning nothing about the language of fake news. Preprocessing
therefore strips agency markers (`Reuters`, `CNN`, `AP`, datelines), and the
training script evaluates every classifier both **with** and **without** the
artifact so the effect is visible on the metrics page.

## Methodology (article model)

1. **Preprocessing** — lowercase, strip URLs/HTML/agency markers, join the title
   (repeated twice — headlines carry disproportionate signal) with the body.
2. **Features** — TF-IDF, unigrams + bigrams, top 60k features, `sublinear_tf`,
   `min_df=3`, `max_df=0.95`.
3. **Models compared** — Multinomial NB, Complement NB, Logistic Regression,
   Linear SVM (each on raw and cleaned text).
4. **Deployment choice** — Linear SVM scored highest overall (98.10% macro-F1),
   but Logistic Regression (97.27%, within the 1-point gap) is deployed because it
   produces native probabilities for the confidence display and a single
   coefficient vector for exact word-level explanations.

### Article results (held-out 20% test set, stratified, seed 42)

Final model — **Logistic Regression**: **97.29% accuracy**, **97.27% macro-F1**.

| Classifier | Cleaned acc / F1 | Raw (artifact) acc / F1 |
|---|---|---|
| MultinomialNB | 94.93% / 94.90% | 95.09% / 95.06% |
| ComplementNB | 94.97% / 94.94% | 95.15% / 95.13% |
| **LogisticRegression** | **97.29% / 97.27%** | 97.52% / 97.50% |
| LinearSVC | 98.11% / 98.10% | 98.44% / 98.43% |

## Methodology (claim model)

1. **Preprocessing** — tokenise + lowercase the statement (no agency markers to
   strip, since these are short claims).
2. **Feature sets** — two are compared: plain bag-of-words (`CountVectorizer`) and
   TF-IDF over 1–4 grams with English stop words (`TfidfVectorizer`).
3. **Classifiers** — five are compared on each feature set: Multinomial NB,
   Logistic Regression, Linear SVM, SGD (hinge) and Random Forest.
4. **Model selection + tuning** — the top-2 candidates are tuned with `GridSearchCV`
   (5-fold macro F1) over n-gram range, `use_idf` and classifier hyperparameters,
   then the best estimator is refit and evaluated on the held-out test split.
5. **Diagnostics** — a learning curve and a precision-recall curve are plotted, and
   the top-50 fake-leaning / real-leaning terms are exported.

### Claim model results (held-out LIAR test split, 1,283 statements)

**Final model — LinearSVC (TF-IDF 1–4 grams): 61.9% accuracy, 60.0% macro-F1**
(F1 of the FAKE class = 51.3%). TF-IDF n-gram baselines on LIAR typically land in
the ~0.60–0.70 band, limited by the small dataset and the difficulty of
single-claim fact-checking.

The confusion matrix, learning curve, PR curve and the full 10-row experiment
comparison are on **`/metrics/liar`**.

## Quick start

```bash
pip install -r requirements.txt

# 1. Train the article model (writes models/model.joblib, metrics.json, ...)
python -m src.train

# 2. Train the claim model (writes models/model_liar.joblib, metrics_liar.json, ...)
python -m src.train_liar

# 3. Run the web app
python app.py        # -> http://127.0.0.1:5000
```

## Web app

- **`/`** — analyse an article three ways: paste a headline + article text,
  paste an **article link** (the server fetches the page and extracts the
  article text with BeautifulSoup), or click a bundled example. You get a
  verdict card: FAKE/REAL badge, confidence bar, P(FAKE)/P(REAL), and the
  top words pushing toward each class. Pasted text takes priority over a link.
- **`/metrics`** — article-model dashboard: dataset stats, per-class precision /
  recall / F1, confusion matrix plot, and the full experiment comparison table.
- **`/metrics/liar`** — claim-model dashboard: LIAR stats, per-class report,
  confusion matrix, learning + precision-recall curves, all 10 experiments and the
  GridSearchCV results.
- **`/api/predict`** — JSON endpoint. Pass `"model": "article"` (default) or
  `"model": "statement"` to choose the classifier:

```bash
# by pasted text
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"title": "...", "text": "..."}'

# by article link (the page is fetched and the article text extracted)
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"url": "https://www.bbc.com/news/..."}'

# score a short claim with the LIAR model
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"model": "statement", "text": "Vaccines cause autism in children."}'
```

```json
{
  "label": "FAKE",
  "is_fake": true,
  "confidence": 0.8459,
  "probabilities": {"FAKE": 0.8459, "REAL": 0.1541},
  "fake_leaning_terms": [{"term": "breaking", "weight": 0.354}],
  "real_leaning_terms": [{"term": "wednesday", "weight": -0.212}]
}
```

## Project structure

```
fake-news-detection/
├── app.py                  # Flask app (/, /api/predict, /metrics, /metrics/liar)
├── data/
│   ├── train.csv           # ISOT dataset (article model)
│   └── liar/               # LIAR train/valid/test.tsv (claim model)
├── models/
│   ├── model.joblib        # fitted article pipeline (TF-IDF + LogReg)
│   ├── model_liar.joblib   # fitted claim pipeline (LinearSVC)
│   ├── metrics.json        # article experiments + final report
│   ├── metrics_liar.json   # claim experiments, grid search + final report
│   ├── examples.json       # demo articles for the UI
│   └── top_features*.json  # most fake/real-leaning terms per model
├── src/
│   ├── preprocess.py       # text cleaning, ISOT dataset loading
│   ├── liar.py             # LIAR loading + 6-to-2 label mapping
│   ├── train.py            # article training + evaluation + artifacts
│   ├── train_liar.py       # claim training (baseline comparison) + artifacts
│   ├── predict.py          # dual-model inference + per-term explanations
│   └── fetch.py            # fetch an article URL and extract its text
├── templates/              # index.html, metrics.html, metrics_liar.html
├── static/                 # style.css, confusion_matrix*.png, *_curve_liar.png
└── requirements.txt
```

## Limitations & future scope

- The models learn **style**, not truth: the article model generalises well within
  the Reuters-vs-clickbait world of its dataset but can be confidently wrong on
  out-of-domain text (science, health, non-US politics). The claim model is weaker
  still (~60% accuracy) and was trained on political fact-checks, so it can be
  confidently wrong on other domains such as health misinformation. Output is a
  signal, not a fact check.
- Ideas for extension: character n-grams + source metadata, transformer
  fine-tuning (DistilBERT), cross-dataset evaluation, SHAP-based explanations, and
  a browser extension wrapper.
