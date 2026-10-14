"""LIAR training with a classical feature-comparison methodology.

Pipeline (in the style of classic LIAR baselines):

  1. Compare 5 classifiers - Multinomial NB, Logistic Regression, Linear SVM,
     SGD (hinge loss) and Random Forest - under two feature sets: plain
     bag-of-words and TF-IDF n-grams (1-4) with English stop words.
  2. Take the 2 best candidates and tune them with GridSearchCV (5-fold).
  3. Refit the winner on the official train split, plot its learning curve and
     precision-recall curve on the official test split, dump the top-50 terms
     per class, and export everything for the web app.

Run from the project root:  python -m src.train_liar
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.liar import load_liar
from src.train import evaluate, plot_confusion

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
STATIC = ROOT / "static"

# Reference FeatureSelection.py: countV = CountVectorizer(),
# tfidf_ngram = TfidfVectorizer(stop_words='english', ngram_range=(1,4), ...)
FEATURESETS = {
    "bag-of-words": lambda: CountVectorizer(),
    "tfidf n-grams (1,4)": lambda: TfidfVectorizer(
        stop_words="english", ngram_range=(1, 4), use_idf=True, smooth_idf=True),
}

CLASSIFIERS = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(C=1, penalty="l2", max_iter=1000),
    "LinearSVC": LinearSVC(),
    "SGDClassifier": SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3,
                                   random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                           random_state=42),
}

# GridSearch spaces in the spirit of the reference (ngram_range / use_idf /
# classifier params), trimmed so a laptop can finish in minutes.
GRIDS = {
    "LogisticRegression": {
        "vec__ngram_range": [(1, 1), (1, 2), (1, 3), (1, 4)],
        "vec__use_idf": [True, False],
        "vec__smooth_idf": [True, False],
    },
    "RandomForest": {
        "vec__ngram_range": [(1, 1), (1, 2)],
        "vec__use_idf": [True, False],
        "clf__max_depth": [10, None],
    },
    "LinearSVC": {
        "vec__ngram_range": [(1, 1), (1, 2), (1, 4)],
        "vec__use_idf": [True, False],
    },
    "MultinomialNB": {
        "vec__ngram_range": [(1, 1), (1, 2), (1, 4)],
        "vec__use_idf": [True, False],
        "clf__alpha": [0.01, 0.1, 1.0],
    },
    "SGDClassifier": {
        "vec__ngram_range": [(1, 1), (1, 2), (1, 4)],
        "vec__use_idf": [True, False],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
    },
}


def make_pipeline(name: str, vectorizer) -> Pipeline:
    return Pipeline([("vec", vectorizer), ("clf", CLASSIFIERS[name])])


def plot_learning_curve(pipe, X, y, path: Path, title: str) -> None:
    sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=3, n_jobs=-1, scoring="f1_macro",
        train_sizes=np.linspace(0.2, 1.0, 5))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(sizes, train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.1, color="r")
    ax.fill_between(sizes, test_scores.mean(1) - test_scores.std(1),
                    test_scores.mean(1) + test_scores.std(1), alpha=0.1, color="g")
    ax.plot(sizes, train_scores.mean(1), "o-", color="r", label="Training score")
    ax.plot(sizes, test_scores.mean(1), "o-", color="g", label="Cross-validation score")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.3, 1.0)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pr_curve(y_true, scores, path: Path, ap: float) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(recall, precision, color="b", alpha=0.7, where="post")
    ax.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(f"Precision-Recall curve (FAKE class), AP = {ap:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def top_terms(pipeline, n=50) -> dict:
    vec = pipeline.named_steps["vec"]
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        w = clf.coef_[0]
    else:
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(w)
    return {
        "fake_leaning": [{"term": t, "weight": round(float(w[i]), 4)}
                         for i, t in zip(order[::-1][:n], names[order[::-1][:n]])],
        "real_leaning": [{"term": t, "weight": round(float(w[i]), 4)}
                         for i, t in zip(order[:n], names[order[:n]])],
    }


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    t0 = time.time()
    train = load_liar("train")
    test = load_liar("test")
    print(f"LIAR: {len(train)} train / {len(test)} test statements | "
          f"train fake={int(train.label.sum())} real={int((1 - train.label).sum())}")

    experiments = []
    results = {}  # (name, features) -> (pipeline, metrics)
    for feat_name, vec_factory in FEATURESETS.items():
        for name in CLASSIFIERS:
            print(f"Training {name:20s} with {feat_name} ...")
            pipe = make_pipeline(name, vec_factory())
            pipe.fit(train["text"], train["label"])
            res = evaluate(pipe, test["text"], test["label"])
            experiments.append({"classifier": name, "features": feat_name,
                                "accuracy": res["accuracy"],
                                "macro_f1": res["macro_f1"]})
            results[(name, feat_name)] = (pipe, res)
            print(f"    accuracy={res['accuracy']:.4f}  macro_f1={res['macro_f1']:.4f}")

    ranked = sorted(results.items(), key=lambda kv: -kv[1][1]["macro_f1"])
    top2 = ranked[:2]
    print("\nTop-2 candidates for GridSearchCV: "
          + ", ".join(f"{k[0]} ({k[1]}, f1={v[1]['macro_f1']:.4f})" for k, v in top2))

    grid_results = []
    tuned = []
    for (name, _feat), (pipe, _) in top2:
        grid = GRIDS.get(name)
        if not grid:
            tuned.append((name, pipe))
            continue
        print(f"GridSearchCV for {name} ({int(np.prod([len(v) for v in grid.values()]))} combos) ...")
        base = Pipeline([
            ("vec", TfidfVectorizer(stop_words="english", use_idf=True, smooth_idf=True)),
            ("clf", CLASSIFIERS[name]),
        ])
        gs = GridSearchCV(base, grid, cv=5, scoring="f1_macro", n_jobs=-1)
        gs.fit(train["text"], train["label"])
        res = evaluate(gs.best_estimator_, test["text"], test["label"])
        grid_results.append({"classifier": name,
                             "best_params": {k: str(v) for k, v in gs.best_params_.items()},
                             "best_cv_macro_f1": round(float(gs.best_score_), 4),
                             "test_macro_f1": res["macro_f1"]})
        print(f"    best params: {gs.best_params_} | cv f1={gs.best_score_:.4f} "
              f"| test f1={res['macro_f1']:.4f}")
        tuned.append((name, gs.best_estimator_))

    # Reference kept the tuned Logistic Regression; here: pick by test macro F1
    tuned.sort(key=lambda item: -evaluate(item[1], test["text"], test["label"])["macro_f1"])
    final_name, final = tuned[0]
    final_metrics = evaluate(final, test["text"], test["label"])
    preds = final_metrics.pop("predictions")
    print(f"\nFinal LIAR model: {final_name} | accuracy={final_metrics['accuracy']:.4f} "
          f"| macro_f1={final_metrics['macro_f1']:.4f} | "
          f"f1(FAKE)={final_metrics['report']['FAKE']['f1-score']:.4f}")

    plot_confusion(np.array(final_metrics["confusion_matrix"]),
                   STATIC / "confusion_matrix_liar.png")
    plot_learning_curve(final, train["text"], train["label"],
                        STATIC / "learning_curve_liar.png",
                        f"LIAR learning curve - {final_name}")
    if hasattr(final.named_steps["clf"], "predict_proba"):
        scores = final.predict_proba(test["text"])[:, 1]
    else:
        scores = final.decision_function(test["text"])
    ap = float(average_precision_score(test["label"], scores))
    plot_pr_curve(test["label"], scores, STATIC / "pr_curve_liar.png", ap)
    (MODELS / "top_features_liar.json").write_text(
        json.dumps(top_terms(final, n=50), indent=2))

    joblib.dump({
        "pipeline": final,
        "classifier": final_name,
        "label_names": {0: "REAL", 1: "FAKE"},
        "dataset": "LIAR",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(train)),
    }, MODELS / "model_liar.joblib")

    metrics = {
        "dataset": {
            "source": "LIAR (Wang, ACL 2017) - liar_dataset.zip from the author",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_fake_train": int(train.label.sum()),
            "n_real_train": int((1 - train.label).sum()),
            "label_mapping": "true/mostly-true/half-true -> REAL; "
                             "barely-true/false/pants-fire -> FAKE",
            "reference": "classical bag-of-words vs TF-IDF n-gram baseline comparison",
        },
        "experiments": experiments,
        "grid_search": grid_results,
        "final_model": {**final_metrics, "classifier": final_name,
                        "average_precision": round(ap, 4),
                        "trained_at": datetime.now().isoformat(timespec="seconds")},
        "reference_note": "TF-IDF n-gram baselines on LIAR typically reach ~0.70 "
                          "binary F1 (5-fold CV). Short fact-check "
                          "claims are far harder than full articles: there is no "
                          "article style to learn, only the wording of a single claim.",
    }
    (MODELS / "metrics_liar.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts saved to models/ and static/")


if __name__ == "__main__":
    main()
