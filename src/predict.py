"""Load the trained models and score news text (used by the Flask app).

Two models are served:
  * ``article`` - ISOT full-article model (TF-IDF 1-2 grams + LogisticRegression)
  * ``statement`` - LIAR short-claim model following the reference project
    (trained by src/train_liar.py)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import numpy as np

from src.preprocess import combine_features

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

_lock = threading.Lock()
_bundles: dict[str, dict] = {}

LABEL_NAMES = {0: "REAL", 1: "FAKE"}

# model key -> (joblib filename, whether text needs title+body preprocessing)
MODEL_SPECS = {
    "article": ("model.joblib", True),
    "statement": ("model_liar.joblib", False),
}


def _load(model: str) -> dict:
    if model not in MODEL_SPECS:
        raise KeyError(f"Unknown model '{model}'")
    if model not in _bundles:
        with _lock:
            if model not in _bundles:
                filename, _ = MODEL_SPECS[model]
                path = MODELS_DIR / filename
                if not path.exists():
                    raise FileNotFoundError(
                        f"{path.name} not found. Train first: "
                        f"python -m src.train{'_liar' if model == 'statement' else ''}")
                _bundles[model] = joblib.load(path)
    return _bundles[model]


def available_models() -> list[str]:
    ready = []
    for model in MODEL_SPECS:
        filename, _ = MODEL_SPECS[model]
        if (MODELS_DIR / filename).exists():
            ready.append(model)
    return ready


def _explain(pipeline, content: str, top_k: int) -> tuple[list, list]:
    """Per-term contributions: tfidf_value * coefficient (positive = FAKE)."""
    vec = pipeline.named_steps["vec"] if "vec" in pipeline.named_steps \
        else pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    x = vec.transform([content])
    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
    else:
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]

    contrib = x.multiply(weights).toarray().ravel()
    nz = x.nonzero()[1]
    pairs = [(int(i), float(contrib[i])) for i in np.unique(nz)]
    names = vec.get_feature_names_out()

    fake = sorted((p for p in pairs if p[1] > 0), key=lambda p: -p[1])[:top_k]
    real = sorted((p for p in pairs if p[1] < 0), key=lambda p: p[1])[:top_k]
    return ([{"term": names[i], "weight": round(w, 3)} for i, w in fake],
            [{"term": names[i], "weight": round(w, 3)} for i, w in real])


def predict_text(title: str, text: str, model: str = "article",
                 top_k: int = 8) -> dict:
    """Score text with the selected model; returns verdict + explanations."""
    bundle = _load(model)
    pipeline = bundle["pipeline"]

    _, needs_combine = MODEL_SPECS[model]
    content = combine_features(title, text) if needs_combine else text

    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipeline.predict_proba([content])[0]
        pred = int(np.argmax(proba))
        p_fake, p_real = float(proba[1]), float(proba[0])
        confidence = float(proba[pred])
        prob_out = {"FAKE": round(p_fake, 4), "REAL": round(p_real, 4)}
    else:
        # Calibrated-margin fallback (e.g. LinearSVC): squash the decision
        # function into [0,1] so we still report a probability-like confidence.
        score = float(pipeline.decision_function([content])[0])
        p_fake = 1.0 / (1.0 + np.exp(-score))
        p_real = 1.0 - p_fake
        pred = int(score >= 0)  # class 1 = FAKE
        confidence = max(p_fake, p_real)
        prob_out = {"FAKE": round(float(p_fake), 4), "REAL": round(float(p_real), 4)}

    fake_terms, real_terms = _explain(pipeline, content, top_k)

    return {
        "label": LABEL_NAMES[pred],
        "is_fake": bool(pred == 1),
        "confidence": round(confidence, 4),
        "probabilities": prob_out,
        "fake_leaning_terms": fake_terms,
        "real_leaning_terms": real_terms,
        "model": model,
        "model_label": {"article": "ISOT article model",
                        "statement": "LIAR claim model"}.get(model, model),
    }


# Backwards-compatible helper used by older call sites/tests
def predict_article(title: str, text: str, top_k: int = 8) -> dict:
    return predict_text(title, text, model="article", top_k=top_k)


def load_metrics(model: str):
    name = "metrics.json" if model == "article" else "metrics_liar.json"
    path = MODELS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
