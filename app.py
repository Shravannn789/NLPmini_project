"""Flask web app for Fake News Detection.

Run from the project root (after training):

    python app.py

Routes:
    /            - analyse an article by pasting text or a link
    /api/predict - POST {"title": "...", "text": "..."} or {"url": "..."} -> JSON
    /metrics     - model evaluation dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.fetch import FetchError, fetch_article
from src.predict import (MODELS_DIR, available_models, load_metrics,
                         predict_text)

app = Flask(__name__)

MIN_CHARS = {"article": 40, "statement": 15}
NOT_ENOUGH = {"article": ("Please paste at least a couple of sentences (~40 characters) "
                          "or provide an article link so the model has enough context."),
              "statement": ("Please type a claim of at least ~15 characters, e.g. "
                            "\"Vaccines cause autism\"")}


def _read_json(name: str, default):
    path = MODELS_DIR / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _resolve_input(title: str, text: str, url: str):
    """Fetch the page when only a link was provided.

    Returns (title, text, url, error); pasted text always wins over the link.
    """
    if not text and url:
        try:
            article = fetch_article(url)
        except FetchError as exc:
            return title, "", url, str(exc)
        title = title or article["title"]
        text = article["text"]
    return title, text, url, None


@app.route("/", methods=["GET", "POST"])
def index():
    examples = _read_json("examples.json", [])
    result, error, url = None, None, ""
    title, text = "", ""
    model = "article"
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        text = request.form.get("text", "").strip()
        url = request.form.get("url", "").strip()
        model = request.form.get("model", "article").strip() or "article"
        if model not in available_models():
            error = f"Model '{model}' is not trained yet."
            model = "article"
        title, text, url, error = _resolve_input(title, text, url)
        if error:
            pass
        elif len(text) < MIN_CHARS.get(model, 40):
            error = NOT_ENOUGH.get(model, NOT_ENOUGH["article"])
        else:
            try:
                result = predict_text(title, text, model=model)
            except FileNotFoundError as exc:
                error = str(exc)
    return render_template("index.html", result=result, error=error,
                           title=title, text=text, url=url, examples=examples,
                           model=model, models=available_models())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    text = (payload.get("text") or "").strip()
    url = (payload.get("url") or "").strip()
    model = (payload.get("model") or "article").strip()
    if model not in available_models():
        return jsonify({"error": f"Unknown or untrained model '{model}'. "
                                 f"Available: {available_models()}"}), 400
    title, text, url, error = _resolve_input(title, text, url)
    if error:
        return jsonify({"error": error}), 422
    if len(text) < MIN_CHARS.get(model, 40):
        return jsonify({"error": NOT_ENOUGH.get(model, NOT_ENOUGH["article"])}), 400
    try:
        verdict = predict_text(title, text, model=model)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    if url:
        verdict["source_url"] = url
        verdict["article_title"] = title
    return jsonify(verdict)


@app.route("/metrics")
def metrics():
    data = _read_json("metrics.json", None)
    return render_template("metrics.html", m=data,
                           has_matrix=(Path(app.root_path) / "static" /
                                       "confusion_matrix.png").exists())


@app.route("/metrics/liar")
def metrics_liar():
    data = load_metrics("statement")
    return render_template("metrics_liar.html", m=data)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
