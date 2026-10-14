"""Text preprocessing utilities shared by training and inference.

The ISOT-derived dataset has a known artefact: ~96% of the REAL articles carry
a "(Reuters)" source marker and no FAKE article does. A model can latch onto
that single token and reach ~99% accuracy without understanding anything about
the language of fake news. ``clean_text`` removes such agency markers so the
classifier has to learn from the actual writing style, vocabulary and
framing - which is the interesting NLP problem.
"""

from __future__ import annotations

import re

import pandas as pd

# Agency / network source markers that leak the label.
AGENCY_MARKERS = re.compile(
    r"\b(reuters|associated press|ap news|afp|cnn|bbc news|sky news)\b",
    re.IGNORECASE,
)
PAREN_SOURCE = re.compile(r"\(([A-Z][A-Za-z ]{1,20})\)\s*-\s*")  # e.g. "WASHINGTON (Reuters) - "
URLS = re.compile(r"http\S+|www\.\S+")
HTML_TAGS = re.compile(r"<[^>]+>")
MOJIBAKE = str.maketrans({"\ufffd": " ", "\u2019": "'", "\u2018": "'",
                          "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})


def clean_text(text: str, strip_sources: bool = True) -> str:
    """Normalise a raw news article into plain, lower-cased text.

    ``strip_sources=False`` keeps agency markers, which lets the training
    script measure how much the label leaks through them.
    """
    if not isinstance(text, str):
        return ""
    text = text.translate(MOJIBAKE)
    text = URLS.sub(" ", text)
    text = HTML_TAGS.sub(" ", text)
    if strip_sources:
        text = PAREN_SOURCE.sub(" ", text)
        text = AGENCY_MARKERS.sub(" ", text)
    text = text.lower()
    text = text.replace("'", "")          # "trump's" -> "trumps"
    text = re.sub(r"[^a-z0-9]+", " ", text)  # everything else -> space
    return re.sub(r"\s+", " ", text).strip()


def combine_features(title: str, text: str, strip_sources: bool = True) -> str:
    """Title and body joined; the title is repeated twice because headlines
    carry disproportionate signal in this task."""
    t = clean_text(title, strip_sources)
    b = clean_text(text, strip_sources)
    return f"{t} {t} {b}".strip()


def load_dataset(path: str, strip_sources: bool = True) -> pd.DataFrame:
    """Load the ISOT-style CSV (sep=';', cols: title, text, label).

    In this copy label 1 = REAL, 0 = FAKE. We remap to the project convention:
    1 = FAKE, 0 = REAL.
    """
    df = pd.read_csv(path, sep=";", index_col=0)
    df = df.dropna(subset=["text"]).copy()
    df["label"] = (df["label"] == 0).astype(int)  # dataset 0(fake)->1, 1(real)->0
    df["content"] = [combine_features(t, x, strip_sources)
                     for t, x in zip(df["title"], df["text"])]
    return df
