"""LIAR dataset loading (Wang, ACL 2017) with the reference project's binary mapping.

The raw TSV has 14 columns:
  0 id, 1 label, 2 statement, 3 subject, 4 speaker, 5 job, 6 state, 7 party,
  8-12 credit-history counts, 13 context

Following the standard binary setup for LIAR we keep only statement + label
and collapse the 6 classes to 2:
  true / mostly-true / half-true            -> REAL (0)
  barely-true / false / pants-fire          -> FAKE (1)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocess import clean_text

LIAR_DIR = Path(__file__).resolve().parent.parent / "data" / "liar"

COLUMNS = ["id", "label", "statement", "subject", "speaker", "job", "state",
           "party", "barely_true", "false", "half_true", "mostly_true",
           "pants_fire", "context"]

# 6-way LIAR label -> project convention (1 = FAKE, 0 = REAL)
LABEL_MAP = {
    "true": 0, "mostly-true": 0, "half-true": 0,
    "barely-true": 1, "false": 1, "pants-fire": 1,
}


def load_liar(split: str, liar_dir: Path | str = LIAR_DIR) -> pd.DataFrame:
    """Load one LIAR split ('train' / 'valid' / 'test') as [text, label]."""
    path = Path(liar_dir) / f"{split}.tsv"
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS,
                     quoting=3, dtype=str, keep_default_na=False)
    df = df[df["label"].isin(LABEL_MAP)].copy()
    df["label"] = df["label"].map(LABEL_MAP).astype(int)
    df["text"] = [clean_text(s, strip_sources=False) for s in df["statement"]]
    df = df[df["text"].str.len() > 0]
    return df[["text", "label", "statement"]].reset_index(drop=True)


def label_distribution(df: pd.DataFrame) -> dict:
    return {"train_like_rows": int(len(df)),
            "fake": int(df.label.sum()),
            "real": int((1 - df.label).sum())}
