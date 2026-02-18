from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


@dataclass
class FinBERTScorer:
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16

    def __post_init__(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )

    def score(self, texts: pd.Series) -> pd.DataFrame:
        cleaned = texts.fillna("").astype(str).str.strip().tolist()
        rows: list[dict[str, float | str]] = []
        outputs = self._pipe(cleaned, batch_size=self.batch_size)

        for output in outputs:
            scores = {str(item["label"]).lower(): float(item["score"]) for item in output}
            p_pos = scores.get("positive", 0.0)
            p_neu = scores.get("neutral", 0.0)
            p_neg = scores.get("negative", 0.0)
            sentiment_score = p_pos - p_neg

            label = "neutral"
            if sentiment_score > 0:
                label = "positive"
            elif sentiment_score < 0:
                label = "negative"

            rows.append(
                {
                    "p_positive": p_pos,
                    "p_neutral": p_neu,
                    "p_negative": p_neg,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": label,
                    "confidence": max(p_pos, p_neu, p_neg),
                }
            )

        return pd.DataFrame(rows)


def build_article_text(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[["title", "description", "content"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
