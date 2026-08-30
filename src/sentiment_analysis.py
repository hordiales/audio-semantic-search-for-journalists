"""Sentiment analysis module for Spanish text."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    positive: float
    negative: float
    neutral: float
    dominant: str


class SentimentAnalyzer:
    """Sentiment analysis using pysentimiento (Spanish-optimized)."""

    def __init__(self, model_name: str = "pysentimiento/robertuito-sentiment-analysis"):
        self.model_name = model_name
        self._analyzer = None

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._load()
        return self._analyzer

    def _load(self):
        """Lazy-load the sentiment analyzer."""
        try:
            from pysentimiento import create_analyzer

            logger.info("Loading sentiment analyzer: pysentimiento (es)")
            self._analyzer = create_analyzer(task="sentiment", lang="es")
        except Exception as e:
            logger.warning("Failed to load pysentimiento: %s. Using fallback.", e)
            self._analyzer = self._create_fallback()

    def _create_fallback(self):
        """Fallback using transformers pipeline."""
        from transformers import pipeline

        logger.info(
            "Loading fallback sentiment model: cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        return pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
        )

    def analyze(self, text: str) -> SentimentResult:
        """Analiza sentimiento de un texto."""
        if not text or not text.strip():
            return SentimentResult(positive=0.0, negative=0.0, neutral=1.0, dominant="neutral")

        try:
            result = self.analyzer.predict(text)
            scores = result.probas
            positive = scores.get("POS", scores.get("positive", 0.0))
            negative = scores.get("NEG", scores.get("negative", 0.0))
            neutral = scores.get("NEU", scores.get("neutral", 0.0))
        except (AttributeError, TypeError):
            positive, negative, neutral = self._fallback_predict(text)

        dominant = max(
            [("positive", positive), ("negative", negative), ("neutral", neutral)],
            key=lambda x: x[1],
        )[0]

        return SentimentResult(
            positive=round(positive, 4),
            negative=round(negative, 4),
            neutral=round(neutral, 4),
            dominant=dominant,
        )

    def _fallback_predict(self, text: str) -> tuple[float, float, float]:
        """Fallback prediction using transformers pipeline."""
        results = self.analyzer(text[:512])
        positive = negative = neutral = 0.0
        if isinstance(results, list) and isinstance(results[0], list):
            results = results[0]
        for r in results:
            label = r["label"].lower()
            score = r["score"]
            if "pos" in label:
                positive = score
            elif "neg" in label:
                negative = score
            else:
                neutral = score
        return positive, negative, neutral

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[SentimentResult]:
        """Analyze sentiment for a batch of texts."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(
                "Sentiment batch %d/%d",
                i // batch_size + 1,
                (len(texts) + batch_size - 1) // batch_size,
            )
            for text in batch:
                results.append(self.analyze(text))
        return results


def analyze_sentiment(text: str) -> SentimentResult:
    """Analiza sentimiento de un texto (convenience function)."""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(text)
