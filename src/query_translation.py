"""Lightweight query translator for English CLAP and AudioSet/YAMNet search.

`laion/clap-htsat-unfused` uses a RoBERTa-base text encoder that was trained on
English text. Queries in other languages should be translated before generating
the CLAP text embedding or matching the English AudioSet class taxonomy.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

ENGLISH_ALIASES = {"en", "english", "ingles", "inglés"}


@lru_cache(maxsize=1)
def _openai_client():
    try:
        import openai
    except ImportError as error:
        raise RuntimeError(
            "Translation requires the 'openai' package. Install it or set "
            "QUERY_LANGUAGE=en to skip translation."
        ) from error
    return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _is_english(language: str) -> bool:
    return language.strip().lower().split("_")[0].split("-")[0] in ENGLISH_ALIASES


def translate_to_english(text: str, source_language: str | None = None) -> str:
    """Translate a query to English if the configured source language is not English.

    Args:
        text: Original query text.
        source_language: Source language code or name. If None, read from
            ``QUERY_LANGUAGE`` environment variable, defaulting to "en".

    Returns:
        The original text if already English or if translation is disabled;
        otherwise the English translation.
    """
    if source_language is None:
        source_language = os.environ.get("QUERY_LANGUAGE", "en")
    if _is_english(source_language):
        return text
    if not text or not text.strip():
        return text

    client = _openai_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    logger.debug("Translating query from %s to English with %s", source_language, model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translator for an audio retrieval system. "
                    "Translate the user's query to English. Keep proper nouns, "
                    "brand names, quoted phrases and numbers unchanged. "
                    "Return only the translation, no explanations."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    translated = response.choices[0].message.content.strip()
    logger.debug("Translated query: %r -> %r", text, translated)
    return translated
