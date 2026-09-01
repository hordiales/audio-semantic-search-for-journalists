"""Persistent inverted index for YAMNet/AudioSet class retrieval."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

YAMNET_INVERTED_INDEX_FILENAME = "yamnet_inverted_index.json"
YAMNET_INVERTED_INDEX_VERSION = 1

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "audio",
        "by",
        "during",
        "event",
        "events",
        "from",
        "in",
        "near",
        "of",
        "or",
        "sound",
        "sounds",
        "the",
        "to",
        "while",
        "with",
    }
)


def normalized_audio_class_tokens(value: str) -> set[str]:
    """Normalize English AudioSet class names and natural-language queries."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    tokens = set(re.findall(r"[a-z0-9]+", normalized.lower()))
    return tokens - _STOPWORDS


def _valid_audio_class(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        class_id = str(value["class_id"])
        class_name = str(value["class_name"])
        score = float(value["score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not class_id or not class_name or not math.isfinite(score):
        return None
    return {"class_id": class_id, "class_name": class_name, "score": score}


def build_yamnet_inverted_index(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Create token postings from the stored top AudioSet classes per segment."""
    postings: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        try:
            segment_id = int(row["segment_id"])
        except (KeyError, TypeError, ValueError):
            continue
        classes = row.get("yamnet_top_classes", [])
        if not isinstance(classes, list):
            continue
        for class_rank, value in enumerate(classes):
            audio_class = _valid_audio_class(value)
            if audio_class is None:
                continue
            posting = {**audio_class, "segment_id": segment_id, "class_rank": class_rank}
            for token in sorted(normalized_audio_class_tokens(audio_class["class_name"])):
                postings[token].append(posting)
    return {
        "version": YAMNET_INVERTED_INDEX_VERSION,
        "postings": dict(sorted(postings.items())),
    }


def write_yamnet_inverted_index(path: str | Path, rows: Iterable[Mapping[str, object]]) -> int:
    """Write the release artifact and return its number of indexed tokens."""
    index = build_yamnet_inverted_index(rows)
    target = Path(path)
    target.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return len(index["postings"])


def load_yamnet_inverted_index(path: str | Path) -> dict[str, list[dict[str, object]]]:
    """Load and validate a previously generated inverted-index artifact."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("version") != YAMNET_INVERTED_INDEX_VERSION:
        raise ValueError("Unsupported YAMNet inverted-index format")
    raw_postings = raw.get("postings")
    if not isinstance(raw_postings, Mapping):
        raise ValueError("YAMNet inverted index has no postings map")

    postings: dict[str, list[dict[str, object]]] = {}
    for token, values in raw_postings.items():
        if not isinstance(token, str) or normalized_audio_class_tokens(token) != {token}:
            raise ValueError("YAMNet inverted index contains an invalid token")
        if not isinstance(values, list):
            raise ValueError("YAMNet inverted index contains invalid postings")
        parsed: list[dict[str, object]] = []
        for value in values:
            audio_class = _valid_audio_class(value)
            try:
                segment_id = int(value["segment_id"]) if isinstance(value, Mapping) else None
                class_rank = int(value["class_rank"]) if isinstance(value, Mapping) else None
            except (KeyError, TypeError, ValueError):
                raise ValueError("YAMNet inverted index contains an invalid posting") from None
            if audio_class is None or segment_id is None or class_rank is None or class_rank < 0:
                raise ValueError("YAMNet inverted index contains an invalid posting")
            parsed.append({**audio_class, "segment_id": segment_id, "class_rank": class_rank})
        postings[token] = parsed
    return postings
