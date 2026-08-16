"""Optional YAMNet classifier for AudioSet acoustic-event labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class YAMNetConfig:
    """Configuration for segment-level YAMNet classification."""

    model_url: str = YAMNET_MODEL_URL
    top_k: int = 5


def aggregate_yamnet_classes(
    window_classes: list[list[dict[str, Any]]], top_k: int
) -> list[dict[str, Any]]:
    """Pool window predictions by maximum score so brief events are retained."""
    by_id: dict[str, dict[str, Any]] = {}
    for classes in window_classes:
        for item in classes:
            class_id = str(item["class_id"])
            score = float(item["score"])
            if class_id not in by_id or score > float(by_id[class_id]["score"]):
                by_id[class_id] = {
                    "class_id": class_id,
                    "class_name": str(item["class_name"]),
                    "score": score,
                }
    return sorted(by_id.values(), key=lambda item: item["score"], reverse=True)[:top_k]


class YAMNetAudioClassifier:
    """Classify 16 kHz mono WAV clips with the TensorFlow Hub YAMNet model."""

    def __init__(self, config: YAMNetConfig | None = None):
        self.config = config or YAMNetConfig()
        self._tf: Any | None = None
        self._model: Any | None = None
        self._class_names: list[dict[str, str]] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as error:
            raise RuntimeError(
                "YAMNet is optional. Install it with `uv sync --extra yamnet` "
                "before enabling yamnet in config/embeddings.toml."
            ) from error

        self._tf = tf
        self._model = hub.load(self.config.model_url)
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path) as class_map:
            self._class_names = list(csv.DictReader(class_map))

    def classify(self, audio_path: str | Path) -> list[dict[str, Any]]:
        """Return the top AudioSet classes for one audio window."""
        if self.config.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self._load()
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1, dtype=np.float32)
        if sample_rate != YAMNET_SAMPLE_RATE:
            raise ValueError(
                f"YAMNet requires {YAMNET_SAMPLE_RATE} Hz audio; got {sample_rate} Hz from {audio_path}"
            )
        scores, _, _ = self._model(self._tf.convert_to_tensor(waveform, dtype=self._tf.float32))
        mean_scores = np.asarray(scores.numpy()).mean(axis=0)
        top_indices = np.argsort(mean_scores)[::-1][: self.config.top_k]
        return [
            {
                "class_id": self._class_names[int(index)]["mid"],
                "class_name": self._class_names[int(index)]["display_name"],
                "score": float(mean_scores[int(index)]),
            }
            for index in top_indices
        ]
