"""Validate this project's CLAP wiring against the public Clotho retrieval benchmark.

Clotho pairs each audio clip with five human-written captions. Embedding every
clip and every caption with CLAP and ranking clips by similarity to each
caption is the standard zero-shot text-to-audio retrieval protocol used in the
CLAP paper (Wu et al., 2023), reported as Recall@1/5/10.

Scope: this benchmark is a sanity check of the implementation (model loading,
normalization, batching), not a validation of CLAP for the Spanish journalism
corpus. Clotho clips are short, clean, English-captioned Freesound recordings
of isolated acoustic events; the project's corpus is Argentine radio speech.
A good score here only rules out an integration bug. See
docs/evaluar-clap-clotho.md for how this fits into the CLAP evaluation plan
started in docs/reporte-comparativo-texto-clap.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

# Inicializar PyTorch antes que FAISS para evitar conflictos con torchlibrosa
# en Apple Silicon.
import torch

torch.set_num_threads(1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from evaluation.retrieval_evaluation import compute_retrieval_metrics  # noqa: E402

from src.clap_audio_embeddings import CLAPConfig, CLAPEmbedding  # noqa: E402

logger = logging.getLogger(__name__)

CAPTION_COLUMNS = ["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]

# Wu et al., "Large-scale Contrastive Language-Audio Pretraining with Feature
# Fusion and Keyword-to-Caption Augmentation" (2023), Table 3, CLAP-HTSAT
# (AudioCaps + Clotho + WT5K), Clotho text-to-audio retrieval. Reference only:
# published numbers vary by checkpoint and eval protocol, use as an order-of-
# magnitude sanity check rather than an exact target.
REFERENCE_CLOTHO_TEXT_TO_AUDIO = {"recall_at_1": 0.167, "recall_at_5": 0.411, "recall_at_10": 0.541}


def _load_captions(captions_csv: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(captions_csv)
    missing = [c for c in ["file_name", *CAPTION_COLUMNS] if c not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{captions_csv} is missing columns {missing}. Expected the official Clotho "
            "captions CSV format (file_name, caption_1..caption_5), e.g. "
            "clotho_captions_evaluation.csv."
        )
    return dataframe


def _resolve_audio_paths(audio_dir: Path, file_names: list[str]) -> list[Path]:
    paths = [audio_dir / name for name in file_names]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files referenced in the captions CSV were not found in "
            f"{audio_dir} (e.g. {missing[:3]}). Extract clotho_audio_evaluation.7z there first."
        )
    return paths


def _load_or_compute_audio_embeddings(
    clap: CLAPEmbedding, audio_paths: list[Path], cache_path: Path | None
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(audio_paths):
            logger.info("Reusing cached audio embeddings from %s", cache_path)
            return cached
        logger.warning(
            "Cache %s has %d vectors but %d audio files were requested; recomputing.",
            cache_path,
            cached.shape[0],
            len(audio_paths),
        )

    embeddings = clap.generate_batch_audio_embeddings([str(p) for p in audio_paths])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
    return embeddings


def evaluate_clap_on_clotho(
    audio_dir: str,
    captions_csv: str,
    clap_model: str = "laion/clap-htsat-unfused",
    max_files: int | None = None,
    max_k: int = 10,
    cache_path: str | None = None,
) -> dict:
    """Zero-shot text-to-audio retrieval: each caption is a query with exactly
    one relevant clip (the one it was written for)."""
    captions = _load_captions(Path(captions_csv))
    if max_files is not None:
        captions = captions.head(max_files)

    audio_paths = _resolve_audio_paths(Path(audio_dir), captions["file_name"].tolist())

    # Force English for the text encoder; this is an English benchmark and
    # avoids accidental translation if QUERY_LANGUAGE is set to something else.
    clap = CLAPEmbedding(CLAPConfig(model_name=clap_model, query_language="en"))
    audio_embeddings = _load_or_compute_audio_embeddings(
        clap, audio_paths, Path(cache_path) if cache_path else None
    )

    # Build query list once so we can embed all captions in a single batch call.
    query_rows = []
    for audio_idx, row in enumerate(captions.itertuples(index=False)):
        for column in CAPTION_COLUMNS:
            caption = getattr(row, column)
            if isinstance(caption, str) and caption.strip():
                query_rows.append(
                    {
                        "audio_idx": audio_idx,
                        "file_name": row.file_name,
                        "caption_column": column,
                        "caption": caption,
                    }
                )

    logger.info("Generating %d text query embeddings", len(query_rows))
    query_embeddings = clap.generate_text_embeddings_batch(
        [row["caption"] for row in query_rows], batch_size=100
    )

    # Audio and query embeddings are already L2-normalized, so their dot product
    # is cosine similarity. Sorting once per query gives the full ranking.
    similarity_matrix = query_embeddings @ audio_embeddings.T
    ranked_indices = np.argsort(-similarity_matrix, axis=1)

    k_values = [1, 5, 10]
    max_k = max(max_k, max(k_values))

    per_query = []
    for query_row, ranking in zip(query_rows, ranked_indices, strict=True):
        ranked_ids = [int(i) for i in ranking]
        metrics = compute_retrieval_metrics(ranked_ids, {query_row["audio_idx"]}, k_values)
        per_query.append(
            {
                "file_name": query_row["file_name"],
                "caption_column": query_row["caption_column"],
                "caption": query_row["caption"],
                "metrics": {
                    "precision_at": metrics.precision_at,
                    "recall_at": metrics.recall_at,
                    "mrr": metrics.mrr,
                    "ndcg_at": metrics.ndcg_at,
                },
            }
        )

    aggregated = {
        "recall_at": {
            k: float(np.mean([q["metrics"]["recall_at"][k] for q in per_query])) for k in k_values
        },
        "precision_at": {
            k: float(np.mean([q["metrics"]["precision_at"][k] for q in per_query]))
            for k in k_values
        },
        "mrr": float(np.mean([q["metrics"]["mrr"] for q in per_query])),
        "ndcg_at": {
            k: float(np.mean([q["metrics"]["ndcg_at"][k] for q in per_query])) for k in k_values
        },
    }

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": (
            "Sanity check of this project's CLAP wiring against the public Clotho "
            "text-to-audio retrieval benchmark. A good score here rules out an "
            "implementation bug; it does not show CLAP works on the Spanish "
            "journalism corpus (different domain and language). See "
            "docs/evaluar-clap-clotho.md."
        ),
        "configuration": {
            "clap_model": clap_model,
            "n_audio_files": len(audio_paths),
            "n_queries": len(per_query),
            "max_k": max_k,
        },
        "reference_published_scores": {
            "source": (
                "Wu et al. 2023, CLAP-HTSAT (AudioCaps+Clotho+WT5K), Table 3 - "
                "reference only, not an exact target"
            ),
            "clotho_text_to_audio": REFERENCE_CLOTHO_TEXT_TO_AUDIO,
        },
        "aggregated": aggregated,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir", required=True, help="Directory with extracted Clotho audio clips"
    )
    parser.add_argument(
        "--captions-csv", required=True, help="Path to clotho_captions_evaluation.csv (or dev/val)"
    )
    parser.add_argument("--output", required=True, help="Path for the results JSON")
    parser.add_argument("--clap-model", default="laion/clap-htsat-unfused")
    parser.add_argument(
        "--max-files", type=int, default=None, help="Subsample the split for a quick smoke test"
    )
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--cache", default=None, help="Optional .npy path to cache audio embeddings across runs"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    report = evaluate_clap_on_clotho(
        audio_dir=args.audio_dir,
        captions_csv=args.captions_csv,
        clap_model=args.clap_model,
        max_files=args.max_files,
        max_k=args.max_k,
        cache_path=args.cache,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}")
    print("CLAP on Clotho (zero-shot text-to-audio retrieval)")
    print(f"{'=' * 60}")
    print(
        f"Audio files: {report['configuration']['n_audio_files']} | "
        f"Queries: {report['configuration']['n_queries']}"
    )
    for k in (1, 5, 10):
        published = REFERENCE_CLOTHO_TEXT_TO_AUDIO[f"recall_at_{k}"]
        print(
            f"Recall@{k}: {report['aggregated']['recall_at'][k]:.4f}  (published ref: {published:.4f})"
        )
    print(f"MRR: {report['aggregated']['mrr']:.4f}")
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
