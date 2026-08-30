"""Genera un dataset de preguntas de evaluación sintéticas a partir del corpus procesado."""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

logger = logging.getLogger(__name__)

DEFAULT_DATASET_SIZE = 20
DEFAULT_MAX_SEGMENTS = 0
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SYNTHETIC_DATASET_PATH = "./evaluation/test_datasets/synthetic/ragas_synthetic_questions.json"
DEFAULT_TIMEOUT = 1200
DEFAULT_MAX_WORKERS = 1
DEFAULT_LANGUAGE = "es"


def load_corpus_segments(dataset_path: str, max_segments: int) -> list[Document]:
    """Carga los segmentos de texto del dataset procesado como Documentos de LangChain."""
    corpus_pkl = Path(dataset_path) / "final" / "complete_dataset.pkl"
    if not corpus_pkl.exists():
        raise FileNotFoundError(f"No se encontró el corpus en {corpus_pkl}")

    df = pd.read_pickle(corpus_pkl)
    documents: list[Document] = []
    for _, row in df.iterrows():
        text = row.get("text")
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue
        metadata = {
            k: row.get(k)
            for k in ("segment_id", "start_time", "end_time", "original_file_name")
            if k in row
        }
        documents.append(Document(page_content=text.strip(), metadata=metadata))

    if max_segments > 0:
        documents = documents[:max_segments]

    return documents


def _testset_to_eval_samples(testset) -> list[dict]:
    """Convierte un Testset de RAGAS al formato de evaluación del proyecto."""
    samples: list[dict] = []
    for raw in testset.to_list():
        item: dict = {"question": raw.get("user_input", "")}
        if raw.get("reference"):
            item["ground_truth"] = raw["reference"]
        if raw.get("reference_contexts"):
            item["ground_truth_contexts"] = raw["reference_contexts"]
        if raw.get("synthesizer_name"):
            item["synthesizer_name"] = raw["synthesizer_name"]
        samples.append(item)
    return samples


def _translate_samples(samples: list[dict], language: str, llm: ChatOpenAI) -> list[dict]:
    """Traduce question y ground_truth de cada muestra al idioma solicitado."""
    if not samples or language == "en":
        return samples

    texts = [
        {"question": s["question"], "ground_truth": s.get("ground_truth", "")}
        for s in samples
    ]
    prompt = (
        f"Translate the 'question' and 'ground_truth' fields of each item to {language}.\n"
        "Keep proper nouns, numbers, quoted names and context references unchanged. "
        "Output ONLY a JSON array of objects with keys 'question' and 'ground_truth', "
        "in the same order as the input. Do not include markdown code fences or explanations.\n\n"
        f"Input:\n{json.dumps(texts, ensure_ascii=False, indent=2)}"
    )
    response = llm.invoke([HumanMessage(content=prompt)]).content

    raw = response.strip()
    if raw.startswith("```"):
        # strip ``` or ```json fences
        raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0]
    translated = json.loads(raw)

    if len(translated) != len(samples):
        raise ValueError(
            f"Translation returned {len(translated)} items, expected {len(samples)}"
        )

    for sample, tr in zip(samples, translated):
        sample["question"] = tr["question"]
        if "ground_truth" in tr:
            sample["ground_truth"] = tr["ground_truth"]
    return samples


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Genera preguntas de evaluación sintéticas con RAGAS a partir del corpus"
    )
    parser.add_argument(
        "--dataset-path",
        default=os.getenv("DATASET_PATH", "./dataset"),
        help="Ruta al dataset procesado que contiene final/complete_dataset.pkl",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("EVALUATION_SYNTHETIC_DATASET_PATH", DEFAULT_SYNTHETIC_DATASET_PATH),
        help="Ruta de salida del JSON con preguntas sintéticas",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=int(os.getenv("EVALUATION_DATASET_SIZE", str(DEFAULT_DATASET_SIZE))),
        help="Cantidad de preguntas a generar",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=int(os.getenv("EVALUATION_DATASET_MAX_SEGMENTS", str(DEFAULT_MAX_SEGMENTS))),
        help="Máximo de segmentos del corpus a usar (0 = todos)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        help="Modelo de lenguaje de OpenAI para generar las preguntas",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("TEXT_EMBEDDING_MODEL", DEFAULT_TEXT_EMBEDDING_MODEL),
        help="Modelo de embeddings local para RAGAS testset generation",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("EVALUATION_GENERATION_MAX_WORKERS", str(DEFAULT_MAX_WORKERS))),
        help="Workers paralelos de RAGAS (1 = secuencial, más estable en CPU/MPS)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("EVALUATION_GENERATION_TIMEOUT", str(DEFAULT_TIMEOUT))),
        help="Timeout en segundos para el generador",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("EVALUATION_GENERATION_LANGUAGE", DEFAULT_LANGUAGE),
        help="Idioma de salida de question y ground_truth (ISO 639-1, ej: es, en)",
    )
    parser.add_argument(
        "--input",
        help="Modo traducción: JSON con preguntas a traducir a --language",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    llm = ChatOpenAI(model_name=args.model, temperature=0)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {args.input}")
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("samples", data.get("questions", []))
        logger.info(
            "Traduciendo %d muestras de %s al idioma %s",
            len(data),
            args.input,
            args.language,
        )
        translated = _translate_samples(data, args.language, llm)
        output_path.write_text(
            json.dumps(translated, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Escritas %d preguntas traducidas en %s",
            len(translated),
            output_path,
        )
        return

    documents = load_corpus_segments(args.dataset_path, args.max_segments)
    logger.info("Cargados %d documentos del corpus en %s", len(documents), args.dataset_path)
    if not documents:
        raise ValueError("No se encontraron documentos de texto en el corpus")

    logger.info(
        "Generando %d preguntas sintéticas con el modelo %s y %d workers",
        args.size,
        args.model,
        args.max_workers,
    )
    # Forzar CPU para evitar segfaults/out-of-memory con MPS en Apple Silicon.
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": "cpu"},
    )
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)

    run_config = RunConfig(timeout=args.timeout, max_workers=args.max_workers)
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=args.size,
        run_config=run_config,
        raise_exceptions=True,
    )

    samples = _testset_to_eval_samples(testset)
    if args.language != "en":
        samples = _translate_samples(samples, args.language, llm)
    output_path.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Escritas %d preguntas sintéticas en %s", len(samples), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
