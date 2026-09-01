"""Genera un dataset balanceado de preguntas sintéticas: mitad conceptuales,
mitad acústicas/auditivas.

Objetivo: comparar cómo responde CLAP cuando las consultas sí mencionan eventos
sonoros vs. cuando solo preguntan por contenido lingüístico. Ver
`docs/evaluar-clap-clotho.md` y `docs/reporte-comparativo-texto-clap.md` §5.1.

El generador toma segmentos del corpus y, para cada uno, pide al LLM una
pregunta periodística de contenido y una pregunta que haga referencia explícita
a propiedades del audio (tono, ruido de fondo, música, interrupciones, calidad
de grabación, etc.). Ambas apuntan al mismo segmento, de modo que la única
variable es el tipo de query.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_OUTPUT = "./evaluation/test_datasets/synthetic/balanced_journalistic_questions.json"
DEFAULT_MAX_SEGMENTS = 20
DEFAULT_LANGUAGE = "es"

PROMPT_TEMPLATE = """A continuación te paso un segmento de audio transcrito de
un programa periodístico argentino. El segmento tiene id {segment_id}, va de
{start_time:.1f}s a {end_time:.1f}s del archivo "{original_file_name}".

Texto del segmento:
---
{text}
---

Generá EXACTAMENTE dos preguntas periodísticas sobre ESTE segmento. La pregunta
acústica debe ser distinta a las de otros segmentos y referirse a sonidos o
propiedades del audio que un oyente podría percibir.

1. "content": pregunta sobre el contenido, hechos, personas o relaciones
   políticas mencionadas. Debe mencionar detalles específicos del texto.
2. "acoustic": pregunta sobre una propiedad del audio o evento sonoro: tono de
   voz, ruido de fondo, música, aplausos, gritos, interrupciones, calidad del
   audio, ambiente, silencios, etc. Debe vincularse al tema específico del
   segmento (p. ej. "¿cómo es el tono cuando habla de X?", "¿se escucha algún
   ruido de fondo mientras discuten Y?", "¿hay cambios de tono al mencionar
   Z?"). NO repitas "Is there any background noise or interruptions present
   in the audio segment?" ni frases genéricas idénticas para todos los
   segmentos.

Para cada pregunta devolvé una respuesta corta (ground_truth). Respondé
ÚNICAMENTE con un objeto JSON con esta estructura exacta:

{{
  "content": {{
    "question": "...",
    "ground_truth": "..."
  }},
  "acoustic": {{
    "question": "...",
    "ground_truth": "..."
  }}
}}

El idioma de salida debe ser {language}.
"""


def load_segments(dataset_path: str, max_segments: int) -> list[dict]:
    corpus_pkl = Path(dataset_path) / "final" / "complete_dataset.pkl"
    if not corpus_pkl.exists():
        raise FileNotFoundError(f"No se encontró el corpus en {corpus_pkl}")

    df = pd.read_pickle(corpus_pkl)
    segments = []
    for _, row in df.iterrows():
        text = row.get("text")
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue
        segments.append(
            {
                "segment_id": int(row.get("segment_id", 0)),
                "start_time": float(row.get("start_time", 0.0)),
                "end_time": float(row.get("end_time", 0.0)),
                "original_file_name": str(row.get("original_file_name", "")),
                "text": text.strip(),
            }
        )

    if max_segments > 0:
        segments = segments[:max_segments]
    return segments


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0]
    return raw.strip()


def generate_for_segment(segment: dict, llm: ChatOpenAI, language: str) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(language=language, **segment)
    response = llm.invoke([HumanMessage(content=prompt)]).content
    try:
        parsed = json.loads(_strip_code_fences(response))
    except json.JSONDecodeError as error:
        logger.warning(
            "Respuesta no JSON para segmento %d: %s... Error: %s",
            segment["segment_id"],
            response[:100],
            error,
        )
        return None

    result: dict = {}
    for qtype in ("content", "acoustic"):
        entry = parsed.get(qtype)
        if not isinstance(entry, dict):
            logger.warning(
                "Estructura inesperada para %s en segmento %d", qtype, segment["segment_id"]
            )
            continue
        question = entry.get("question", "").strip()
        ground_truth = entry.get("ground_truth", "").strip()
        if not question or not ground_truth:
            logger.warning("Pregunta o ground_truth vacío en segmento %d", segment["segment_id"])
            continue
        result[qtype] = {
            "type": qtype,
            "question": question,
            "ground_truth": ground_truth,
            "ground_truth_contexts": [segment["text"]],
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "original_file_name": segment["original_file_name"],
        }
    return result


def generate_balanced_questions(
    dataset_path: str,
    max_segments: int,
    model: str,
    language: str,
) -> list[dict]:
    llm = ChatOpenAI(model_name=model, temperature=0)
    segments = load_segments(dataset_path, max_segments)
    logger.info("Generando preguntas para %d segmentos con %s", len(segments), model)

    samples: list[dict] = []
    for segment in segments:
        result = generate_for_segment(segment, llm, language)
        if result:
            for qtype in ("content", "acoustic"):
                if qtype in result:
                    samples.append(result[qtype])
        else:
            logger.warning("No se pudo generar preguntas para segmento %d", segment["segment_id"])

    return samples


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Genera preguntas periodísticas balanceadas (contenido + acústicas)"
    )
    parser.add_argument(
        "--dataset-path",
        default=os.getenv("DATASET_PATH", "./dataset"),
        help="Ruta al dataset procesado que contiene final/complete_dataset.pkl",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=int(os.getenv("EVALUATION_DATASET_MAX_SEGMENTS", str(DEFAULT_MAX_SEGMENTS))),
        help="Cantidad máxima de segmentos a usar (0 = todos)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Alias legado; se ignora, el tamaño queda determinado por --max-segments",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="Modelo de OpenAI para generar las preguntas",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("EVALUATION_GENERATION_LANGUAGE", DEFAULT_LANGUAGE),
        help="Idioma de salida de las preguntas (ISO 639-1)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Ruta de salida del JSON con preguntas balanceadas",
    )
    args = parser.parse_args()

    if args.size is not None:
        logger.warning(
            "--size no se usa en este generador; controlá la cantidad con --max-segments"
        )

    samples = generate_balanced_questions(
        dataset_path=args.dataset_path,
        max_segments=args.max_segments,
        model=args.model,
        language=args.language,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Escritas %d preguntas balanceadas en %s", len(samples), output_path)
    print(
        f"Generadas {len(samples)} preguntas ({len(samples) // 2} contenido + {len(samples) // 2} acústicas)"
    )
    print(f"Guardado en: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
