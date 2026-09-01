"""FastAPI backend for blind human grading of CLAP retrieval candidates."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_REVIEW_SET = PROJECT_ROOT / "evaluation/results/clap_human_review_candidates.json"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "evaluation/annotations/clap_human_annotations.json"


class AnnotationInput(BaseModel):
    eval_case_id: str
    segment_id: int
    relevance: int = Field(ge=0, le=3)
    event_present: bool | None = None
    confidence: int = Field(default=3, ge=1, le=5)
    notes: str = Field(default="", max_length=2000)
    reviewer: str = Field(default="anonymous", max_length=200)


class AnnotationStore:
    """Small atomic JSON store suitable for one local reviewer session."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    @staticmethod
    def _key(annotation: dict) -> str:
        return (
            f"{annotation.get('reviewer', 'anonymous')}:"
            f"{annotation['eval_case_id']}:{annotation['segment_id']}"
        )

    def load(self) -> dict:
        if not self.path.exists():
            return {
                "schema_version": "1.0",
                "updated_at": None,
                "annotations": [],
            }
        return json.loads(self.path.read_text())

    def upsert(self, annotation: dict) -> dict:
        with self._lock:
            data = self.load()
            annotations = {
                self._key(existing): existing for existing in data.get("annotations", [])
            }
            annotations[self._key(annotation)] = annotation
            data["updated_at"] = datetime.now(UTC).isoformat()
            data["annotations"] = sorted(
                annotations.values(),
                key=lambda item: (item["eval_case_id"], item["segment_id"]),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
            temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            os.replace(temporary, self.path)
            return annotation


def _load_review_set(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Review set not found at {path}. Run evaluation.prepare_acoustic_human_review."
        )
    return json.loads(path.read_text())


def create_app(
    *,
    review_set_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
) -> FastAPI:
    review_path = Path(review_set_path or os.getenv("HUMAN_REVIEW_SET", str(DEFAULT_REVIEW_SET)))
    corpus_path = Path(dataset_path or os.getenv("DATASET_PATH", "./dataset"))
    store = AnnotationStore(
        annotations_path or os.getenv("HUMAN_REVIEW_ANNOTATIONS", str(DEFAULT_ANNOTATIONS))
    )
    review_set = _load_review_set(review_path)
    candidate_pairs = {
        (case["eval_case_id"], int(candidate["segment"]["segment_id"]))
        for case in review_set.get("cases", [])
        for candidate in case.get("candidates", [])
    }
    candidate_ids = {segment_id for _, segment_id in candidate_pairs}

    app = FastAPI(title="Revisión humana de retrieval acústico", version="1.0.0")

    @app.get("/api/review-set")
    def get_review_set() -> dict:
        return {**review_set, "saved_annotations": store.load().get("annotations", [])}

    @app.post("/api/annotations")
    def save_annotation(payload: AnnotationInput) -> dict:
        if (payload.eval_case_id, payload.segment_id) not in candidate_pairs:
            raise HTTPException(status_code=404, detail="Candidate is not part of this review set")
        annotation = payload.model_dump()
        annotation["annotated_at"] = datetime.now(UTC).isoformat()
        store.upsert(annotation)
        return {"saved": True, "annotation": annotation}

    @app.get("/api/export")
    def export_annotations() -> FileResponse:
        if not store.path.exists():
            raise HTTPException(status_code=404, detail="No annotations have been saved")
        return FileResponse(
            store.path,
            media_type="application/json",
            filename="clap_human_annotations.json",
        )

    @app.get("/api/audio/{segment_id}")
    def get_audio(segment_id: int) -> FileResponse:
        if segment_id not in candidate_ids:
            raise HTTPException(status_code=404, detail="Unknown review candidate")
        clip_path = corpus_path / "segment_clips" / f"segment_{segment_id}.opus"
        if not clip_path.is_file():
            raise HTTPException(status_code=404, detail="Audio clip is unavailable")
        return FileResponse(clip_path, media_type="audio/ogg", filename=clip_path.name)

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="review-assets")

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


try:
    app = create_app()
except FileNotFoundError as startup_error:
    startup_error_message = str(startup_error)
    app = FastAPI(title="Revisión humana de retrieval acústico", version="1.0.0")

    @app.get("/", response_class=PlainTextResponse)
    def missing_review_set() -> PlainTextResponse:
        return PlainTextResponse(startup_error_message, status_code=503)
