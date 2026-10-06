"""Screen the full corpus with YAMNet for the events targeted by the CLAP acoustic questions.

YAMNet runs continuously over every converted recording, including audio between
transcribed segments. Its scores are independent of CLAP and serve as screening
evidence of whether a target event exists in the corpus; they are not ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16_000
FRAME_HOP_SECONDS = 0.48
STRONG_THRESHOLD = 0.5
WEAK_THRESHOLD = 0.2

# None marks queries whose target (tone, dropouts, interpretation...) has no AudioSet class.
QUERY_EVENT_CLASSES: dict[str, list[str] | None] = {
    "voice_calm_01": None,
    "voice_assertive_01": None,
    "voice_angry_01": None,
    "voice_bad_mood_01": None,
    "voice_timid_01": None,
    "voice_nervous_01": None,
    "voice_fearful_01": None,
    "voice_sad_01": None,
    "voice_excited_01": None,
    "voice_sarcastic_01": None,
    "voice_whisper_01": ["Whispering"],
    "voice_shout_01": ["Shout", "Yell"],
    "crowd_applause_end_01": ["Applause"],
    "crowd_applause_interrupt_01": ["Applause"],
    "crowd_cheering_01": ["Cheering"],
    "crowd_booing_01": None,
    "crowd_chant_01": ["Chant"],
    "crowd_murmur_01": ["Hubbub, speech noise, speech babble"],
    "crowd_laughter_01": ["Laughter", "Belly laugh"],
    "crowd_crying_01": ["Crying, sobbing"],
    "crowd_argument_01": ["Hubbub, speech noise, speech babble", "Shout"],
    "crowd_overlap_01": ["Hubbub, speech noise, speech babble"],
    "impact_door_knock_01": ["Knock"],
    "impact_table_hit_01": ["Slap, smack", "Thump, thud", "Knock"],
    "impact_repeated_bangs_01": ["Bang", "Thump, thud"],
    "impact_glass_break_01": ["Glass", "Shatter"],
    "impact_gunshots_01": ["Gunshot, gunfire"],
    "impact_explosion_01": ["Explosion"],
    "impact_door_slam_01": ["Slam", "Door"],
    "impact_running_footsteps_01": ["Run", "Walk, footsteps"],
    "impact_object_fall_01": ["Thump, thud"],
    "impact_scuffle_01": ["Thump, thud", "Slap, smack"],
    "environment_traffic_01": ["Traffic noise, roadway noise", "Vehicle", "Motor vehicle (road)"],
    "environment_siren_01": [
        "Siren",
        "Police car (siren)",
        "Ambulance (siren)",
        "Fire engine, fire truck (siren)",
        "Emergency vehicle",
    ],
    "environment_helicopter_01": ["Helicopter"],
    "environment_protest_01": ["Crowd", "Chant"],
    "environment_wind_01": ["Wind", "Wind noise (microphone)"],
    "environment_rain_thunder_01": ["Rain", "Rain on surface", "Thunder", "Thunderstorm"],
    "environment_construction_01": ["Jackhammer", "Drill", "Power tool"],
    "environment_machinery_01": ["Mechanisms", "Engine", "Heavy engine (low frequency)"],
    "environment_indoor_echo_01": ["Echo", "Reverberation"],
    "environment_aircraft_01": [
        "Aircraft",
        "Fixed-wing aircraft, airplane",
        "Aircraft engine",
        "Jet engine",
    ],
    "environment_train_01": ["Train", "Rail transport", "Railroad car, train wagon"],
    "environment_animals_01": ["Dog", "Bark"],
    "environment_water_01": ["Water", "Stream", "Waterfall"],
    "quality_static_01": ["Static"],
    "quality_hiss_01": ["Hiss", "White noise"],
    "quality_clipping_01": ["Distortion"],
    "quality_low_volume_01": None,
    "quality_mic_handling_01": None,
    "quality_dropouts_01": None,
    "quality_hum_01": ["Hum", "Mains hum"],
    "quality_background_music_01": ["Music", "Background music"],
    "quality_broadcast_bleed_01": ["Television", "Radio"],
    "quality_feedback_01": None,
    "workflow_phone_ring_01": ["Telephone bell ringing", "Ringtone", "Telephone"],
    "workflow_camera_01": ["Camera", "Single-lens reflex camera"],
    "workflow_police_radio_01": None,
    "workflow_typing_01": ["Typing", "Computer keyboard"],
    "workflow_papers_01": ["Rustle"],
    "workflow_cough_01": ["Cough"],
    "workflow_translation_01": None,
    "workflow_censor_beep_01": ["Beep, bleep"],
    "workflow_long_pause_01": ["Silence"],
    "workflow_alarm_01": [
        "Alarm",
        "Alarm clock",
        "Smoke detector, smoke alarm",
        "Fire alarm",
        "Car alarm",
        "Buzzer",
    ],
}


def verdict(max_score: float) -> str:
    if max_score >= STRONG_THRESHOLD:
        return "strong_evidence"
    if max_score >= WEAK_THRESHOLD:
        return "weak_evidence"
    return "no_evidence"


def _load_yamnet():
    import tensorflow as tf
    import tensorflow_hub as hub

    model = hub.load(YAMNET_MODEL_URL)
    with tf.io.gfile.GFile(model.class_map_path().numpy().decode("utf-8")) as class_map:
        names = [row["display_name"] for row in csv.DictReader(class_map)]
    return tf, model, names


def _frame_scores(tf, model, path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if sample_rate != YAMNET_SAMPLE_RATE:
        raise ValueError(f"YAMNet requires {YAMNET_SAMPLE_RATE} Hz audio; got {sample_rate} Hz")
    scores, _, _ = model(tf.convert_to_tensor(waveform, dtype=tf.float32))
    return np.asarray(scores.numpy())


def _segment_at(segments: pd.DataFrame, file_name: str, seconds: float) -> int | None:
    rows = segments[
        (segments["original_file_name"] == file_name)
        & (segments["start_time"] <= seconds + 0.96)
        & (segments["end_time"] >= seconds)
    ]
    return int(rows.iloc[0]["segment_id"]) if not rows.empty else None


def _human_evidence(review_set: dict, annotation_data: dict) -> dict[str, dict]:
    present: dict[tuple[str, int], list[bool | None]] = defaultdict(list)
    for annotation in annotation_data.get("annotations", []):
        present[(annotation["eval_case_id"], int(annotation["segment_id"]))].append(
            annotation.get("event_present")
        )
    evidence = {}
    for case in review_set.get("cases", []):
        values = [
            value
            for candidate in case["candidates"]
            for value in present.get((case["eval_case_id"], int(candidate["segment"]["segment_id"])), [])
        ]
        evidence[case["eval_case_id"]] = {
            "candidates": len(case["candidates"]),
            "event_present_true": sum(value is True for value in values),
            "event_present_false": sum(value is False for value in values),
            "event_present_unknown": sum(value is None for value in values),
        }
    return evidence


def audit_corpus(
    dataset_path: Path, questions_path: Path, review_set_path: Path, annotations_path: Path
) -> dict:
    tf, model, class_names = _load_yamnet()
    class_index = {name: index for index, name in enumerate(class_names)}
    unknown = sorted(
        {name for names in QUERY_EVENT_CLASSES.values() if names for name in names} - set(class_index)
    )
    if unknown:
        raise ValueError(f"Unknown AudioSet class names in mapping: {unknown}")

    segments = pd.read_pickle(dataset_path / "final" / "complete_dataset.pkl")
    recordings = sorted((dataset_path / "converted").glob("*.wav"))
    per_file = []
    for path in recordings:
        per_file.append((path.name, _frame_scores(tf, model, path)))

    all_scores = np.concatenate([scores for _, scores in per_file])
    top1 = Counter(class_names[int(index)] for index in all_scores.argmax(axis=1))
    mean_scores = all_scores.mean(axis=0)
    corpus_profile = {
        "recordings": [
            {
                "file_name": name,
                "duration_minutes": round(len(scores) * FRAME_HOP_SECONDS / 60, 1),
                "segments": int((segments["original_file_name"] == name).sum()),
            }
            for name, scores in per_file
        ],
        "frames": int(len(all_scores)),
        "top1_class_share": {
            name: round(count / len(all_scores), 4) for name, count in top1.most_common(10)
        },
        "highest_mean_classes": {
            class_names[int(index)]: round(float(mean_scores[int(index)]), 4)
            for index in np.argsort(mean_scores)[::-1][:10]
        },
    }

    def class_summary(name: str) -> dict:
        index = class_index[name]
        best_score, best_file, best_frame = -1.0, None, 0
        frames_weak = frames_strong = 0
        for file_name, scores in per_file:
            column = scores[:, index]
            frames_weak += int((column >= WEAK_THRESHOLD).sum())
            frames_strong += int((column >= STRONG_THRESHOLD).sum())
            frame = int(column.argmax())
            if float(column[frame]) > best_score:
                best_score, best_file, best_frame = float(column[frame]), file_name, frame
        seconds = best_frame * FRAME_HOP_SECONDS
        return {
            "class_name": name,
            "max_score": round(best_score, 4),
            "frames_at_or_above_weak": frames_weak,
            "frames_at_or_above_strong": frames_strong,
            "max_file": best_file,
            "max_time_seconds": round(seconds, 2),
            "max_segment_id": _segment_at(segments, best_file, seconds),
        }

    review_set = json.loads(review_set_path.read_text())
    human = _human_evidence(review_set, json.loads(annotations_path.read_text()))
    questions = json.loads(questions_path.read_text())["questions"]
    queries = []
    for question in questions:
        case_id = question["eval_case_id"]
        names = QUERY_EVENT_CLASSES.get(case_id)
        classes = [class_summary(name) for name in names] if names else []
        max_score = max((item["max_score"] for item in classes), default=None)
        queries.append(
            {
                "eval_case_id": case_id,
                "category": question.get("category", "unknown"),
                "clap_query_en": question.get("clap_query_en"),
                "audioset_classes": classes,
                "yamnet_max_score": max_score,
                "yamnet_verdict": verdict(max_score) if max_score is not None else "not_applicable",
                "human": human.get(case_id),
            }
        )

    verdicts = Counter(query["yamnet_verdict"] for query in queries)
    by_category: dict[str, Counter] = defaultdict(Counter)
    for query in queries:
        by_category[query["category"]][query["yamnet_verdict"]] += 1
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "method": {
            "model": YAMNET_MODEL_URL,
            "scope": "continuous frames over every converted recording, not only transcribed segments",
            "frame_hop_seconds": FRAME_HOP_SECONDS,
            "strong_threshold": STRONG_THRESHOLD,
            "weak_threshold": WEAK_THRESHOLD,
            "warning": (
                "YAMNet scores are independent screening evidence, not ground truth. "
                "no_evidence means no frame reached the weak threshold; it does not prove absence."
            ),
        },
        "corpus_profile": corpus_profile,
        "summary": {
            "queries": len(queries),
            "verdicts": dict(sorted(verdicts.items())),
            "by_category": {key: dict(sorted(value.items())) for key, value in sorted(by_category.items())},
        },
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--questions",
        default="evaluation/test_datasets/synthetic/clap_journalistic_acoustic_questions.json",
    )
    parser.add_argument("--review-set", default="evaluation/results/clap_all_questions_candidates.json")
    parser.add_argument("--annotations", default="evaluation/annotations/clap_human_annotations.json")
    parser.add_argument("--output", default="evaluation/results/corpus_acoustic_event_audit.json")
    args = parser.parse_args()
    dataset_path = args.dataset_path or os.getenv("DATASET_PATH")
    if not dataset_path:
        parser.error("--dataset-path is required when DATASET_PATH is not defined in .env")

    result = audit_corpus(
        Path(dataset_path), Path(args.questions), Path(args.review_set), Path(args.annotations)
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
