from src.agent_service.search_engine import AudioSearchEngine
from src.yamnet_audio_classifier import aggregate_yamnet_classes


def test_aggregate_yamnet_classes_keeps_peak_score_from_each_window():
    classes = aggregate_yamnet_classes(
        [
            [
                {"class_id": "/m/applause", "class_name": "Applause", "score": 0.43},
                {"class_id": "/m/speech", "class_name": "Speech", "score": 0.81},
            ],
            [
                {"class_id": "/m/applause", "class_name": "Applause", "score": 0.91},
                {"class_id": "/m/music", "class_name": "Music", "score": 0.52},
            ],
        ],
        top_k=2,
    )

    assert classes == [
        {"class_id": "/m/applause", "class_name": "Applause", "score": 0.91},
        {"class_id": "/m/speech", "class_name": "Speech", "score": 0.81},
    ]


def test_parse_audio_classes_accepts_dataset_json_and_skips_invalid_values():
    parsed = AudioSearchEngine._parse_audio_classes(
        '[{"class_id":"/m/applause","class_name":"Applause","score":0.91}, {"bad": true}]'
    )

    assert parsed == [{"class_id": "/m/applause", "class_name": "Applause", "score": 0.91}]
