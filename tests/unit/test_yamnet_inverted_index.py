import json

from src.yamnet_inverted_index import (
    build_yamnet_inverted_index,
    load_yamnet_inverted_index,
    write_yamnet_inverted_index,
)


def test_yamnet_inverted_index_persists_each_audio_class_under_its_tokens(tmp_path):
    rows = [
        {
            "segment_id": 7,
            "yamnet_top_classes": [
                {"class_id": "/m/applause", "class_name": "Applause", "score": 0.8},
                {"class_id": "/m/speech", "class_name": "Speech", "score": 0.9},
            ],
        }
    ]
    path = tmp_path / "yamnet_inverted_index.json"

    token_count = write_yamnet_inverted_index(path, rows)

    assert token_count == 2
    assert json.loads(path.read_text())["version"] == 1
    assert load_yamnet_inverted_index(path)["applause"] == [
        {
            "class_id": "/m/applause",
            "class_name": "Applause",
            "score": 0.8,
            "segment_id": 7,
            "class_rank": 0,
        }
    ]


def test_yamnet_inverted_index_indexes_multiword_audio_classes():
    index = build_yamnet_inverted_index(
        [
            {
                "segment_id": 8,
                "yamnet_top_classes": [
                    {"class_id": "/m/car", "class_name": "Motor vehicle (road)", "score": 0.7}
                ],
            }
        ]
    )

    assert set(index["postings"]) == {"motor", "vehicle", "road"}
