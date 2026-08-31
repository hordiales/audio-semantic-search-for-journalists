# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deterministic contract tests for the ADK entrypoint."""


def test_adk_entrypoint_exposes_the_audio_tools() -> None:
    """The production entrypoint must expose the intended ADK function tools."""
    from src.agent import app, root_agent

    assert app.root_agent is root_agent
    assert root_agent.name == "audio_search_agent"
    assert {tool.__name__ for tool in root_agent.tools} == {
        "buscar_audio",
        "buscar_evento_acustico",
        "buscar_clase_audio",
        "obtener_info_segmento",
        "obtener_clases_audio",
    }


def test_evidence_extraction_keeps_only_retrieval_fields() -> None:
    from src.agent_service.agent import AudioAgent

    contexts, segments = AudioAgent._extract_evidence(
        {
            "results": [
                {
                    "segment_id": 42,
                    "text": "El público aplaudió al final.",
                    "original_file_name": "discurso.wav",
                    "start_time": 12.0,
                    "end_time": 17.0,
                    "clip_url": "https://signed.example/segment_42.opus",
                    "clip_start_time": 7.0,
                    "clip_end_time": 22.0,
                }
            ]
        }
    )

    assert contexts == ["El público aplaudió al final."]
    assert segments == [
        {
            "segment_id": 42,
            "original_file_name": "discurso.wav",
            "start_time": 12.0,
            "end_time": 17.0,
            "text": "El público aplaudió al final.",
            "clip_url": "https://signed.example/segment_42.opus",
            "clip_start_time": 7.0,
            "clip_end_time": 22.0,
        }
    ]


def test_serialized_results_identify_the_search_index() -> None:
    from src.agent_service.tools import _serialize_results

    results = _serialize_results(
        [
            {
                "segment": {
                    "segment_id": 42,
                    "text": "El público aplaudió al final.",
                    "start_time": 12.0,
                    "end_time": 17.0,
                    "original_file_name": "discurso.wav",
                    "language": "es",
                    "confidence": 0.98,
                    "clip_url": "https://signed.example/segment_42.opus",
                    "clip_start_time": 7.0,
                    "clip_end_time": 22.0,
                    "clip_expires_at": "2026-08-29T16:30:00+00:00",
                    "yamnet_audio_classes": [
                        {"class_id": "/m/028ght", "class_name": "Applause", "score": 0.82}
                    ],
                },
                "similarity": 0.8,
            }
        ],
        search_index="audio",
        search_index_label="Índice de audio (CLAP)",
    )

    assert results[0]["search_index"] == "audio"
    assert results[0]["search_index_label"] == "Índice de audio (CLAP)"
    assert results[0]["clip_url"] == "https://signed.example/segment_42.opus"
    assert results[0]["clip_start_time"] == 7.0
    assert results[0]["clip_end_time"] == 22.0
    assert results[0]["clip_expires_at"] == "2026-08-29T16:30:00+00:00"
    assert results[0]["yamnet_audio_classes"][0]["class_name"] == "Applause"
