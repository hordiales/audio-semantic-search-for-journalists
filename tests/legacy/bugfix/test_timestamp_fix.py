#!/usr/bin/env python3
"""
Script para probar la corrección de timestamps en la segmentación por silencio
"""

import os
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import SRC_ROOT, ensure_sys_path

ensure_sys_path([SRC_ROOT])


from audio_transcription import AudioTranscriber


def test_timestamp_fix():
    print("🧪 Probando la corrección de timestamps en segmentación por silencio\n")

    # Inicializar transcriptor
    transcriber = AudioTranscriber(model_name="base")

    # Archivo de prueba
    audio_file = "data/en.20081117.21.1-060.m4a"

    if not Path(audio_file).exists():
        audio_file = "dataset/converted/en.20081117.21.1-060.wav"

    if not Path(audio_file).exists():
        print("❌ No se encontró archivo de audio para la prueba")
        return

    print(f"📁 Usando archivo: {audio_file}")

    # Hacer segmentación con la nueva función
    print("\n🔄 Ejecutando segmentación por silencio (método corregido)...")
    segments = transcriber.segment_by_silence(
        audio_file,
        min_silence_len=500,
        silence_thresh=-40
    )

    print("\n📊 Resultados:")
    print(f"   Segmentos encontrados: {len(segments)}")

    if len(segments) > 0:
        print(f"   Primer segmento: {segments[0]['start_time']:.3f}s - {segments[0]['end_time']:.3f}s")
        print(f"   Último segmento: {segments[-1]['start_time']:.3f}s - {segments[-1]['end_time']:.3f}s")

        # Mostrar primeros 5 segmentos para verificar
        print("\n📋 Primeros 5 segmentos:")
        print(f"{'ID':<3} {'Start':<8} {'End':<8} {'Duration':<8} {'Gap':<8}")
        print("-" * 45)

        prev_end = 0
        for i, seg in enumerate(segments[:5]):
            gap = seg['start_time'] - prev_end if i > 0 else 0
            print(f"{seg['segment_id']:<3} {seg['start_time']:<8.3f} {seg['end_time']:<8.3f} {seg['duration']:<8.3f} {gap:<8.3f}")
            prev_end = seg['end_time']

        # Verificar que los timestamps son crecientes y no hay overlaps
        print("\n✅ Verificaciones:")
        timestamps_ok = True
        for i in range(1, len(segments)):
            if segments[i]['start_time'] < segments[i-1]['end_time']:
                print(f"❌ Overlap detectado entre segmentos {i-1} y {i}")
                timestamps_ok = False

        if timestamps_ok:
            print("✅ Timestamps correctos - sin overlaps")

        # Verificar que los gaps son razonables (representan silencios)
        gaps = []
        for i in range(1, len(segments)):
            gap = segments[i]['start_time'] - segments[i-1]['end_time']
            gaps.append(gap)

        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"✅ Gap promedio entre segmentos: {avg_gap:.3f}s (representa silencios removidos)")

        # Limpiar archivos temporales
        print("\n🧹 Limpiando archivos temporales...")
        for seg in segments:
            temp_file = seg['temp_path']
            if os.path.exists(temp_file):
                os.remove(temp_file)

        print("✅ Limpieza completada")

    print("\n🎯 Prueba completada!")

if __name__ == "__main__":
    test_timestamp_fix()
