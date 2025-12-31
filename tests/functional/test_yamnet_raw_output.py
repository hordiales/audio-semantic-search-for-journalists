#!/usr/bin/env python3
"""
Script para examinar directamente qué clases detecta YAMNet
"""

from pathlib import Path

import librosa
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


def load_yamnet_classes():
    """Carga las clases de YAMNet"""
    class_map_path = hub.resolve('https://tfhub.dev/google/yamnet/1') + '/assets/yamnet_class_map.csv'
    class_names = list(pd.read_csv(class_map_path)['display_name'])
    return class_names

def analyze_segment_yamnet_output():
    """Analiza directamente la salida de YAMNet para algunos segmentos"""

    # Cargar YAMNet
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_names = load_yamnet_classes()
    print(f"📊 YAMNet cargado con {len(class_names)} clases")

    # Cargar dataset para obtener archivos de audio
    df = pd.read_pickle('dataset/final/complete_dataset.pkl')

    # Analizar algunos segmentos
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        audio_file = Path("dataset/converted") / Path(row['source_file']).name

        if not audio_file.exists():
            print(f"❌ Archivo no encontrado: {audio_file}")
            continue

        print(f"\n🎵 ANÁLISIS SEGMENTO {idx}:")
        print(f"   📄 Texto: {row['text'][:50]}...")
        print(f"   📁 Archivo: {audio_file.name}")
        print(f"   ⏱️  Tiempo: {row['start_time']:.1f}s - {row['end_time']:.1f}s")

        try:
            # Cargar audio para el segmento específico
            audio, _sr = librosa.load(str(audio_file),
                                   offset=row['start_time'],
                                   duration=row['end_time'] - row['start_time'],
                                   sr=16000)

            print(f"   🎧 Audio cargado: {len(audio)} samples, {len(audio)/16000:.1f}s")

            # Procesar con YAMNet
            scores, _embeddings, _spectrogram = yamnet_model(audio)

            # Obtener scores promedio
            mean_scores = tf.reduce_mean(scores, axis=0)

            # Top 10 clases detectadas
            top_indices = tf.nn.top_k(mean_scores, k=10).indices
            top_scores = tf.nn.top_k(mean_scores, k=10).values

            print("   🏆 TOP 10 CLASES DETECTADAS:")
            for i, (idx_class, score) in enumerate(zip(top_indices.numpy(), top_scores.numpy(), strict=False)):
                class_name = class_names[idx_class]
                print(f"      {i+1:2d}. {class_name:30} {score:.4f}")

            # Buscar específicamente clases relacionadas con aplausos/risas
            print("   🔍 CLASES DE INTERÉS:")
            interest_keywords = [
                'applause', 'clap', 'laughter', 'laugh', 'cheer', 'crowd',
                'audience', 'shout', 'yell', 'boo', 'giggle', 'chuckle'
            ]

            found_interesting = False
            for i, class_name in enumerate(class_names):
                score = mean_scores[i].numpy()
                if any(keyword in class_name.lower() for keyword in interest_keywords):
                    if score > 0.01:  # Mostrar incluso scores bajos
                        print(f"      🎯 {class_name:30} {score:.4f}")
                        found_interesting = True

            if not found_interesting:
                print("      ❌ Sin clases de interés detectadas con score > 0.01")

        except Exception as e:
            print(f"   ❌ Error procesando audio: {e}")

if __name__ == "__main__":
    analyze_segment_yamnet_output()
