#!/usr/bin/env python3
"""
Test simple de generación de embeddings de audio
Genera embeddings localmente y los compara sin usar la base de datos
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import artifacts_dir

RESULTS_ARTIFACTS = artifacts_dir("audio_embeddings_simple")

# Configurar warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class EmbeddingResult:
    """Resultado de generar embedding"""
    success: bool
    embedding: np.ndarray | None = None
    dimension: int = 0
    confidence: float = 0.0
    processing_time_ms: int = 0
    error_message: str = ""
    model_name: str = ""

class SimpleAudioEmbeddingTester:
    """Test simple de embeddings de audio"""

    def __init__(self):
        self.audio_folder = Path("dataset")
        self._loaded_models = {}

    def _load_yamnet_model(self):
        """Cargar modelo YAMNet"""
        if 'yamnet' in self._loaded_models:
            return self._loaded_models['yamnet']

        try:
            import tensorflow_hub as hub

            print("🔄 Cargando YAMNet...")
            model = hub.load("https://tfhub.dev/google/yamnet/1")
            self._loaded_models['yamnet'] = model
            print("✅ YAMNet cargado")
            return model

        except Exception as e:
            print(f"❌ Error cargando YAMNet: {e}")
            print("💡 Instala: pip install tensorflow tensorflow-hub")
            return None

    def _load_clap_model(self):
        """Cargar modelo CLAP"""
        if 'clap' in self._loaded_models:
            return self._loaded_models['clap']

        try:
            import laion_clap

            print("🔄 Cargando CLAP...")
            model = laion_clap.CLAP_Module(enable_fusion=False)
            model.load_ckpt()
            self._loaded_models['clap'] = model
            print("✅ CLAP cargado")
            return model

        except Exception as e:
            print(f"❌ Error cargando CLAP: {e}")
            print("💡 Instala: pip install laion-clap")
            return None

    def generate_yamnet_embedding(self, audio_path: str) -> EmbeddingResult:
        """Generar embedding con YAMNet"""
        start_time = time.time()

        try:
            model = self._load_yamnet_model()
            if model is None:
                return EmbeddingResult(
                    success=False,
                    error_message="YAMNet no disponible",
                    model_name="yamnet"
                )

            import librosa
            import tensorflow as tf

            # Cargar audio
            audio, _sr = librosa.load(audio_path, sr=16000, mono=True)

            # Convertir a tensor
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

            # Generar embeddings
            _, embeddings, _ = model(audio_tensor)

            # Promedio de los embeddings temporales
            avg_embedding = tf.reduce_mean(embeddings, axis=0)
            embedding_np = avg_embedding.numpy()

            processing_time = int((time.time() - start_time) * 1000)

            return EmbeddingResult(
                success=True,
                embedding=embedding_np,
                dimension=len(embedding_np),
                confidence=0.8,
                processing_time_ms=processing_time,
                model_name="yamnet"
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            return EmbeddingResult(
                success=False,
                processing_time_ms=processing_time,
                error_message=f"Error YAMNet: {e!s}",
                model_name="yamnet"
            )

    def generate_clap_embedding(self, audio_path: str) -> EmbeddingResult:
        """Generar embedding con CLAP"""
        start_time = time.time()

        try:
            model = self._load_clap_model()
            if model is None:
                return EmbeddingResult(
                    success=False,
                    error_message="CLAP no disponible",
                    model_name="clap_laion"
                )

            # Generar embedding de audio
            audio_embed = model.get_audio_embedding_from_filelist([audio_path], use_tensor=False)
            embedding_np = audio_embed[0]

            processing_time = int((time.time() - start_time) * 1000)

            return EmbeddingResult(
                success=True,
                embedding=embedding_np,
                dimension=len(embedding_np),
                confidence=0.9,
                processing_time_ms=processing_time,
                model_name="clap_laion"
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            return EmbeddingResult(
                success=False,
                processing_time_ms=processing_time,
                error_message=f"Error CLAP: {e!s}",
                model_name="clap_laion"
            )

    def find_audio_files(self) -> list[str]:
        """Encontrar archivos de audio"""
        search_paths = [
            self.audio_folder,
            self.audio_folder / "converted",
            self.audio_folder / "final"
        ]

        audio_files = []
        for base_path in search_paths:
            if base_path.exists():
                for ext in ['*.wav', '*.mp3', '*.m4a']:
                    audio_files.extend(base_path.glob(ext))

        return [str(f) for f in audio_files[:5]]  # Máximo 5 archivos

    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calcular similitud coseno entre embeddings"""
        try:
            # Normalizar embeddings
            norm1 = embedding1 / np.linalg.norm(embedding1)
            norm2 = embedding2 / np.linalg.norm(embedding2)

            # Similitud coseno
            similarity = np.dot(norm1, norm2)
            return float(similarity)

        except Exception as e:
            print(f"⚠️  Error calculando similitud: {e}")
            return 0.0

    def run_test(self):
        """Ejecutar test de embeddings"""
        print("🚀 TEST DE EMBEDDINGS DE AUDIO")
        print("=" * 50)

        # Encontrar archivos de audio
        audio_files = self.find_audio_files()

        if not audio_files:
            print("❌ No se encontraron archivos de audio")
            print("💡 Verifica que existan archivos .wav, .mp3 o .m4a en dataset/")
            return

        print(f"📁 Encontrados {len(audio_files)} archivos de audio")

        results = []

        for i, audio_path in enumerate(audio_files, 1):
            filename = Path(audio_path).name
            print(f"\n🎵 Archivo {i}/{len(audio_files)}: {filename}")

            # Generar embeddings con ambos modelos
            yamnet_result = self.generate_yamnet_embedding(audio_path)
            clap_result = self.generate_clap_embedding(audio_path)

            # Mostrar resultados
            if yamnet_result.success:
                print(f"   ✅ YAMNet: {yamnet_result.dimension}D, {yamnet_result.processing_time_ms}ms")
            else:
                print(f"   ❌ YAMNet: {yamnet_result.error_message}")

            if clap_result.success:
                print(f"   ✅ CLAP: {clap_result.dimension}D, {clap_result.processing_time_ms}ms")
            else:
                print(f"   ❌ CLAP: {clap_result.error_message}")

            # Comparar embeddings si ambos fueron generados
            if yamnet_result.success and clap_result.success:
                # Como son de diferentes dimensiones, comparamos solo una muestra
                sample_size = min(yamnet_result.dimension, clap_result.dimension, 100)

                yamnet_sample = yamnet_result.embedding[:sample_size]
                clap_sample = clap_result.embedding[:sample_size]

                similarity = self.compare_embeddings(yamnet_sample, clap_sample)
                print(f"   🔄 Similitud YAMNet-CLAP (muestra): {similarity:.3f}")

            results.append({
                'file': filename,
                'yamnet': yamnet_result,
                'clap': clap_result
            })

        # Resumen final
        print("\n" + "=" * 50)
        print("📊 RESUMEN DE RESULTADOS")
        print("=" * 50)

        yamnet_success = sum(1 for r in results if r['yamnet'].success)
        clap_success = sum(1 for r in results if r['clap'].success)

        print(f"✅ YAMNet exitosos: {yamnet_success}/{len(results)}")
        print(f"✅ CLAP exitosos: {clap_success}/{len(results)}")

        if yamnet_success > 0:
            avg_yamnet_time = np.mean([r['yamnet'].processing_time_ms for r in results if r['yamnet'].success])
            avg_yamnet_dim = np.mean([r['yamnet'].dimension for r in results if r['yamnet'].success])
            print(f"📏 YAMNet promedio: {avg_yamnet_dim:.0f}D, {avg_yamnet_time:.0f}ms")

        if clap_success > 0:
            avg_clap_time = np.mean([r['clap'].processing_time_ms for r in results if r['clap'].success])
            avg_clap_dim = np.mean([r['clap'].dimension for r in results if r['clap'].success])
            print(f"📏 CLAP promedio: {avg_clap_dim:.0f}D, {avg_clap_time:.0f}ms")

        # Guardar resultados localmente
        output_file = RESULTS_ARTIFACTS / "embeddings_results.json"
        try:
            serializable_results = []
            for r in results:
                serializable_results.append({
                    'file': r['file'],
                    'yamnet': {
                        'success': r['yamnet'].success,
                        'dimension': r['yamnet'].dimension,
                        'processing_time_ms': r['yamnet'].processing_time_ms,
                        'error_message': r['yamnet'].error_message
                    },
                    'clap': {
                        'success': r['clap'].success,
                        'dimension': r['clap'].dimension,
                        'processing_time_ms': r['clap'].processing_time_ms,
                        'error_message': r['clap'].error_message
                    }
                })

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            print(f"💾 Resultados guardados en: {output_file}")

        except Exception as e:
            print(f"⚠️  Error guardando resultados: {e}")

def main():
    """Función principal"""
    try:
        tester = SimpleAudioEmbeddingTester()
        tester.run_test()

    except KeyboardInterrupt:
        print("\n⏹️  Test cancelado")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
