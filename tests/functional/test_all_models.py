#!/usr/bin/env python3
"""
Test completo de todos los modelos de embeddings disponibles
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

RESULTS_DIR = artifacts_dir("all_models")

# Configurar warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class ModelTestResult:
    """Resultado del test de un modelo"""
    model_name: str
    success: bool
    embedding_dimension: int = 0
    processing_time_ms: int = 0
    error_message: str = ""
    confidence: float = 0.0

class CompleteModelTester:
    """Test completo de todos los modelos disponibles"""

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
            return None

    def _load_speechdpr_model(self):
        """Cargar modelo SpeechDPR"""
        if 'speechdpr' in self._loaded_models:
            return self._loaded_models['speechdpr']

        try:
            from speechdpr_implementation import SpeechDPRModel

            print("🔄 Cargando SpeechDPR...")
            model = SpeechDPRModel()
            self._loaded_models['speechdpr'] = model
            print("✅ SpeechDPR cargado")
            return model

        except Exception as e:
            print(f"❌ Error cargando SpeechDPR: {e}")
            return None

    def test_yamnet(self, audio_path: str) -> ModelTestResult:
        """Test YAMNet"""
        start_time = time.time()

        try:
            model = self._load_yamnet_model()
            if model is None:
                return ModelTestResult(
                    model_name="yamnet",
                    success=False,
                    error_message="YAMNet no disponible"
                )

            import librosa
            import tensorflow as tf

            # Cargar audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

            # Generar embeddings
            _, embeddings, _ = model(audio_tensor)
            avg_embedding = tf.reduce_mean(embeddings, axis=0)
            embedding_np = avg_embedding.numpy()

            processing_time = int((time.time() - start_time) * 1000)

            return ModelTestResult(
                model_name="yamnet",
                success=True,
                embedding_dimension=len(embedding_np),
                processing_time_ms=processing_time,
                confidence=0.8
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            return ModelTestResult(
                model_name="yamnet",
                success=False,
                processing_time_ms=processing_time,
                error_message=f"Error YAMNet: {e!s}"
            )

    def test_clap(self, audio_path: str) -> ModelTestResult:
        """Test CLAP"""
        start_time = time.time()

        try:
            model = self._load_clap_model()
            if model is None:
                return ModelTestResult(
                    model_name="clap_laion",
                    success=False,
                    error_message="CLAP no disponible"
                )

            # Generar embedding de audio
            audio_embed = model.get_audio_embedding_from_filelist([audio_path], use_tensor=False)
            embedding_np = audio_embed[0]

            processing_time = int((time.time() - start_time) * 1000)

            return ModelTestResult(
                model_name="clap_laion",
                success=True,
                embedding_dimension=len(embedding_np),
                processing_time_ms=processing_time,
                confidence=0.9
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            return ModelTestResult(
                model_name="clap_laion",
                success=False,
                processing_time_ms=processing_time,
                error_message=f"Error CLAP: {e!s}"
            )

    def test_speechdpr(self, audio_path: str) -> ModelTestResult:
        """Test SpeechDPR"""
        start_time = time.time()

        try:
            model = self._load_speechdpr_model()
            if model is None:
                return ModelTestResult(
                    model_name="speechdpr",
                    success=False,
                    error_message="SpeechDPR no disponible"
                )

            # Generar embedding de audio
            embedding_np = model.generate_speech_embedding(audio_path)

            if embedding_np is None:
                return ModelTestResult(
                    model_name="speechdpr",
                    success=False,
                    error_message="Error generando embedding SpeechDPR"
                )

            processing_time = int((time.time() - start_time) * 1000)

            return ModelTestResult(
                model_name="speechdpr",
                success=True,
                embedding_dimension=len(embedding_np),
                processing_time_ms=processing_time,
                confidence=0.85
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            return ModelTestResult(
                model_name="speechdpr",
                success=False,
                processing_time_ms=processing_time,
                error_message=f"Error SpeechDPR: {e!s}"
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

        return [str(f) for f in audio_files[:2]]  # Máximo 2 archivos para test

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

    def run_complete_test(self):
        """Ejecutar test completo de todos los modelos"""
        print("🚀 TEST COMPLETO DE MODELOS DE EMBEDDINGS")
        print("=" * 60)

        # Encontrar archivos de audio
        audio_files = self.find_audio_files()

        if not audio_files:
            print("❌ No se encontraron archivos de audio")
            return

        print(f"📁 Encontrados {len(audio_files)} archivos de audio para test")

        # Modelos a probar
        models_to_test = [
            ("YAMNet", self.test_yamnet),
            ("CLAP", self.test_clap),
            ("SpeechDPR", self.test_speechdpr)
        ]

        all_results = []

        for audio_path in audio_files:
            filename = Path(audio_path).name
            print(f"\n🎵 Procesando: {filename}")
            print("-" * 50)

            file_results = {
                'file': filename,
                'results': {}
            }

            for model_name, test_function in models_to_test:
                print(f"   🔄 Probando {model_name}...")

                result = test_function(audio_path)

                if result.success:
                    print(f"   ✅ {model_name}: {result.embedding_dimension}D, {result.processing_time_ms}ms")
                else:
                    print(f"   ❌ {model_name}: {result.error_message}")

                file_results['results'][model_name.lower()] = result

            all_results.append(file_results)

        # Resumen final
        self.print_summary(all_results, models_to_test)

        # Guardar resultados
        self.save_results(all_results)

    def print_summary(self, all_results: list[dict], models_to_test: list[tuple]):
        """Imprimir resumen de resultados"""
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE RESULTADOS")
        print("=" * 60)

        # Estadísticas por modelo
        for model_name, _ in models_to_test:
            model_key = model_name.lower()
            successful_tests = []
            failed_tests = []

            for file_result in all_results:
                result = file_result['results'][model_key]
                if result.success:
                    successful_tests.append(result)
                else:
                    failed_tests.append(result)

            print(f"\n🎯 {model_name.upper()}:")
            print(f"   ✅ Exitosos: {len(successful_tests)}/{len(all_results)}")

            if successful_tests:
                avg_time = np.mean([r.processing_time_ms for r in successful_tests])
                avg_dim = np.mean([r.embedding_dimension for r in successful_tests])
                print(f"   📏 Dimensión promedio: {avg_dim:.0f}D")
                print(f"   ⏱️  Tiempo promedio: {avg_time:.0f}ms")

            if failed_tests:
                print(f"   ❌ Fallos: {len(failed_tests)}")
                for fail in failed_tests:
                    print(f"      - {fail.error_message}")

        # Comparación de rendimiento
        print("\n🏆 RANKING DE MODELOS (por tiempo de procesamiento):")
        print("-" * 40)

        successful_models = []
        for model_name, _ in models_to_test:
            model_key = model_name.lower()
            successful_tests = [
                file_result['results'][model_key]
                for file_result in all_results
                if file_result['results'][model_key].success
            ]

            if successful_tests:
                avg_time = np.mean([r.processing_time_ms for r in successful_tests])
                successful_models.append((model_name, avg_time))

        # Ordenar por tiempo (menor es mejor)
        successful_models.sort(key=lambda x: x[1])

        for i, (model_name, avg_time) in enumerate(successful_models, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"{emoji} {i}. {model_name}: {avg_time:.0f}ms")

    def save_results(self, all_results: list[dict]):
        """Guardar resultados en archivo JSON"""
        output_file = RESULTS_DIR / "complete_model_test_results.json"

        try:
            # Preparar datos serializables
            serializable_results = []
            for file_result in all_results:
                serializable_file_result = {
                    'file': file_result['file'],
                    'results': {}
                }

                for model_name, result in file_result['results'].items():
                    serializable_file_result['results'][model_name] = {
                        'model_name': result.model_name,
                        'success': result.success,
                        'embedding_dimension': result.embedding_dimension,
                        'processing_time_ms': result.processing_time_ms,
                        'error_message': result.error_message,
                        'confidence': result.confidence
                    }

                serializable_results.append(serializable_file_result)

            # Guardar
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            print(f"\n💾 Resultados guardados en: {output_file}")

        except Exception as e:
            print(f"⚠️  Error guardando resultados: {e}")

def main():
    """Función principal"""
    try:
        tester = CompleteModelTester()
        tester.run_complete_test()

    except KeyboardInterrupt:
        print("\n⏹️  Test cancelado")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
