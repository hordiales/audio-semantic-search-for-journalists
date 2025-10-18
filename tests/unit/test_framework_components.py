#!/usr/bin/env python3
"""
Test de componentes individuales del framework sin cargar modelos pesados.
Verifica que cada componente funcione independientemente.
"""

import sys
import os
import logging
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import artifacts_dir, ensure_sys_path, SRC_ROOT

COMPONENTS_ARTIFACTS = artifacts_dir("components")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Verifica que todos los imports funcionen correctamente"""
    print("📦 Probando imports de módulos...")

    results = {}

    # Test import de generador de datos
    try:
        ensure_sys_path([SRC_ROOT])
        from test_data_generator import SyntheticTestDataGenerator
        results['test_data_generator'] = True
        print("✅ test_data_generator importado")
    except Exception as e:
        results['test_data_generator'] = False
        print(f"❌ test_data_generator: {e}")

    # Test import de visualization dashboard
    try:
        from visualization_dashboard import EmbeddingVisualizationDashboard
        results['visualization_dashboard'] = True
        print("✅ visualization_dashboard importado")
    except Exception as e:
        results['visualization_dashboard'] = False
        print(f"❌ visualization_dashboard: {e}")

    # Test import de embedding evaluation framework
    try:
        from embedding_evaluation_framework import EmbeddingBenchmark, TestCase, EvaluationMetrics
        results['embedding_evaluation_framework'] = True
        print("✅ embedding_evaluation_framework importado")
    except Exception as e:
        results['embedding_evaluation_framework'] = False
        print(f"❌ embedding_evaluation_framework: {e}")

    # Test import de models config
    try:
        from models_config import get_models_config, AudioEmbeddingModel
        results['models_config'] = True
        print("✅ models_config importado")
    except Exception as e:
        results['models_config'] = False
        print(f"❌ models_config: {e}")

    # Test import de audio embeddings (sin cargar modelos)
    try:
        from audio_embeddings import AudioEmbeddingGenerator
        results['audio_embeddings'] = True
        print("✅ audio_embeddings importado")
    except Exception as e:
        results['audio_embeddings'] = False
        print(f"❌ audio_embeddings: {e}")

    return results

def test_data_generation_only():
    """Prueba solo la generación de datos sin modelos"""
    print("🔧 Probando generación de datos...")

    try:
        ensure_sys_path([SRC_ROOT])
        from test_data_generator import SyntheticTestDataGenerator

        # Crear generador
        generator = SyntheticTestDataGenerator(str(COMPONENTS_ARTIFACTS))

        # Generar dataset muy pequeño
        df = generator.generate_test_dataset(num_samples=5)

        # Verificaciones básicas
        assert len(df) == 5
        assert 'text' in df.columns
        assert 'category' in df.columns

        print(f"✅ Dataset generado: {len(df)} muestras")

        # Generar consultas
        queries = generator.create_ground_truth_queries(df, num_queries=2)
        assert len(queries) == 2

        print(f"✅ Consultas generadas: {len(queries)}")
        return True

    except Exception as e:
        print(f"❌ Error en generación de datos: {e}")
        return False

def test_metrics_calculations():
    """Prueba cálculos de métricas sin modelos"""
    print("📊 Probando cálculos de métricas...")

    success_count = 0
    total_tests = 0

    # Test BERTScore si está disponible
    total_tests += 1
    try:
        from bert_score import score as bert_score

        predictions = ["El gobierno anunció nuevas medidas", "Los mercados reaccionaron positivamente"]
        references = ["El estado presentó políticas nuevas", "Los mercados tuvieron reacción favorable"]

        P, R, F1 = bert_score(predictions, references, lang="es", verbose=False)

        print(f"✅ BERTScore: P={P.mean():.3f}, R={R.mean():.3f}, F1={F1.mean():.3f}")
        success_count += 1

    except ImportError:
        print("⚠️  BERTScore no disponible")
    except Exception as e:
        print(f"❌ Error en BERTScore: {e}")

    # Test Sentence Transformers
    total_tests += 1
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        texts = ["política económica", "medidas económicas", "tecnología digital"]
        embeddings = model.encode(texts)
        similarities = cosine_similarity(embeddings)

        print(f"✅ Sentence Transformers: similitud calculada {similarities[0,1]:.3f}")
        success_count += 1

    except ImportError:
        print("⚠️  Sentence Transformers no disponible")
    except Exception as e:
        print(f"❌ Error en Sentence Transformers: {e}")

    # Test métricas básicas de recuperación
    total_tests += 1
    try:
        # Simular resultados
        ground_truth = [1, 3, 5]
        top_5_results = [1, 2, 3, 4, 5]

        precision_at_5 = len(set(top_5_results).intersection(set(ground_truth))) / 5
        recall_at_5 = len(set(top_5_results).intersection(set(ground_truth))) / len(ground_truth)

        print(f"✅ Métricas de recuperación: P@5={precision_at_5:.3f}, R@5={recall_at_5:.3f}")
        success_count += 1

    except Exception as e:
        print(f"❌ Error en métricas de recuperación: {e}")

    return success_count, total_tests

def test_visualization_creation():
    """Prueba creación de visualizaciones básicas"""
    print("📈 Probando creación de visualizaciones...")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Datos sintéticos para prueba
        models = ['Model A', 'Model B', 'Model C']
        scores = [0.75, 0.82, 0.78]

        # Crear gráfico simple
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(models, scores, color=['skyblue', 'lightgreen', 'coral'])
        ax.set_title('Test Visualization')
        ax.set_ylabel('Score')

        # Guardar
        output_dir = COMPONENTS_ARTIFACTS
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "test_viz.png", dpi=150)
        plt.close()

        print("✅ Visualización básica creada")
        return True

    except Exception as e:
        print(f"❌ Error en visualización: {e}")
        return False

def test_config_loading():
    """Prueba carga de configuración"""
    print("⚙️  Probando carga de configuración...")

    try:
        ensure_sys_path([SRC_ROOT])
        from models_config import get_models_config, AudioEmbeddingModel, ModelsConfigLoader

        # Test enum
        assert AudioEmbeddingModel.YAMNET.value == "yamnet"
        assert AudioEmbeddingModel.SPEECHDPR.value == "speechdpr"
        print("✅ Enums de modelos funcionan")

        # Test loader
        loader = ModelsConfigLoader()
        config = loader.load_config()

        if config is not None:
            print(f"✅ Configuración cargada: {type(config)}")
        else:
            print("⚠️  Configuración retornó None")

        return True

    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def test_framework_structure():
    """Verifica la estructura básica del framework"""
    print("🏗️  Probando estructura del framework...")

    try:
        ensure_sys_path([SRC_ROOT])
        from embedding_evaluation_framework import EmbeddingBenchmark, TestCase, EvaluationMetrics

        # Test creación de TestCase
        test_case = TestCase(
            query_text="test query",
            expected_keywords=["test", "keyword"],
            category="test",
            difficulty="easy"
        )

        assert test_case.query_text == "test query"
        print("✅ TestCase funciona")

        # Test creación de EvaluationMetrics
        metrics = EvaluationMetrics(model_name="test_model")
        assert metrics.model_name == "test_model"
        print("✅ EvaluationMetrics funciona")

        # Test inicialización de benchmark (sin cargar modelos)
        eval_dir = COMPONENTS_ARTIFACTS / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        benchmark = EmbeddingBenchmark(str(eval_dir))
        assert benchmark.output_dir.exists()
        print("✅ EmbeddingBenchmark se inicializa")

        return True

    except Exception as e:
        print(f"❌ Error en estructura del framework: {e}")
        return False

def main():
    """Función principal de test de componentes"""
    print("🧪 Test de Componentes del Framework")
    print("=" * 60)

    tests = [
        ("Imports de módulos", test_imports),
        ("Configuración", test_config_loading),
        ("Estructura del framework", test_framework_structure),
        ("Generación de datos", test_data_generation_only),
        ("Cálculos de métricas", test_metrics_calculations),
        ("Creación de visualizaciones", test_visualization_creation),
    ]

    results = {}
    total_success = 0

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🔍 Ejecutando: {test_name}")
        print(f"{'='*60}")

        try:
            if test_name == "Cálculos de métricas":
                success_count, total_tests = test_func()
                success = success_count == total_tests
                print(f"📊 Métricas: {success_count}/{total_tests} exitosas")
            elif test_name == "Imports de módulos":
                import_results = test_func()
                success = all(import_results.values())
                successful_imports = sum(import_results.values())
                total_imports = len(import_results)
                print(f"📦 Imports: {successful_imports}/{total_imports} exitosos")
            else:
                success = test_func()

            results[test_name] = success

            if success:
                print(f"✅ {test_name}: EXITOSO")
                total_success += 1
            else:
                print(f"❌ {test_name}: FALLÓ")

        except Exception as e:
            print(f"💥 {test_name}: ERROR INESPERADO - {e}")
            results[test_name] = False

    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE COMPONENTES")
    print(f"{'='*60}")

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    total_tests = len(results)
    print(f"\n📈 Resultado: {total_success}/{total_tests} componentes funcionando")

    if total_success == total_tests:
        print("🎉 ¡Todos los componentes funcionan correctamente!")
        print("💡 El framework está listo para uso (sin cargar modelos pesados)")
    elif total_success >= total_tests * 0.8:
        print("✨ La mayoría de componentes funcionan. Sistema es funcional.")
    else:
        print("⚠️  Múltiples componentes tienen problemas. Revisa las dependencias.")

    return total_success == total_tests

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrumpidos")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
