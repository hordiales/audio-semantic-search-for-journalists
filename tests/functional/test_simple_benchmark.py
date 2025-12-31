#!/usr/bin/env python3
"""
Prueba simplificada del sistema de benchmark.
Evita problemas de imports y se enfoca en funcionalidad básica.
"""

import logging
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import SRC_ROOT, artifacts_dir, ensure_sys_path

SIMPLE_ARTIFACTS = artifacts_dir("simple_benchmark")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_generation():
    """Prueba la generación de datos sintéticos"""
    print("📊 Probando generación de datos sintéticos...")

    try:
        # Imports locales
        ensure_sys_path([SRC_ROOT])
        from test_data_generator import SyntheticTestDataGenerator

        # Crear generador
        generator = SyntheticTestDataGenerator(str(SIMPLE_ARTIFACTS))

        # Generar dataset pequeño
        df = generator.generate_test_dataset(num_samples=10)

        # Verificar estructura
        assert len(df) == 10
        assert 'text' in df.columns
        assert 'category' in df.columns
        assert 'keywords' in df.columns

        print(f"✅ Dataset generado: {len(df)} muestras")
        print(f"📊 Categorías: {df['category'].unique()}")

        # Generar consultas
        queries = generator.create_ground_truth_queries(df, num_queries=3)
        assert len(queries) == 3

        print(f"✅ Consultas generadas: {len(queries)}")

        return True

    except Exception as e:
        print(f"❌ Error en generación de datos: {e}")
        return False

def test_bertscore():
    """Prueba el cálculo de BERTScore"""
    print("📈 Probando cálculo de BERTScore...")

    try:
        from bert_score import score as bert_score

        # Datos de prueba
        predictions = [
            "El presidente anunció nuevas medidas económicas",
            "Los mercados financieros muestran volatilidad"
        ]
        references = [
            "El mandatario presentó políticas económicas",
            "Los mercados presentan inestabilidad financiera"
        ]

        # Calcular BERTScore
        P, R, F1 = bert_score(predictions, references, lang="es", verbose=False)

        print("✅ BERTScore calculado:")
        print(f"   Precision: {P.mean():.3f}")
        print(f"   Recall: {R.mean():.3f}")
        print(f"   F1: {F1.mean():.3f}")

        return True

    except ImportError:
        print("⚠️  BERTScore no disponible")
        return False
    except Exception as e:
        print(f"❌ Error calculando BERTScore: {e}")
        return False

def test_sentence_similarity():
    """Prueba el cálculo de similitud semántica"""
    print("🔤 Probando similitud semántica...")

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        # Modelo multilingüe
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Textos de prueba
        texts = [
            "discurso político sobre economía",
            "medidas económicas del gobierno",
            "conferencia de tecnología"
        ]

        # Generar embeddings
        embeddings = model.encode(texts)

        # Calcular similitudes
        similarities = cosine_similarity(embeddings)

        print("✅ Similitudes calculadas:")
        print(f"   Economía-Política: {similarities[0,1]:.3f}")
        print(f"   Economía-Tecnología: {similarities[0,2]:.3f}")

        return True

    except ImportError:
        print("⚠️  Sentence Transformers no disponible")
        return False
    except Exception as e:
        print(f"❌ Error calculando similitud: {e}")
        return False

def test_visualization():
    """Prueba las capacidades de visualización"""
    print("📊 Probando visualización...")

    try:
        import matplotlib.pyplot as plt

        # Datos sintéticos para prueba
        models = ['YAMNet', 'SpeechDPR', 'CLAP']
        bertscore_f1 = [0.756, 0.812, 0.789]
        query_time = [0.045, 0.156, 0.123]

        # Crear gráfico simple
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # BERTScore comparison
        ax1.bar(models, bertscore_f1, color=['skyblue', 'lightgreen', 'coral'])
        ax1.set_title('BERTScore F1 Comparison')
        ax1.set_ylabel('F1 Score')

        # Query time comparison
        ax2.bar(models, query_time, color=['skyblue', 'lightgreen', 'coral'])
        ax2.set_title('Query Time Comparison')
        ax2.set_ylabel('Time (seconds)')

        plt.tight_layout()

        # Guardar como prueba
        output_dir = SIMPLE_ARTIFACTS
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "test_visualization.png", dpi=150)
        plt.close()

        print("✅ Visualización creada exitosamente")
        return True

    except Exception as e:
        print(f"❌ Error en visualización: {e}")
        return False

def test_metrics_calculation():
    """Prueba el cálculo de métricas de recuperación"""
    print("🎯 Probando métricas de recuperación...")

    try:

        # Simular resultados de búsqueda
        # Ground truth: elementos 1, 3, 5 son relevantes
        ground_truth = [1, 3, 5]

        # Resultados simulados (top-5)
        top_5_results = [1, 2, 3, 4, 5]  # 3 relevantes de 5 recuperados

        # Calcular precision@5 y recall@5
        precision_at_5 = len(set(top_5_results).intersection(set(ground_truth))) / 5
        recall_at_5 = len(set(top_5_results).intersection(set(ground_truth))) / len(ground_truth)

        print("✅ Métricas calculadas:")
        print(f"   Precision@5: {precision_at_5:.3f}")
        print(f"   Recall@5: {recall_at_5:.3f}")

        # Probar diferentes valores de K
        for k in [1, 3, 5]:
            top_k = top_5_results[:k]
            p_at_k = len(set(top_k).intersection(set(ground_truth))) / k
            r_at_k = len(set(top_k).intersection(set(ground_truth))) / len(ground_truth)
            print(f"   P@{k}: {p_at_k:.3f}, R@{k}: {r_at_k:.3f}")

        return True

    except Exception as e:
        print(f"❌ Error calculando métricas: {e}")
        return False

def test_basic_model_availability():
    """Prueba disponibilidad básica de modelos sin cargarlos completamente"""
    print("🤖 Verificando disponibilidad de modelos...")

    available_models = []

    # Test TensorFlow/YAMNet
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        print("✅ TensorFlow disponible para YAMNet")
        available_models.append("yamnet")
    except ImportError:
        print("❌ TensorFlow no disponible")

    # Test PyTorch/Transformers for SpeechDPR
    try:
        import torch
        import transformers
        print("✅ PyTorch y Transformers disponibles para SpeechDPR")
        available_models.append("speechdpr")
    except ImportError:
        print("❌ PyTorch/Transformers no disponibles")

    # Test CLAP
    try:
        import laion_clap
        print("✅ LAION CLAP disponible")
        available_models.append("clap")
    except ImportError:
        print("⚠️  LAION CLAP no disponible")

    print(f"📊 Modelos disponibles: {available_models}")
    return len(available_models) > 0

def main():
    """Función principal de prueba simplificada"""
    print("🧪 Prueba Simplificada del Sistema de Benchmark")
    print("=" * 60)

    tests = [
        ("Disponibilidad de modelos", test_basic_model_availability),
        ("Generación de datos", test_data_generation),
        ("Cálculo de BERTScore", test_bertscore),
        ("Similitud semántica", test_sentence_similarity),
        ("Métricas de recuperación", test_metrics_calculation),
        ("Visualización", test_visualization),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🔍 Ejecutando: {test_name}")
        print(f"{'='*60}")

        try:
            success = test_func()
            results[test_name] = success

            if success:
                print(f"✅ {test_name}: EXITOSO")
            else:
                print(f"❌ {test_name}: FALLÓ")

        except Exception as e:
            print(f"💥 {test_name}: ERROR INESPERADO - {e}")
            results[test_name] = False

    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE PRUEBAS")
    print(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n📈 Resultado final: {passed}/{total} pruebas exitosas")

    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El sistema está funcionando correctamente.")
        print("💡 Los componentes básicos del benchmark están operativos.")
    elif passed >= total * 0.7:
        print("✨ La mayoría de pruebas pasaron. El sistema es funcional con limitaciones menores.")
    else:
        print("⚠️  Múltiples pruebas fallaron. Revisa las dependencias y configuración.")

    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Pruebas interrumpidas")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
