#!/usr/bin/env python3
"""
Prueba rápida del sistema de benchmark de embeddings.
Ejecuta una evaluación pequeña para verificar que todo funciona.
"""

import sys
import logging
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import artifacts_dir

BENCHMARK_ARTIFACTS = artifacts_dir("benchmark_quick")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_quick_benchmark():
    """Ejecuta una prueba rápida del benchmark"""
    print("🧪 Prueba Rápida del Sistema de Benchmark")
    print("=" * 50)

    try:
        # Importar desde el script principal
        sys.path.insert(0, '.')
        from run_embedding_benchmark import ComprehensiveEmbeddingBenchmark

        # Crear benchmark con directorio de prueba
        benchmark = ComprehensiveEmbeddingBenchmark(str(BENCHMARK_ARTIFACTS))

        print("\n🔍 Verificando dependencias...")
        dependencies = benchmark.check_dependencies()

        # Verificar que tenemos al menos un modelo funcional
        available_models = sum([
            dependencies.get("torch", False) or dependencies.get("tensorflow", False),
            dependencies.get("transformers", False)
        ])

        if available_models < 2:
            print("⚠️  Dependencias insuficientes para prueba completa")
            print("💡 Instalando dependencias con: pip install -r requirements_benchmark.txt")
            return False

        print("\n📊 Generando datos de prueba pequeños...")
        success = benchmark.generate_test_data(
            num_samples=20,  # Muy pequeño para prueba rápida
            num_queries=5,
            regenerate=True
        )

        if not success:
            print("❌ Error generando datos de prueba")
            return False

        print("\n🎯 Ejecutando evaluación rápida...")

        # Solo evaluar modelos disponibles
        available_models = []
        if dependencies.get("tensorflow", False):
            available_models.append("yamnet")
        if dependencies.get("torch", False) and dependencies.get("transformers", False):
            available_models.append("speechdpr")

        if not available_models:
            print("❌ No hay modelos disponibles para evaluar")
            return False

        print(f"🤖 Evaluando modelos: {available_models}")

        success = benchmark.run_model_evaluations(available_models)

        if success:
            print("✅ Evaluación completada exitosamente")

            # Mostrar resultados básicos
            if benchmark.benchmark.results:
                print("\n📊 Resultados:")
                for model_name, metrics in benchmark.benchmark.results.items():
                    print(f"   {model_name}:")
                    print(f"     - BERTScore F1: {metrics.bert_score_f1:.3f}")
                    print(f"     - Consultas exitosas: {metrics.successful_queries}/{metrics.total_queries}")
                    print(f"     - Tiempo promedio: {metrics.query_time:.3f}s")

            print("\n📊 Generando visualizaciones...")
            benchmark.generate_visualizations()

            print("\n📋 Creando reporte...")
            benchmark.create_summary_report()

            print(f"\n✅ Prueba completada. Resultados en: {benchmark.output_dir}")
            return True
        else:
            print("❌ Error en evaluación")
            return False

    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_quick_benchmark()
        print("\n" + "=" * 50)
        if success:
            print("🎉 ¡Prueba exitosa! El sistema de benchmark está funcionando correctamente.")
            print("💡 Para ejecutar benchmark completo: python run_embedding_benchmark.py")
        else:
            print("⚠️  La prueba no se completó exitosamente. Verifica las dependencias.")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n🛑 Prueba interrumpida")
        sys.exit(1)
