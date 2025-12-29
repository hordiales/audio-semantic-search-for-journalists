#!/usr/bin/env python3
"""
Test y comparación de rendimiento entre bases de datos vectoriales:
FAISS, ChromaDB, Supabase y Memoria
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import SRC_ROOT, artifacts_dir, ensure_sys_path

ensure_sys_path([SRC_ROOT])
OUTPUT_ROOT = artifacts_dir("vector_databases")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vector_databases():
    """Prueba y compara el rendimiento de las bases de datos vectoriales"""
    print("🧪 Test de Bases de Datos Vectoriales")
    print("=" * 80)

    try:
        from vector_database_config import ConfigurationPreset, get_configurator
        from vector_database_interface import (
            VectorDBType,
        )

        # Configurar para pruebas
        configurator = get_configurator("test_vector_db_config.json")
        configurator.apply_preset(ConfigurationPreset.DEVELOPMENT)

        # Definir bases de datos a probar
        databases_to_test = [
            VectorDBType.MEMORY,
            VectorDBType.FAISS,
            # VectorDBType.CHROMADB,  # Descomentaremos si está disponible
            # VectorDBType.SUPABASE   # Requiere configuración específica
        ]

        # Generar datos de prueba
        test_data = generate_test_data(num_docs=1000, embedding_dim=256)
        test_queries = generate_test_queries(num_queries=50, embedding_dim=256)

        print("📊 Datos de prueba generados:")
        print(f"   📄 Documentos: {len(test_data)}")
        print(f"   🔍 Consultas: {len(test_queries)}")

        # Resultados de pruebas
        results = {}

        # Probar cada base de datos
        for db_type in databases_to_test:
            print(f"\n{'='*60}")
            print(f"🧪 Probando: {db_type.value.upper()}")
            print(f"{'='*60}")

            try:
                result = test_single_database(db_type, configurator, test_data, test_queries)
                results[db_type.value] = result
                print(f"✅ {db_type.value}: Prueba completada")

            except Exception as e:
                print(f"❌ {db_type.value}: Error - {e}")
                results[db_type.value] = {"error": str(e)}

        # Generar reporte de comparación
        generate_comparison_report(results)

        return True

    except Exception as e:
        print(f"❌ Error en tests: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_test_data(num_docs: int, embedding_dim: int) -> list[VectorDocument]:
    """Genera datos de prueba sintéticos"""
    print(f"📊 Generando {num_docs} documentos de prueba...")

    np.random.seed(42)  # Para reproducibilidad
    documents = []

    categories = ['política', 'economía', 'tecnología', 'salud', 'deportes']
    sources = ['radio', 'podcast', 'entrevista', 'debate', 'noticia']

    for i in range(num_docs):
        # Generar embedding con estructura semántica
        category_idx = i % len(categories)
        category = categories[category_idx]

        # Crear embedding con correlación por categoría
        base_embedding = np.random.randn(embedding_dim) * 0.1
        category_signal = np.zeros(embedding_dim)
        start_idx = category_idx * (embedding_dim // len(categories))
        end_idx = min(start_idx + (embedding_dim // len(categories)), embedding_dim)
        category_signal[start_idx:end_idx] = 1.0 + np.random.randn(end_idx - start_idx) * 0.2

        embedding = base_embedding + category_signal * 0.8
        embedding = embedding / np.linalg.norm(embedding)  # Normalizar

        # Crear documento
        doc = VectorDocument(
            id=f"doc_{i:04d}",
            embedding=embedding,
            text=f"Contenido sobre {category} número {i}. Información relevante para análisis semántico.",
            metadata={
                'category': category,
                'source': sources[i % len(sources)],
                'duration': 30 + np.random.randint(0, 300),
                'confidence': 0.7 + np.random.random() * 0.3,
                'speaker_count': np.random.randint(1, 5),
                'language': 'es'
            },
            category=category,
            timestamp=time.time() - np.random.randint(0, 86400 * 30),  # Último mes
            audio_file_path=f"test_audio_{i:04d}.wav"
        )

        documents.append(doc)

    print(f"✅ {len(documents)} documentos generados")
    return documents

def generate_test_queries(num_queries: int, embedding_dim: int) -> list[np.ndarray]:
    """Genera consultas de prueba"""
    print(f"🔍 Generando {num_queries} consultas de prueba...")

    np.random.seed(123)  # Seed diferente para consultas
    queries = []

    for i in range(num_queries):
        # Generar consulta con estructura similar a los documentos
        query_embedding = np.random.randn(embedding_dim)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        queries.append(query_embedding)

    print(f"✅ {len(queries)} consultas generadas")
    return queries

def test_single_database(db_type: VectorDBType, configurator, test_data: list[VectorDocument],
                        test_queries: list[np.ndarray]) -> dict[str, Any]:
    """Prueba una base de datos específica"""

    # Crear configuración específica
    config = configurator.get_vector_db_config(db_type)

    # Ajustes específicos para pruebas
    if db_type == VectorDBType.FAISS:
        config.faiss_index_type = "flat"  # Más simple para pruebas
        config.faiss_gpu = False
        config.faiss_index_path = str(OUTPUT_ROOT / "faiss_test_index.bin")

    elif db_type == VectorDBType.CHROMADB:
        config.chromadb_path = str(OUTPUT_ROOT / "chromadb_test")
        config.chromadb_collection_name = "test_collection"

    # Crear base de datos
    db = create_vector_database(config)

    # Métricas de prueba
    metrics = {
        "db_type": db_type.value,
        "config": config.__dict__.copy(),
        "initialization_time": 0.0,
        "insertion_time": 0.0,
        "insertion_rate": 0.0,
        "search_times": [],
        "avg_search_time": 0.0,
        "search_rate": 0.0,
        "memory_usage": 0.0,
        "accuracy_metrics": {},
        "errors": []
    }

    try:
        # 1. Inicialización
        print(f"🚀 Inicializando {db_type.value}...")
        start_time = time.time()
        if not db.initialize():
            raise Exception("Error en inicialización")
        metrics["initialization_time"] = time.time() - start_time
        print(f"   ⏱️  Inicialización: {metrics['initialization_time']:.3f}s")

        # 2. Inserción de documentos
        print(f"📥 Insertando {len(test_data)} documentos...")
        start_time = time.time()

        # Insertar en lotes para mejor rendimiento
        batch_size = 100
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            if not db.add_documents(batch):
                raise Exception(f"Error insertando lote {i//batch_size + 1}")

        insertion_time = time.time() - start_time
        metrics["insertion_time"] = insertion_time
        metrics["insertion_rate"] = len(test_data) / insertion_time
        print(f"   ⏱️  Inserción: {insertion_time:.3f}s ({metrics['insertion_rate']:.1f} docs/s)")

        # 3. Pruebas de búsqueda
        print(f"🔍 Realizando {len(test_queries)} búsquedas...")
        search_times = []
        search_results = []

        for i, query in enumerate(test_queries):
            start_time = time.time()
            results = db.search(query, k=10)
            search_time = time.time() - start_time

            search_times.append(search_time)
            search_results.append(results)

            if i % 10 == 0:
                print(f"   🔍 Búsqueda {i+1}/{len(test_queries)}: {search_time:.3f}s ({len(results)} resultados)")

        metrics["search_times"] = search_times
        metrics["avg_search_time"] = np.mean(search_times)
        metrics["search_rate"] = 1.0 / metrics["avg_search_time"] if metrics["avg_search_time"] > 0 else 0
        print(f"   ⏱️  Búsqueda promedio: {metrics['avg_search_time']:.3f}s ({metrics['search_rate']:.1f} búsquedas/s)")

        # 4. Evaluación de precisión
        print("📊 Evaluando precisión...")
        accuracy_metrics = evaluate_search_accuracy(test_data, test_queries, search_results)
        metrics["accuracy_metrics"] = accuracy_metrics

        # 5. Estadísticas de la base de datos
        db_stats = db.get_statistics()
        metrics["db_statistics"] = db_stats
        print(f"   📊 Documentos almacenados: {db_stats.get('document_count', 0)}")

        # 6. Limpiar
        if hasattr(db, 'clear'):
            db.clear()

        print(f"✅ Prueba de {db_type.value} completada exitosamente")

    except Exception as e:
        error_msg = f"Error en {db_type.value}: {e}"
        print(f"❌ {error_msg}")
        metrics["errors"].append(error_msg)

    return metrics

def evaluate_search_accuracy(test_data: list[VectorDocument], test_queries: list[np.ndarray],
                           search_results: list[list]) -> dict[str, float]:
    """Evalúa la precisión de los resultados de búsqueda"""

    if not search_results or not test_queries:
        return {"error": "No hay resultados para evaluar"}

    # Métricas básicas
    total_searches = len(search_results)
    successful_searches = sum(1 for results in search_results if len(results) > 0)
    avg_results_per_search = np.mean([len(results) for results in search_results])

    # Evaluar coherencia de categorías
    category_coherence_scores = []
    for results in search_results:
        if len(results) == 0:
            continue

        categories = [r.document.category for r in results if r.document.category]
        if len(categories) > 1:
            # Calcular coherencia (proporción de la categoría más común)
            from collections import Counter
            category_counts = Counter(categories)
            most_common_count = category_counts.most_common(1)[0][1]
            coherence = most_common_count / len(categories)
            category_coherence_scores.append(coherence)

    avg_category_coherence = np.mean(category_coherence_scores) if category_coherence_scores else 0.0

    # Evaluar distribución de puntajes de similitud
    similarity_scores = []
    for results in search_results:
        for result in results:
            similarity_scores.append(result.similarity_score)

    return {
        "success_rate": successful_searches / total_searches,
        "avg_results_per_search": avg_results_per_search,
        "avg_category_coherence": avg_category_coherence,
        "similarity_score_mean": np.mean(similarity_scores) if similarity_scores else 0.0,
        "similarity_score_std": np.std(similarity_scores) if similarity_scores else 0.0,
        "total_searches": total_searches,
        "successful_searches": successful_searches
    }

def generate_comparison_report(results: dict[str, dict[str, Any]]):
    """Genera reporte de comparación entre bases de datos"""
    print(f"\n{'='*80}")
    print("📊 REPORTE DE COMPARACIÓN DE BASES DE DATOS VECTORIALES")
    print(f"{'='*80}")

    # Crear tabla de comparación
    comparison_data = []

    for db_name, metrics in results.items():
        if "error" in metrics:
            comparison_data.append({
                "Database": db_name.upper(),
                "Status": "❌ ERROR",
                "Init Time": "N/A",
                "Insert Rate": "N/A",
                "Search Time": "N/A",
                "Success Rate": "N/A",
                "Error": metrics["error"]
            })
            continue

        comparison_data.append({
            "Database": db_name.upper(),
            "Status": "✅ OK",
            "Init Time": f"{metrics.get('initialization_time', 0):.3f}s",
            "Insert Rate": f"{metrics.get('insertion_rate', 0):.1f} docs/s",
            "Search Time": f"{metrics.get('avg_search_time', 0):.3f}s",
            "Search Rate": f"{metrics.get('search_rate', 0):.1f} q/s",
            "Success Rate": f"{metrics.get('accuracy_metrics', {}).get('success_rate', 0)*100:.1f}%",
            "Coherence": f"{metrics.get('accuracy_metrics', {}).get('avg_category_coherence', 0)*100:.1f}%"
        })

    # Mostrar tabla
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\n📈 RESUMEN DE RENDIMIENTO:")
        print(df.to_string(index=False))

    # Detalles específicos
    print("\n📋 DETALLES POR BASE DE DATOS:")
    for db_name, metrics in results.items():
        print(f"\n🔸 {db_name.upper()}:")

        if "error" in metrics:
            print(f"   ❌ Error: {metrics['error']}")
            continue

        print(f"   ⚡ Inicialización: {metrics.get('initialization_time', 0):.3f}s")
        print(f"   📥 Inserción: {metrics.get('insertion_rate', 0):.1f} documentos/segundo")
        print(f"   🔍 Búsqueda: {metrics.get('avg_search_time', 0):.3f}s promedio")

        accuracy = metrics.get('accuracy_metrics', {})
        if accuracy:
            print("   📊 Precisión:")
            print(f"      • Tasa de éxito: {accuracy.get('success_rate', 0)*100:.1f}%")
            print(f"      • Coherencia de categorías: {accuracy.get('avg_category_coherence', 0)*100:.1f}%")
            print(f"      • Resultados promedio: {accuracy.get('avg_results_per_search', 0):.1f}")

    # Recomendaciones
    print("\n💡 RECOMENDACIONES:")

    successful_dbs = [db for db, metrics in results.items() if "error" not in metrics]

    if not successful_dbs:
        print("   ⚠️  Ninguna base de datos funcionó correctamente")
        return

    # Mejor en velocidad de inserción
    fastest_insert = max(successful_dbs,
                        key=lambda db: results[db].get('insertion_rate', 0))
    print(f"   🚀 Mejor para inserción masiva: {fastest_insert.upper()}")

    # Mejor en velocidad de búsqueda
    fastest_search = min(successful_dbs,
                        key=lambda db: results[db].get('avg_search_time', float('inf')))
    print(f"   ⚡ Mejor para búsquedas rápidas: {fastest_search.upper()}")

    # Mejor precisión
    most_accurate = max(successful_dbs,
                       key=lambda db: results[db].get('accuracy_metrics', {}).get('success_rate', 0))
    print(f"   🎯 Mejor precisión: {most_accurate.upper()}")

    print("\n💾 Casos de uso recomendados:")
    print("   • Desarrollo/Prototipado: MEMORY")
    print("   • Producción local: FAISS")
    print("   • Aplicaciones web: CHROMADB")
    print("   • Producción escalable: SUPABASE")

    # Guardar resultados detallados
    output_dir = OUTPUT_ROOT
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "vector_db_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n💾 Resultados detallados guardados en: {results_file}")

def main():
    """Función principal"""
    try:
        success = test_vector_databases()
        if success:
            print("\n🏆 TESTS COMPLETADOS EXITOSAMENTE")
            return 0
        print("\n💥 TESTS FALLARON")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Tests interrumpidos")
        return 1
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
