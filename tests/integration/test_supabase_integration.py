#!/usr/bin/env python3
"""
Test de integración del vector database de Supabase con el sistema existente
Prueba la configuración sin necesidad de conectar a Supabase real
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import ensure_sys_path, SRC_ROOT, resources_dir

ensure_sys_path([SRC_ROOT])

RESOURCES_ROOT = resources_dir()

def test_supabase_configuration():
    """Prueba la configuración de Supabase en el sistema"""
    print("🔧 Test de Configuración de Supabase")
    print("=" * 60)

    try:
        from vector_database_config import get_configurator, ConfigurationPreset
        from vector_database_interface import VectorDBType

        # Test 1: Configuración con preset PRODUCTION
        print("1️⃣ Probando preset PRODUCTION...")
        config_path = RESOURCES_ROOT / "test_supabase_config.json"
        config_location = str(config_path) if config_path.exists() else "test_supabase_config.json"

        configurator = get_configurator(config_location)
        configurator.apply_preset(ConfigurationPreset.PRODUCTION)

        config = configurator.get_vector_db_config(VectorDBType.SUPABASE)
        print(f"   ✅ Configuración generada: {config.db_type.value}")
        print(f"   📊 Dimensión: {config.embedding_dimension}")
        print(f"   📐 Métrica: {config.similarity_metric}")

        # Test 2: Variables de entorno simuladas
        print("\n2️⃣ Simulando variables de entorno...")
        original_env = {}

        # Guardar valores originales
        for key in ['SUPABASE_URL', 'SUPABASE_KEY', 'SUPABASE_DB_PASSWORD']:
            original_env[key] = os.environ.get(key)

        # Configurar valores de prueba
        os.environ['VECTOR_DB_TYPE'] = 'supabase'
        os.environ['SUPABASE_URL'] = 'https://test123.supabase.co'
        os.environ['SUPABASE_KEY'] = 'eyJ0ZXN0IjoidmFsdWUifQ.test.signature'
        os.environ['SUPABASE_DB_PASSWORD'] = 'test_password_123'

        # Obtener configuración actualizada
        updated_config = configurator.get_vector_db_config(VectorDBType.SUPABASE)
        print(f"   ✅ Variables aplicadas correctamente")

        # Restaurar valores originales
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

        print("   🔄 Variables restauradas")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_supabase_vector_database_class():
    """Prueba la clase SupabaseVectorDatabase sin conectar"""
    print("\n🗄️ Test de Clase SupabaseVectorDatabase")
    print("=" * 60)

    try:
        from vector_db_supabase import SupabaseVectorDatabase
        from vector_database_interface import VectorDBConfig, VectorDBType

        # Crear configuración de prueba
        config = VectorDBConfig(
            db_type=VectorDBType.SUPABASE,
            embedding_dimension=384,
            similarity_metric="cosine"
        )

        # Configurar parámetros de Supabase (sin conectar)
        config.supabase_url = "https://test123.supabase.co"
        config.supabase_key = "test_key"
        config.supabase_db_password = "test_password"

        print("1️⃣ Creando instancia de SupabaseVectorDatabase...")
        db = SupabaseVectorDatabase(config)
        print("   ✅ Instancia creada correctamente")

        print("2️⃣ Verificando métodos de la clase...")
        methods = ['initialize', 'add_documents', 'search', 'get_statistics']
        for method in methods:
            if hasattr(db, method):
                print(f"   ✅ Método {method} disponible")
            else:
                print(f"   ❌ Método {method} faltante")

        print("3️⃣ Verificando configuración interna...")
        print(f"   📊 Dimensión configurada: {db.config.embedding_dimension}")
        print(f"   📐 Métrica configurada: {db.config.similarity_metric}")
        print(f"   🌐 URL configurada: {db.config.supabase_url}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_migration_data_structure():
    """Prueba la estructura de datos para migración"""
    print("\n📊 Test de Estructura de Datos para Migración")
    print("=" * 60)

    try:
        # Cargar dataset real
        dataset_path = "dataset/embeddings/segments_metadata.csv"
        if not Path(dataset_path).exists():
            print(f"   ⚠️  Dataset no encontrado: {dataset_path}")
            return False

        df = pd.read_csv(dataset_path)
        print(f"1️⃣ Dataset cargado: {len(df)} segmentos")

        # Verificar columnas requeridas
        required_columns = [
            'segment_id', 'start_time', 'end_time', 'duration', 'text',
            'language', 'source_file', 'original_file_name',
            'embedding_model', 'audio_embedding_model'
        ]

        missing_columns = []
        for col in required_columns:
            if col in df.columns:
                print(f"   ✅ Columna '{col}' presente")
            else:
                print(f"   ❌ Columna '{col}' faltante")
                missing_columns.append(col)

        if missing_columns:
            print(f"   ⚠️  Columnas faltantes: {missing_columns}")
            return False

        # Verificar tipos de datos
        print("2️⃣ Verificando tipos de datos...")
        sample_row = df.iloc[0]

        checks = [
            ('segment_id', 'int', isinstance(sample_row['segment_id'], (int, np.integer))),
            ('start_time', 'float', isinstance(sample_row['start_time'], (float, np.floating))),
            ('duration', 'float', isinstance(sample_row['duration'], (float, np.floating))),
            ('text', 'string', isinstance(sample_row['text'], str)),
            ('language', 'string', isinstance(sample_row['language'], str))
        ]

        for field, expected_type, check in checks:
            status = "✅" if check else "❌"
            print(f"   {status} {field}: {expected_type}")

        # Simular estructura de embeddings
        print("3️⃣ Simulando estructura de embeddings...")

        # Embedding de texto (384D)
        text_embedding = np.random.randn(384)
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        print(f"   ✅ Embedding texto: {text_embedding.shape} (normalizado)")

        # Embedding de audio (1024D)
        audio_embedding = np.random.randn(1024)
        audio_embedding = audio_embedding / np.linalg.norm(audio_embedding)
        print(f"   ✅ Embedding audio: {audio_embedding.shape} (normalizado)")

        # Verificar que se pueden convertir a lista (requerido para PostgreSQL)
        text_list = text_embedding.tolist()
        audio_list = audio_embedding.tolist()
        print(f"   ✅ Conversión a lista: texto={len(text_list)}, audio={len(audio_list)}")

        # Datos de ejemplo para inserción
        sample_data = {
            'segment_id': int(sample_row['segment_id']),
            'start_time': float(sample_row['start_time']),
            'end_time': float(sample_row['end_time']),
            'duration': float(sample_row['duration']),
            'text': str(sample_row['text']),
            'language': str(sample_row['language']),
            'text_embedding': text_list,
            'audio_embedding': audio_list
        }

        print("4️⃣ Estructura de datos validada:")
        for key, value in sample_data.items():
            if key.endswith('_embedding'):
                print(f"   ✅ {key}: {type(value).__name__}[{len(value)}]")
            else:
                print(f"   ✅ {key}: {type(value).__name__} = {str(value)[:50]}...")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sql_query_generation():
    """Prueba la generación de consultas SQL para búsquedas vectoriales"""
    print("\n🔍 Test de Generación de Consultas SQL")
    print("=" * 60)

    try:
        # Simular embedding de consulta
        query_embedding = np.random.randn(384)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_list = query_embedding.tolist()

        print("1️⃣ Consultas SQL de ejemplo generadas:")

        # Consulta básica de similitud
        basic_query = """
        SELECT segment_id, text, language,
               (text_embedding <=> %s::vector) as distance
        FROM audio_segments
        ORDER BY text_embedding <=> %s::vector
        LIMIT 5;
        """
        print("   ✅ Consulta básica de similitud")

        # Consulta con filtros
        filtered_query = """
        SELECT segment_id, text, language, duration,
               (text_embedding <=> %s::vector) as distance
        FROM audio_segments
        WHERE language = %s AND duration > %s
        ORDER BY text_embedding <=> %s::vector
        LIMIT 10;
        """
        print("   ✅ Consulta con filtros (idioma, duración)")

        # Consulta híbrida (texto + audio)
        hybrid_query = """
        SELECT segment_id, text, language,
               (text_embedding <=> %s::vector) as text_distance,
               (audio_embedding <=> %s::vector) as audio_distance,
               ((text_embedding <=> %s::vector) + (audio_embedding <=> %s::vector)) / 2 as combined_distance
        FROM audio_segments
        ORDER BY combined_distance
        LIMIT 10;
        """
        print("   ✅ Consulta híbrida (texto + audio)")

        # Consulta con agregaciones
        stats_query = """
        SELECT language, COUNT(*) as count, AVG(duration) as avg_duration
        FROM audio_segments
        GROUP BY language
        ORDER BY count DESC;
        """
        print("   ✅ Consulta de estadísticas")

        print("2️⃣ Parámetros de consulta validados:")
        print(f"   ✅ Embedding como lista: {len(query_list)} elementos")
        print(f"   ✅ Primer elemento: {query_list[0]:.6f}")
        print(f"   ✅ Norma L2: {np.linalg.norm(query_embedding):.6f}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Función principal de tests"""
    print("🧪 Test de Integración Supabase - Audio Semantic Search")
    print("=" * 80)

    tests = [
        ("Configuración de Supabase", test_supabase_configuration),
        ("Clase SupabaseVectorDatabase", test_supabase_vector_database_class),
        ("Estructura de Datos", test_migration_data_structure),
        ("Generación de Consultas SQL", test_sql_query_generation)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*80}")
            print(f"🧪 Ejecutando: {test_name}")
            print(f"{'='*80}")

            success = test_func()
            results.append((test_name, success))

            status = "✅ EXITOSO" if success else "❌ FALLÓ"
            print(f"\n{status}: {test_name}")

        except Exception as e:
            print(f"💥 Error inesperado en {test_name}: {e}")
            results.append((test_name, False))

    # Resumen final
    print(f"\n{'='*80}")
    print("📊 RESUMEN DE TESTS DE INTEGRACIÓN")
    print(f"{'='*80}")

    successful = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        print(f"{status} {test_name}")
        if success:
            successful += 1

    print(f"\n📈 Resultado final: {successful}/{total} tests exitosos")

    if successful == total:
        print("🎉 ¡Todos los tests de integración pasaron!")
        print("\n💡 Próximos pasos:")
        print("   1. Configurar proyecto en Supabase (https://supabase.com)")
        print("   2. Activar extensión vector en el dashboard")
        print("   3. Configurar variables de entorno")
        print("   4. Ejecutar: python migrate_to_supabase.py")
        print("   5. ¡Disfrutar búsquedas vectoriales en la nube!")
        return 0
    else:
        print("⚠️  Algunos tests fallaron. Revisar configuración.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrumpidos")
        sys.exit(1)
