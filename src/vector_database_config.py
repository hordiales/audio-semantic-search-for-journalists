"""
Sistema de configuración para bases de datos vectoriales.
Permite configurar y cambiar entre FAISS, ChromaDB y Supabase fácilmente.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    from .vector_database_interface import VectorDBConfig, VectorDBType
except ImportError:
    from vector_database_interface import VectorDBConfig, VectorDBType

logger = logging.getLogger(__name__)

class ConfigurationPreset(Enum):
    """Presets de configuración para diferentes casos de uso"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    RESEARCH = "research"
    DEMO = "demo"

@dataclass
class VectorDatabaseSettings:
    """Configuración completa del sistema de bases de datos vectoriales"""

    # Configuración principal
    active_database: VectorDBType = VectorDBType.MEMORY
    embedding_dimension: int = 512
    similarity_metric: str = "cosine"

    # Configuraciones específicas por DB
    faiss_config: dict[str, Any] = field(default_factory=dict)
    chromadb_config: dict[str, Any] = field(default_factory=dict)
    supabase_config: dict[str, Any] = field(default_factory=dict)

    # Configuración de fallback
    fallback_database: VectorDBType = VectorDBType.MEMORY
    auto_fallback: bool = True

    # Configuración de rendimiento
    batch_size: int = 100
    search_timeout: float = 30.0
    connection_pool_size: int = 10

    # Configuración de almacenamiento
    data_directory: str = "data/vector_db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24

class VectorDatabaseConfigurator:
    """Gestiona la configuración de bases de datos vectoriales"""

    def __init__(self, config_file: str | None = None):
        self.config_file = config_file or "vector_db_config.json"
        self.settings = VectorDatabaseSettings()
        self._load_config()

    def _load_config(self):
        """Carga la configuración desde archivo"""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, encoding='utf-8') as f:
                    config_data = json.load(f)

                # Actualizar configuración
                self._update_settings_from_dict(config_data)
                logger.info(f"📋 Configuración cargada desde: {config_path}")
            else:
                logger.info("📋 Archivo de configuración no encontrado, usando valores por defecto")
                self._load_from_environment()
        except Exception as e:
            logger.error(f"❌ Error cargando configuración: {e}")
            self._load_from_environment()

    def _load_from_environment(self):
        """Carga configuración desde variables de entorno"""
        try:
            # Base de datos activa
            db_type = os.getenv('VECTOR_DB_TYPE', 'memory').lower()
            if db_type in [db.value for db in VectorDBType]:
                self.settings.active_database = VectorDBType(db_type)

            # Dimensión de embeddings
            if os.getenv('EMBEDDING_DIMENSION'):
                self.settings.embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION'))

            # Métrica de similitud
            if os.getenv('SIMILARITY_METRIC'):
                self.settings.similarity_metric = os.getenv('SIMILARITY_METRIC')

            # Configuraciones específicas
            self._load_faiss_env()
            self._load_chromadb_env()
            self._load_supabase_env()

            logger.info("🌍 Configuración cargada desde variables de entorno")

        except Exception as e:
            logger.error(f"❌ Error cargando configuración de entorno: {e}")

    def _load_faiss_env(self):
        """Carga configuración FAISS desde variables de entorno"""
        faiss_config = {}

        if os.getenv('FAISS_INDEX_TYPE'):
            faiss_config['index_type'] = os.getenv('FAISS_INDEX_TYPE')

        if os.getenv('FAISS_INDEX_PATH'):
            faiss_config['index_path'] = os.getenv('FAISS_INDEX_PATH')

        if os.getenv('FAISS_GPU'):
            faiss_config['gpu'] = os.getenv('FAISS_GPU').lower() == 'true'

        if os.getenv('FAISS_NPROBE'):
            faiss_config['nprobe'] = int(os.getenv('FAISS_NPROBE'))

        self.settings.faiss_config.update(faiss_config)

    def _load_chromadb_env(self):
        """Carga configuración ChromaDB desde variables de entorno"""
        chromadb_config = {}

        if os.getenv('CHROMADB_PATH'):
            chromadb_config['path'] = os.getenv('CHROMADB_PATH')

        if os.getenv('CHROMADB_COLLECTION'):
            chromadb_config['collection_name'] = os.getenv('CHROMADB_COLLECTION')

        if os.getenv('CHROMADB_DISTANCE'):
            chromadb_config['distance_function'] = os.getenv('CHROMADB_DISTANCE')

        self.settings.chromadb_config.update(chromadb_config)

    def _load_supabase_env(self):
        """Carga configuración Supabase desde variables de entorno"""
        supabase_config = {}

        if os.getenv('SUPABASE_URL'):
            supabase_config['url'] = os.getenv('SUPABASE_URL')

        if os.getenv('SUPABASE_KEY'):
            supabase_config['key'] = os.getenv('SUPABASE_KEY')

        if os.getenv('SUPABASE_TABLE'):
            supabase_config['table_name'] = os.getenv('SUPABASE_TABLE')

        if os.getenv('SUPABASE_DB_PASSWORD'):
            supabase_config['db_password'] = os.getenv('SUPABASE_DB_PASSWORD')

        self.settings.supabase_config.update(supabase_config)

    def _update_settings_from_dict(self, config_data: dict[str, Any]):
        """Actualiza configuración desde diccionario"""
        try:
            # Configuración principal
            if 'active_database' in config_data:
                self.settings.active_database = VectorDBType(config_data['active_database'])

            if 'embedding_dimension' in config_data:
                self.settings.embedding_dimension = config_data['embedding_dimension']

            if 'similarity_metric' in config_data:
                self.settings.similarity_metric = config_data['similarity_metric']

            # Configuraciones específicas
            if 'faiss_config' in config_data:
                self.settings.faiss_config.update(config_data['faiss_config'])

            if 'chromadb_config' in config_data:
                self.settings.chromadb_config.update(config_data['chromadb_config'])

            if 'supabase_config' in config_data:
                self.settings.supabase_config.update(config_data['supabase_config'])

            # Otras configuraciones
            for key in ['fallback_database', 'auto_fallback', 'batch_size',
                       'search_timeout', 'connection_pool_size', 'data_directory',
                       'backup_enabled', 'backup_interval_hours']:
                if key in config_data:
                    setattr(self.settings, key, config_data[key])

        except Exception as e:
            logger.error(f"❌ Error actualizando configuración: {e}")

    def save_config(self):
        """Guarda la configuración actual en archivo"""
        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            config_data = {
                'active_database': self.settings.active_database.value,
                'embedding_dimension': self.settings.embedding_dimension,
                'similarity_metric': self.settings.similarity_metric,
                'faiss_config': self.settings.faiss_config,
                'chromadb_config': self.settings.chromadb_config,
                'supabase_config': self.settings.supabase_config,
                'fallback_database': self.settings.fallback_database.value,
                'auto_fallback': self.settings.auto_fallback,
                'batch_size': self.settings.batch_size,
                'search_timeout': self.settings.search_timeout,
                'connection_pool_size': self.settings.connection_pool_size,
                'data_directory': self.settings.data_directory,
                'backup_enabled': self.settings.backup_enabled,
                'backup_interval_hours': self.settings.backup_interval_hours
            }

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Configuración guardada en: {config_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error guardando configuración: {e}")
            return False

    def get_vector_db_config(self, db_type: VectorDBType | None = None) -> VectorDBConfig:
        """Genera configuración VectorDBConfig para la base de datos especificada"""
        if db_type is None:
            db_type = self.settings.active_database

        config = VectorDBConfig(
            db_type=db_type,
            embedding_dimension=self.settings.embedding_dimension,
            similarity_metric=self.settings.similarity_metric
        )

        # Configurar según el tipo de DB
        if db_type == VectorDBType.FAISS:
            faiss_cfg = self.settings.faiss_config
            config.faiss_index_type = faiss_cfg.get('index_type', 'flat')
            config.faiss_index_path = faiss_cfg.get('index_path', f"{self.settings.data_directory}/faiss_index.bin")
            config.faiss_gpu = faiss_cfg.get('gpu', False)
            config.faiss_nprobe = faiss_cfg.get('nprobe', 10)

        elif db_type == VectorDBType.CHROMADB:
            chromadb_cfg = self.settings.chromadb_config
            config.chromadb_path = chromadb_cfg.get('path', f"{self.settings.data_directory}/chromadb")
            config.chromadb_collection_name = chromadb_cfg.get('collection_name', 'audio_embeddings')
            config.chromadb_distance_function = chromadb_cfg.get('distance_function', 'cosine')

        elif db_type == VectorDBType.SUPABASE:
            supabase_cfg = self.settings.supabase_config
            config.supabase_url = supabase_cfg.get('url')
            config.supabase_key = supabase_cfg.get('key')
            config.supabase_table_name = supabase_cfg.get('table_name', 'audio_embeddings')
            config.supabase_connection_pool_size = self.settings.connection_pool_size

        return config

    def apply_preset(self, preset: ConfigurationPreset):
        """Aplica un preset de configuración"""
        if preset == ConfigurationPreset.DEVELOPMENT:
            self._apply_development_preset()
        elif preset == ConfigurationPreset.PRODUCTION:
            self._apply_production_preset()
        elif preset == ConfigurationPreset.RESEARCH:
            self._apply_research_preset()
        elif preset == ConfigurationPreset.DEMO:
            self._apply_demo_preset()

        logger.info(f"🎛️  Preset aplicado: {preset.value}")

    def _apply_development_preset(self):
        """Configuración optimizada para desarrollo"""
        self.settings.active_database = VectorDBType.MEMORY
        self.settings.fallback_database = VectorDBType.MEMORY
        self.settings.auto_fallback = True
        self.settings.batch_size = 50
        self.settings.search_timeout = 10.0
        self.settings.backup_enabled = False

    def _apply_production_preset(self):
        """Configuración optimizada para producción"""
        self.settings.active_database = VectorDBType.SUPABASE
        self.settings.fallback_database = VectorDBType.FAISS
        self.settings.auto_fallback = True
        self.settings.batch_size = 500
        self.settings.search_timeout = 5.0
        self.settings.backup_enabled = True
        self.settings.backup_interval_hours = 6

        # Configuración FAISS para fallback
        self.settings.faiss_config.update({
            'index_type': 'ivf',
            'gpu': True,
            'nprobe': 20
        })

    def _apply_research_preset(self):
        """Configuración optimizada para investigación"""
        self.settings.active_database = VectorDBType.FAISS
        self.settings.fallback_database = VectorDBType.CHROMADB
        self.settings.auto_fallback = True
        self.settings.batch_size = 200
        self.settings.search_timeout = 30.0
        self.settings.backup_enabled = True

        # Configuración FAISS para investigación
        self.settings.faiss_config.update({
            'index_type': 'hnsw',
            'gpu': False,  # Para reproducibilidad
            'nprobe': 50
        })

    def _apply_demo_preset(self):
        """Configuración optimizada para demos"""
        self.settings.active_database = VectorDBType.CHROMADB
        self.settings.fallback_database = VectorDBType.MEMORY
        self.settings.auto_fallback = True
        self.settings.batch_size = 100
        self.settings.search_timeout = 15.0
        self.settings.backup_enabled = False

        # Configuración ChromaDB para demo
        self.settings.chromadb_config.update({
            'path': 'demo_data/chromadb',
            'collection_name': 'demo_audio_embeddings'
        })

    def validate_configuration(self) -> dict[str, bool]:
        """Valida la configuración actual"""
        validation_results = {}

        # Validar configuración principal
        validation_results['embedding_dimension_valid'] = self.settings.embedding_dimension > 0
        validation_results['similarity_metric_valid'] = self.settings.similarity_metric in ['cosine', 'euclidean', 'dot_product']

        # Validar configuraciones específicas
        if self.settings.active_database == VectorDBType.FAISS:
            validation_results['faiss_config_valid'] = self._validate_faiss_config()
        elif self.settings.active_database == VectorDBType.CHROMADB:
            validation_results['chromadb_config_valid'] = self._validate_chromadb_config()
        elif self.settings.active_database == VectorDBType.SUPABASE:
            validation_results['supabase_config_valid'] = self._validate_supabase_config()

        # Validar directorios
        validation_results['data_directory_valid'] = self._validate_data_directory()

        return validation_results

    def _validate_faiss_config(self) -> bool:
        """Valida configuración FAISS"""
        faiss_cfg = self.settings.faiss_config
        valid_index_types = ['flat', 'ivf', 'hnsw']
        return faiss_cfg.get('index_type', 'flat') in valid_index_types

    def _validate_chromadb_config(self) -> bool:
        """Valida configuración ChromaDB"""
        chromadb_cfg = self.settings.chromadb_config
        collection_name = chromadb_cfg.get('collection_name', '')
        return len(collection_name) > 0

    def _validate_supabase_config(self) -> bool:
        """Valida configuración Supabase"""
        supabase_cfg = self.settings.supabase_config
        return bool(supabase_cfg.get('url')) and bool(supabase_cfg.get('key'))

    def _validate_data_directory(self) -> bool:
        """Valida directorio de datos"""
        try:
            data_path = Path(self.settings.data_directory)
            data_path.mkdir(parents=True, exist_ok=True)
            return data_path.exists() and data_path.is_dir()
        except:
            return False

    def get_status_report(self) -> dict[str, Any]:
        """Genera reporte de estado de la configuración"""
        validation_results = self.validate_configuration()

        return {
            'active_database': self.settings.active_database.value,
            'embedding_dimension': self.settings.embedding_dimension,
            'similarity_metric': self.settings.similarity_metric,
            'configuration_valid': all(validation_results.values()),
            'validation_details': validation_results,
            'config_file': self.config_file,
            'fallback_enabled': self.settings.auto_fallback,
            'fallback_database': self.settings.fallback_database.value
        }

    def switch_database(self, new_db_type: VectorDBType, save_config: bool = True):
        """Cambia la base de datos activa"""
        old_db = self.settings.active_database
        self.settings.active_database = new_db_type

        if save_config:
            self.save_config()

        logger.info(f"🔄 Base de datos cambiada: {old_db.value} → {new_db_type.value}")

# Instancia global del configurador
_global_configurator: VectorDatabaseConfigurator | None = None

def get_configurator(config_file: str | None = None) -> VectorDatabaseConfigurator:
    """Obtiene la instancia global del configurador"""
    global _global_configurator

    if _global_configurator is None:
        _global_configurator = VectorDatabaseConfigurator(config_file)

    return _global_configurator

def create_example_configs():
    """Crea archivos de configuración de ejemplo"""
    configs = {
        'vector_db_config_development.json': ConfigurationPreset.DEVELOPMENT,
        'vector_db_config_production.json': ConfigurationPreset.PRODUCTION,
        'vector_db_config_research.json': ConfigurationPreset.RESEARCH,
        'vector_db_config_demo.json': ConfigurationPreset.DEMO
    }

    for config_file, preset in configs.items():
        configurator = VectorDatabaseConfigurator(config_file)
        configurator.apply_preset(preset)
        configurator.save_config()
        logger.info(f"📄 Configuración de ejemplo creada: {config_file}")

# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Crear configurador
    configurator = get_configurator()

    # Aplicar preset de desarrollo
    configurator.apply_preset(ConfigurationPreset.DEVELOPMENT)

    # Obtener configuración para FAISS
    faiss_config = configurator.get_vector_db_config(VectorDBType.FAISS)
    print(f"Configuración FAISS: {faiss_config}")

    # Generar reporte de estado
    status = configurator.get_status_report()
    print(f"Estado: {status}")

    # Crear configuraciones de ejemplo
    create_example_configs()
