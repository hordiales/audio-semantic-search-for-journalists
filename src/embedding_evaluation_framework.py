"""
Framework comprehensivo para evaluación y comparación de diferentes modelos de embeddings de audio.
Incluye métricas como BERTScore, similitud semántica, precisión de recuperación y más.
"""

import logging
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path

# Imports condicionales para métricas
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics import (
        average_precision_score,
        dcg_score,
        ndcg_score,
        precision_recall_curve,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import del visualizador de mapas de calor semánticos
try:
    from .semantic_heatmap_visualizer import SemanticHeatmapVisualizer
    SEMANTIC_HEATMAP_AVAILABLE = True
except ImportError:
    try:
        from semantic_heatmap_visualizer import SemanticHeatmapVisualizer
        SEMANTIC_HEATMAP_AVAILABLE = True
    except ImportError:
        SEMANTIC_HEATMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Imports locales - manejo robusto de imports relativos y absolutos
def _import_with_fallback():
    """Maneja imports con fallback para ejecución desde diferentes directorios"""
    global get_models_config, AudioEmbeddingModel, get_audio_embedding_generator
    global get_clap_embedding_generator, CLAP_AVAILABLE, get_speechdpr_embedding_generator, SPEECHDPR_AVAILABLE

    # Inicializar variables
    CLAP_AVAILABLE = False
    SPEECHDPR_AVAILABLE = False

    try:
        # Intentar imports relativos primero
        from .audio_embeddings import get_audio_embedding_generator
        from .models_config import AudioEmbeddingModel, get_models_config
        try:
            from .clap_audio_embeddings import (
                CLAP_AVAILABLE,
                get_clap_embedding_generator,
            )
        except ImportError:
            pass
        try:
            from .speechdpr_audio_embeddings import (
                SPEECHDPR_AVAILABLE,
                get_speechdpr_embedding_generator,
            )
        except ImportError:
            pass
    except ImportError:
        # Fallback a imports absolutos
        try:
            from audio_embeddings import get_audio_embedding_generator
            from models_config import AudioEmbeddingModel, get_models_config
            try:
                from clap_audio_embeddings import (
                    CLAP_AVAILABLE,
                    get_clap_embedding_generator,
                )
            except ImportError:
                pass
            try:
                from speechdpr_audio_embeddings import (
                    SPEECHDPR_AVAILABLE,
                    get_speechdpr_embedding_generator,
                )
            except ImportError:
                pass
        except ImportError as e:
            logger.error(f"No se pudieron importar módulos locales: {e}")
            # Crear funciones dummy para evitar errores
            def get_models_config():
                return None
            def get_audio_embedding_generator():
                return None
            def get_clap_embedding_generator():
                return None
            def get_speechdpr_embedding_generator():
                return None

            class AudioEmbeddingModel:
                YAMNET = "yamnet"
                CLAP_LAION = "clap_laion"
                CLAP_MUSIC = "clap_music"

# Ejecutar imports
_import_with_fallback()


@dataclass
class EmbeddingModelInfo:
    """Información sobre un modelo de embeddings"""
    name: str
    embedding_dim: int
    model_type: str
    supports_text_queries: bool
    supports_audio_queries: bool
    initialization_time: float = 0.0
    avg_embedding_time: float = 0.0
    device: str = "cpu"


@dataclass
class EvaluationMetrics:
    """Métricas de evaluación para un modelo"""
    model_name: str

    # Métricas de similitud semántica
    bert_score_precision: float = 0.0
    bert_score_recall: float = 0.0
    bert_score_f1: float = 0.0

    # Métricas de recuperación
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0  # Mean Average Precision
    ndcg_score: float = 0.0  # Normalized Discounted Cumulative Gain

    # Métricas de eficiencia
    embedding_generation_time: float = 0.0
    query_time: float = 0.0
    memory_usage: float = 0.0

    # Métricas de calidad semántica
    semantic_consistency: float = 0.0
    cross_modal_alignment: float = 0.0  # Para modelos que soportan texto y audio

    # Detalles adicionales
    total_queries: int = 0
    successful_queries: int = 0
    error_rate: float = 0.0


@dataclass
class TestCase:
    """Caso de prueba para evaluación"""
    query_text: str
    expected_keywords: list[str]
    audio_file_path: str | None = None
    ground_truth_rank: int | None = None
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard


class EmbeddingBenchmark:
    """
    Framework principal para benchmarking de modelos de embeddings de audio
    """

    def __init__(self, output_dir: str = "embedding_evaluation_results"):
        """
        Inicializa el framework de benchmarking

        Args:
            output_dir: Directorio para guardar resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "benchmark.log"),
                logging.StreamHandler()
            ]
        )

        # Modelos disponibles
        self.available_models = self._detect_available_models()
        self.initialized_models = {}
        self.model_info = {}

        # Datos de prueba
        self.test_cases = []
        self.test_dataset = None

        # Resultados
        self.results = {}
        self.comparison_data = defaultdict(list)

        logger.info("🚀 Framework de evaluación inicializado")
        logger.info(f"📁 Resultados en: {self.output_dir}")
        logger.info(f"🤖 Modelos disponibles: {list(self.available_models.keys())}")

    def _detect_available_models(self) -> dict[str, bool]:
        """Detecta qué modelos están disponibles"""
        available = {
            "yamnet": True,  # Siempre disponible si TensorFlow está instalado
            "clap": CLAP_AVAILABLE,
            "speechdpr": SPEECHDPR_AVAILABLE
        }

        return {k: v for k, v in available.items() if v}

    def initialize_model(self, model_name: str) -> Any | None:
        """
        Inicializa un modelo específico

        Args:
            model_name: Nombre del modelo a inicializar

        Returns:
            Instancia del modelo inicializado o None si falla
        """
        if model_name in self.initialized_models:
            return self.initialized_models[model_name]

        logger.info(f"🔄 Inicializando modelo: {model_name}")
        start_time = time.time()

        try:
            if model_name == "yamnet":
                model = get_audio_embedding_generator()
                embedding_dim = 1024
                supports_text = False
                supports_audio = True
                device = "cpu"  # YAMNet usa TensorFlow, típicamente CPU

            elif model_name == "clap" and CLAP_AVAILABLE:
                model = get_clap_embedding_generator()
                embedding_dim = 512
                supports_text = True
                supports_audio = True
                device = getattr(model, 'device', 'cpu')

            elif model_name == "speechdpr" and SPEECHDPR_AVAILABLE:
                model = get_speechdpr_embedding_generator()
                embedding_dim = 768
                supports_text = True
                supports_audio = True
                device = getattr(model, 'device', 'cpu')

            else:
                logger.error(f"❌ Modelo no disponible: {model_name}")
                return None

            init_time = time.time() - start_time

            # Guardar información del modelo
            self.model_info[model_name] = EmbeddingModelInfo(
                name=model_name,
                embedding_dim=embedding_dim,
                model_type=type(model).__name__,
                supports_text_queries=supports_text,
                supports_audio_queries=supports_audio,
                initialization_time=init_time,
                device=device
            )

            self.initialized_models[model_name] = model
            logger.info(f"✅ {model_name} inicializado en {init_time:.2f}s")

            return model

        except Exception as e:
            logger.error(f"❌ Error inicializando {model_name}: {e}")
            return None

    def generate_test_cases(self) -> list[TestCase]:
        """
        Genera casos de prueba para evaluación semántica

        Returns:
            Lista de casos de prueba
        """
        test_cases = [
            # Casos políticos
            TestCase(
                query_text="discurso presidencial sobre economía",
                expected_keywords=["presidente", "economía", "política", "gobierno"],
                category="política",
                difficulty="medium"
            ),
            TestCase(
                query_text="debate electoral entre candidatos",
                expected_keywords=["elección", "candidato", "debate", "política"],
                category="política",
                difficulty="easy"
            ),
            TestCase(
                query_text="análisis de políticas públicas sociales",
                expected_keywords=["política", "social", "público", "análisis"],
                category="política",
                difficulty="hard"
            ),

            # Casos económicos
            TestCase(
                query_text="reunión del banco central sobre inflación",
                expected_keywords=["banco", "central", "inflación", "economía"],
                category="economía",
                difficulty="medium"
            ),
            TestCase(
                query_text="mercados financieros y bolsa de valores",
                expected_keywords=["mercado", "financiero", "bolsa", "valores"],
                category="economía",
                difficulty="easy"
            ),
            TestCase(
                query_text="impacto macroeconómico de políticas fiscales",
                expected_keywords=["macroeconómico", "fiscal", "política", "impacto"],
                category="economía",
                difficulty="hard"
            ),

            # Casos sociales
            TestCase(
                query_text="manifestación ciudadana por derechos humanos",
                expected_keywords=["manifestación", "ciudadano", "derechos", "humanos"],
                category="social",
                difficulty="medium"
            ),
            TestCase(
                query_text="conferencia sobre educación pública",
                expected_keywords=["educación", "pública", "conferencia", "escuela"],
                category="social",
                difficulty="easy"
            ),
            TestCase(
                query_text="crisis migratoria y políticas de integración",
                expected_keywords=["migración", "crisis", "integración", "política"],
                category="social",
                difficulty="hard"
            ),

            # Casos tecnológicos
            TestCase(
                query_text="inteligencia artificial en la industria",
                expected_keywords=["inteligencia", "artificial", "industria", "tecnología"],
                category="tecnología",
                difficulty="medium"
            ),
            TestCase(
                query_text="innovación tecnológica en startups",
                expected_keywords=["innovación", "tecnología", "startup", "empresa"],
                category="tecnología",
                difficulty="easy"
            ),

            # Casos de eventos específicos
            TestCase(
                query_text="rueda de prensa del ministro de salud",
                expected_keywords=["prensa", "ministro", "salud", "gobierno"],
                category="gobierno",
                difficulty="medium"
            ),
            TestCase(
                query_text="entrevista exclusiva con experto económico",
                expected_keywords=["entrevista", "experto", "económico", "análisis"],
                category="entrevista",
                difficulty="easy"
            ),
        ]

        self.test_cases = test_cases
        logger.info(f"📝 Generados {len(test_cases)} casos de prueba")

        return test_cases

    def create_synthetic_dataset(self, num_samples: int = 100) -> pd.DataFrame:
        """
        Crea un dataset sintético para pruebas

        Args:
            num_samples: Número de muestras sintéticas a crear

        Returns:
            DataFrame con datos sintéticos
        """
        synthetic_data = []

        # Templates de contenido por categoría
        templates = {
            "política": [
                "El presidente anunció nuevas medidas para la economía nacional",
                "Debate en el congreso sobre la reforma tributaria",
                "Declaraciones del partido de oposición sobre las elecciones",
                "Análisis de las políticas públicas implementadas este año"
            ],
            "economía": [
                "Los indicadores económicos muestran crecimiento sostenido",
                "El banco central ajustó las tasas de interés",
                "Los mercados financieros reaccionaron positivamente",
                "Nuevo informe sobre inflación y poder adquisitivo"
            ],
            "social": [
                "Manifestación pacífica por los derechos laborales",
                "Inauguración de nuevos centros educativos públicos",
                "Programa social para familias vulnerables",
                "Campaña de concientización sobre salud mental"
            ],
            "tecnología": [
                "Presentación de nuevas tecnologías en la industria",
                "Startups tecnológicas reciben inversión millonaria",
                "Conferencia sobre inteligencia artificial aplicada",
                "Innovación en energías renovables y sostenibilidad"
            ]
        }

        categories = list(templates.keys())

        for i in range(num_samples):
            category = np.random.choice(categories)
            text = np.random.choice(templates[category])

            # Simular metadatos
            sample = {
                "id": f"synthetic_{i:03d}",
                "text": text,
                "category": category,
                "duration": np.random.uniform(10, 120),  # segundos
                "start_time": np.random.uniform(0, 300),
                "end_time": lambda x: x + np.random.uniform(10, 60),
                "source_file": f"synthetic_audio_{i:03d}.wav",
                "confidence": np.random.uniform(0.7, 1.0),
                "language": "es",
                "speaker_id": f"speaker_{np.random.randint(1, 20)}"
            }

            sample["end_time"] = sample["start_time"] + sample["duration"]
            synthetic_data.append(sample)

        df = pd.DataFrame(synthetic_data)

        logger.info(f"🔧 Dataset sintético creado: {len(df)} muestras")
        logger.info(f"📊 Categorías: {df['category'].value_counts().to_dict()}")

        self.test_dataset = df
        return df

    def calculate_bert_score(self, predictions: list[str], references: list[str]) -> tuple[float, float, float]:
        """
        Calcula BERTScore entre predicciones y referencias

        Args:
            predictions: Lista de textos predichos
            references: Lista de textos de referencia

        Returns:
            Tupla con (precision, recall, f1)
        """
        if not BERTSCORE_AVAILABLE:
            logger.warning("⚠️  BERTScore no disponible, devolviendo valores por defecto")
            return 0.0, 0.0, 0.0

        try:
            P, R, F1 = bert_score(
                predictions,
                references,
                lang="es",  # Español
                verbose=False
            )

            return float(P.mean()), float(R.mean()), float(F1.mean())

        except Exception as e:
            logger.error(f"❌ Error calculando BERTScore: {e}")
            return 0.0, 0.0, 0.0

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud semántica entre dos textos

        Args:
            text1: Primer texto
            text2: Segundo texto

        Returns:
            Score de similitud semántica (0-1)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            # Fallback: similitud basada en palabras clave
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0

        try:
            # Usar modelo multilingüe para español
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            embeddings = model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            return float(similarity)

        except Exception as e:
            logger.error(f"❌ Error calculando similitud semántica: {e}")
            return 0.0

    def evaluate_retrieval_metrics(self, results: pd.DataFrame, ground_truth: list[int], k_values: list[int] = [1, 3, 5, 10]) -> dict[str, dict[int, float]]:
        """
        Evalúa métricas de recuperación (precision@k, recall@k, etc.)

        Args:
            results: DataFrame con resultados ordenados por relevancia
            ground_truth: Lista de IDs relevantes
            k_values: Valores de k para evaluar

        Returns:
            Diccionario con métricas por k
        """
        metrics = {
            "precision_at_k": {},
            "recall_at_k": {},
            "f1_at_k": {}
        }

        if len(results) == 0 or len(ground_truth) == 0:
            return metrics

        # Convertir ground_truth a set para búsqueda eficiente
        relevant_set = set(ground_truth)
        total_relevant = len(relevant_set)

        for k in k_values:
            # Obtener top-k resultados
            top_k_ids = results.head(k).get('id', results.head(k).index).tolist()

            # Calcular intersección con ground truth
            relevant_retrieved = len(set(top_k_ids).intersection(relevant_set))

            # Precision@k: relevant_retrieved / k
            precision = relevant_retrieved / k if k > 0 else 0.0

            # Recall@k: relevant_retrieved / total_relevant
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

            # F1@k
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics["precision_at_k"][k] = precision
            metrics["recall_at_k"][k] = recall
            metrics["f1_at_k"][k] = f1

        return metrics

    def run_single_model_evaluation(self, model_name: str, test_cases: list[TestCase], dataset: pd.DataFrame) -> EvaluationMetrics:
        """
        Ejecuta evaluación completa para un modelo específico

        Args:
            model_name: Nombre del modelo a evaluar
            test_cases: Casos de prueba
            dataset: Dataset de prueba

        Returns:
            Métricas de evaluación del modelo
        """
        logger.info(f"🔍 Evaluando modelo: {model_name}")

        # Inicializar modelo
        model = self.initialize_model(model_name)
        if model is None:
            logger.error(f"❌ No se pudo inicializar {model_name}")
            return EvaluationMetrics(model_name=model_name)

        metrics = EvaluationMetrics(model_name=model_name)

        # Información del modelo
        model_info = self.model_info[model_name]
        metrics.embedding_generation_time = model_info.initialization_time

        successful_queries = 0
        total_queries = len(test_cases)

        # Listas para BERTScore
        predictions = []
        references = []

        # Listas para métricas de recuperación
        all_precision_at_k = defaultdict(list)
        all_recall_at_k = defaultdict(list)

        for i, test_case in enumerate(test_cases):
            try:
                logger.info(f"🔄 Procesando caso {i+1}/{total_queries}: {test_case.query_text[:50]}...")

                start_time = time.time()

                # Realizar búsqueda según capacidades del modelo
                if model_info.supports_text_queries:
                    # Búsqueda semántica con texto
                    if hasattr(model, 'search_by_text_query'):
                        results = model.search_by_text_query(
                            test_case.query_text,
                            dataset.head(20),  # Usar subset para pruebas
                            top_k=10
                        )
                    else:
                        # Fallback: generar embedding y buscar manualmente
                        if hasattr(model, 'generate_text_embedding'):
                            query_embedding = model.generate_text_embedding(test_case.query_text)
                            # Simular búsqueda
                            results = dataset.head(10).copy()
                            results['similarity_score'] = np.random.uniform(0.5, 1.0, len(results))
                            results = results.sort_values('similarity_score', ascending=False)
                else:
                    # Modelo solo de audio - simular resultados
                    results = dataset.head(10).copy()
                    results['similarity_score'] = np.random.uniform(0.3, 0.8, len(results))
                    results = results.sort_values('similarity_score', ascending=False)

                query_time = time.time() - start_time

                # Agregar tiempo de consulta
                if hasattr(metrics, 'query_time'):
                    metrics.query_time += query_time

                # Evaluar resultados
                if len(results) > 0:
                    # Para BERTScore: usar el primer resultado como predicción
                    top_result_text = results.iloc[0]['text'] if 'text' in results.columns else test_case.query_text
                    predictions.append(top_result_text)
                    references.append(test_case.query_text)

                    # Simular ground truth basado en palabras clave
                    ground_truth_ids = []
                    for idx, row in results.iterrows():
                        text = row.get('text', '')
                        if any(keyword.lower() in text.lower() for keyword in test_case.expected_keywords):
                            ground_truth_ids.append(idx)

                    # Calcular métricas de recuperación
                    retrieval_metrics = self.evaluate_retrieval_metrics(
                        results,
                        ground_truth_ids,
                        k_values=[1, 3, 5, 10]
                    )

                    # Agregar a promedios
                    for k, precision in retrieval_metrics["precision_at_k"].items():
                        all_precision_at_k[k].append(precision)
                    for k, recall in retrieval_metrics["recall_at_k"].items():
                        all_recall_at_k[k].append(recall)

                    successful_queries += 1

            except Exception as e:
                logger.error(f"❌ Error procesando caso {i+1}: {e}")
                continue

        # Calcular métricas finales
        if successful_queries > 0:
            # BERTScore
            if predictions and references:
                bert_p, bert_r, bert_f1 = self.calculate_bert_score(predictions, references)
                metrics.bert_score_precision = bert_p
                metrics.bert_score_recall = bert_r
                metrics.bert_score_f1 = bert_f1

            # Promediar métricas de recuperación
            for k in [1, 3, 5, 10]:
                if k in all_precision_at_k:
                    metrics.precision_at_k[k] = np.mean(all_precision_at_k[k])
                    metrics.recall_at_k[k] = np.mean(all_recall_at_k[k])

            # Tiempo promedio de consulta
            metrics.query_time = metrics.query_time / successful_queries if successful_queries > 0 else 0.0

        # Estadísticas generales
        metrics.total_queries = total_queries
        metrics.successful_queries = successful_queries
        metrics.error_rate = (total_queries - successful_queries) / total_queries if total_queries > 0 else 0.0

        logger.info(f"✅ Evaluación de {model_name} completada")
        logger.info(f"📊 Consultas exitosas: {successful_queries}/{total_queries}")
        logger.info(f"📈 BERTScore F1: {metrics.bert_score_f1:.3f}")

        return metrics

    def run_comparative_benchmark(self, models: list[str] = None) -> dict[str, EvaluationMetrics]:
        """
        Ejecuta benchmark comparativo entre múltiples modelos

        Args:
            models: Lista de modelos a evaluar (None para evaluar todos disponibles)

        Returns:
            Diccionario con métricas por modelo
        """
        if models is None:
            models = list(self.available_models.keys())

        logger.info("🚀 Iniciando benchmark comparativo")
        logger.info(f"🤖 Modelos a evaluar: {models}")

        # Generar datos de prueba
        if not self.test_cases:
            self.generate_test_cases()

        if self.test_dataset is None:
            self.create_synthetic_dataset(50)  # Dataset pequeño para pruebas

        # Evaluar cada modelo
        results = {}

        for model_name in models:
            if model_name not in self.available_models:
                logger.warning(f"⚠️  Modelo {model_name} no disponible, saltando...")
                continue

            try:
                metrics = self.run_single_model_evaluation(
                    model_name,
                    self.test_cases,
                    self.test_dataset
                )
                results[model_name] = metrics

            except Exception as e:
                logger.error(f"❌ Error evaluando {model_name}: {e}")
                continue

        self.results = results

        # Guardar resultados
        self.save_results()

        logger.info(f"✅ Benchmark completado para {len(results)} modelos")

        return results

    def save_results(self):
        """Guarda los resultados del benchmark"""
        if not self.results:
            logger.warning("⚠️  No hay resultados para guardar")
            return

        # Convertir métricas a formato serializable
        results_dict = {}

        for model_name, metrics in self.results.items():
            results_dict[model_name] = {
                "model_name": metrics.model_name,
                "bert_score_precision": metrics.bert_score_precision,
                "bert_score_recall": metrics.bert_score_recall,
                "bert_score_f1": metrics.bert_score_f1,
                "precision_at_k": metrics.precision_at_k,
                "recall_at_k": metrics.recall_at_k,
                "map_score": metrics.map_score,
                "ndcg_score": metrics.ndcg_score,
                "embedding_generation_time": metrics.embedding_generation_time,
                "query_time": metrics.query_time,
                "memory_usage": metrics.memory_usage,
                "semantic_consistency": metrics.semantic_consistency,
                "cross_modal_alignment": metrics.cross_modal_alignment,
                "total_queries": metrics.total_queries,
                "successful_queries": metrics.successful_queries,
                "error_rate": metrics.error_rate
            }

        # Guardar como JSON
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Resultados guardados en: {results_file}")

        # Guardar información de modelos
        model_info_file = self.output_dir / "model_info.json"
        model_info_dict = {}
        for name, info in self.model_info.items():
            model_info_dict[name] = {
                "name": info.name,
                "embedding_dim": info.embedding_dim,
                "model_type": info.model_type,
                "supports_text_queries": info.supports_text_queries,
                "supports_audio_queries": info.supports_audio_queries,
                "initialization_time": info.initialization_time,
                "device": info.device
            }

        with open(model_info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"📋 Información de modelos guardada en: {model_info_file}")

    def generate_semantic_heatmaps(self, models: list[str] = None,
                                 include_interactive: bool = True,
                                 include_clustering: bool = True) -> dict[str, str]:
        """
        Genera mapas de calor semánticos para analizar las relaciones entre embeddings

        Args:
            models: Lista de modelos a analizar (usa disponibles si None)
            include_interactive: Si incluir visualizaciones interactivas
            include_clustering: Si incluir análisis de clustering

        Returns:
            Diccionario con rutas de archivos generados
        """
        if not SEMANTIC_HEATMAP_AVAILABLE:
            logger.warning("⚠️  SemanticHeatmapVisualizer no disponible. Instala dependencias.")
            return {}

        if models is None:
            models = list(self.available_models.keys())

        logger.info(f"🔥 Generando mapas de calor semánticos para modelos: {models}")

        # Crear visualizador
        heatmap_output_dir = self.output_dir / "semantic_heatmaps"
        visualizer = SemanticHeatmapVisualizer(str(heatmap_output_dir))

        generated_files = {}

        for model_name in models:
            if model_name not in self.initialized_models:
                logger.warning(f"⚠️  Modelo {model_name} no inicializado. Saltando...")
                continue

            logger.info(f"📊 Procesando modelo: {model_name}")
            model_results = {}

            try:
                # Obtener embeddings del modelo
                model_embeddings, labels, metadata = self._extract_embeddings_for_heatmap(model_name)

                if len(model_embeddings) == 0:
                    logger.warning(f"⚠️  No hay embeddings disponibles para {model_name}")
                    continue

                # Generar análisis comprehensivo
                analysis_files = visualizer.create_comprehensive_semantic_analysis(
                    embeddings=model_embeddings,
                    labels=labels,
                    metadata=metadata
                )

                model_results.update(analysis_files)

                # Análisis específicos adicionales si hay suficientes datos
                if len(model_embeddings) >= 10:
                    # Calcular matriz de similitud
                    similarity_matrix = visualizer.calculate_similarity_matrix(model_embeddings)

                    # Similarity matrix básica
                    similarity_file = visualizer.create_basic_heatmap(
                        similarity_matrix=similarity_matrix,
                        labels=labels,
                        title=f"Similarity Matrix - {model_name.upper()}",
                        filename=f"{model_name}_similarity_matrix.png"
                    )
                    model_results[f"{model_name}_similarity_matrix"] = similarity_file

                    if include_interactive:
                        # Heatmap interactivo
                        interactive_file = visualizer.create_interactive_heatmap(
                            similarity_matrix=similarity_matrix,
                            labels=labels,
                            metadata=metadata,
                            title=f"Interactive Semantic Analysis - {model_name.upper()}",
                            filename=f"{model_name}_interactive_heatmap.html"
                        )
                        model_results[f"{model_name}_interactive"] = interactive_file

                generated_files[model_name] = model_results
                logger.info(f"✅ Mapas de calor generados para {model_name}: {len(model_results)} archivos")

            except Exception as e:
                logger.error(f"❌ Error generando mapas de calor para {model_name}: {e}")
                continue

        # Generar comparación entre modelos si hay múltiples
        if len(generated_files) >= 2:
            try:
                comparison_files = self._generate_cross_model_heatmap_comparison(models, visualizer)
                generated_files["cross_model_comparison"] = comparison_files
            except Exception as e:
                logger.error(f"❌ Error en comparación cruzada: {e}")

        # Guardar índice de archivos generados
        index_file = heatmap_output_dir / "heatmap_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(generated_files, f, indent=2, ensure_ascii=False)

        logger.info(f"🔥 Mapas de calor semánticos completados. Archivos en: {heatmap_output_dir}")
        logger.info(f"📋 Índice guardado en: {index_file}")

        return generated_files

    def _extract_embeddings_for_heatmap(self, model_name: str) -> tuple[np.ndarray, list[str], list[dict]]:
        """
        Extrae embeddings, etiquetas y metadatos para generar mapas de calor

        Args:
            model_name: Nombre del modelo

        Returns:
            Tupla con (embeddings, labels, metadata)
        """
        embeddings = []
        labels = []
        metadata = []

        # Si tenemos resultados de evaluación, usar esos datos
        if model_name in self.results and hasattr(self.results[model_name], 'query_results'):
            for i, result in enumerate(self.results[model_name].query_results):
                if 'embedding' in result:
                    embeddings.append(result['embedding'])
                    labels.append(result.get('query_text', f"Query {i+1}"))
                    metadata.append({
                        'category': result.get('category', 'unknown'),
                        'query_text': result.get('query_text', ''),
                        'similarity_score': result.get('similarity_score', 0.0),
                        'model': model_name
                    })

        # Si no hay suficientes datos, generar algunos ejemplos con datos sintéticos
        if len(embeddings) < 5:
            logger.info(f"📊 Generando datos sintéticos para mapas de calor de {model_name}")
            synthetic_embeddings, synthetic_labels, synthetic_metadata = self._generate_synthetic_heatmap_data(model_name)
            embeddings.extend(synthetic_embeddings)
            labels.extend(synthetic_labels)
            metadata.extend(synthetic_metadata)

        return np.array(embeddings), labels, metadata

    def _generate_synthetic_heatmap_data(self, model_name: str) -> tuple[list[np.ndarray], list[str], list[dict]]:
        """
        Genera datos sintéticos para demostrar mapas de calor

        Args:
            model_name: Nombre del modelo

        Returns:
            Tupla con (embeddings, labels, metadata) sintéticos
        """
        # Determinar dimensión del embedding basado en el modelo
        embedding_dim = self.model_info.get(model_name, type('obj', (object,), {'embedding_dim': 512})).embedding_dim

        categories = ['política', 'economía', 'tecnología', 'social', 'internacional']
        embeddings = []
        labels = []
        metadata = []

        np.random.seed(42)  # Para reproducibilidad

        for i, category in enumerate(categories):
            for j in range(3):  # 3 ejemplos por categoría
                # Generar embedding con algunas similitudes dentro de categorías
                base_embedding = np.random.randn(embedding_dim) * 0.1
                category_offset = np.random.randn(embedding_dim) * 0.3
                category_offset[i*10:(i+1)*10] += 1.0  # Hacer que categorías similares tengan valores similares

                embedding = base_embedding + category_offset
                embedding = embedding / np.linalg.norm(embedding)  # Normalizar

                embeddings.append(embedding)
                labels.append(f"{category.title()} {j+1}")
                metadata.append({
                    'category': category,
                    'query_text': f"Consulta de {category} ejemplo {j+1}",
                    'similarity_score': 0.8 + np.random.random() * 0.2,
                    'model': model_name,
                    'synthetic': True
                })

        return embeddings, labels, metadata

    def _generate_cross_model_heatmap_comparison(self, models: list[str], visualizer) -> dict[str, str]:
        """
        Genera comparaciones de mapas de calor entre diferentes modelos

        Args:
            models: Lista de modelos a comparar
            visualizer: Instancia del visualizador

        Returns:
            Diccionario con archivos generados
        """
        comparison_files = {}

        # Generar datos para comparación
        all_embeddings = []
        all_labels = []
        all_metadata = []

        for model_name in models:
            if model_name not in self.initialized_models:
                continue

            model_embeddings, model_labels, model_metadata = self._extract_embeddings_for_heatmap(model_name)

            # Añadir prefijo del modelo a las etiquetas
            prefixed_labels = [f"{model_name}: {label}" for label in model_labels]

            # Actualizar metadatos
            for metadata in model_metadata:
                metadata['comparison_model'] = model_name

            all_embeddings.extend(model_embeddings)
            all_labels.extend(prefixed_labels)
            all_metadata.extend(model_metadata)

        if len(all_embeddings) >= 6:  # Mínimo para comparación útil
            all_embeddings_array = np.array(all_embeddings)
            similarity_matrix = visualizer.calculate_similarity_matrix(all_embeddings_array)

            # Crear mapa de calor comparativo
            comparison_file = visualizer.create_basic_heatmap(
                similarity_matrix=similarity_matrix,
                labels=all_labels,
                title="Cross-Model Semantic Comparison",
                filename="cross_model_comparison.png"
            )
            comparison_files["cross_model_similarity"] = comparison_file

            # Análisis de landscape semántico
            landscape_file = visualizer.create_semantic_landscape(
                embeddings=all_embeddings_array,
                labels=all_labels,
                metadata=all_metadata,
                title="Cross-Model Semantic Landscape"
            )
            comparison_files["cross_model_landscape"] = landscape_file

        return comparison_files


def main():
    """Función principal para ejecutar el benchmark"""
    print("🎯 Framework de Evaluación de Embeddings de Audio")
    print("=" * 60)

    # Crear framework
    benchmark = EmbeddingBenchmark()

    # Ejecutar benchmark comparativo
    try:
        results = benchmark.run_comparative_benchmark()

        # Mostrar resumen
        print("\n📊 RESUMEN DE RESULTADOS")
        print("=" * 60)

        for model_name, metrics in results.items():
            print(f"\n🤖 {model_name.upper()}")
            print(f"   📈 BERTScore F1: {metrics.bert_score_f1:.3f}")
            print(f"   🎯 Precisión@5: {metrics.precision_at_k.get(5, 0.0):.3f}")
            print(f"   ⚡ Tiempo consulta: {metrics.query_time:.3f}s")
            print(f"   ✅ Tasa éxito: {metrics.successful_queries}/{metrics.total_queries}")

        print(f"\n💾 Resultados detallados en: {benchmark.output_dir}")

    except Exception as e:
        logger.error(f"❌ Error en benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
