"""Motor de búsqueda semántica de audio adaptado para el servicio de agente"""

import ast
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Evitar warning de tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class AudioSearchEngine:
    """Sistema de búsqueda semántica local con FAISS para uso en servicio"""

    def __init__(self, dataset_path: str):
        """
        Inicializa el motor de búsqueda

        Args:
            dataset_path: Ruta al directorio del dataset
        """
        self.dataset_path = Path(dataset_path)
        self.text_model: SentenceTransformer | None = None
        self.df: pd.DataFrame | None = None
        self.faiss_index = None
        self.embeddings: np.ndarray | None = None

        self._load_dataset()
        self._load_text_model()

    def _load_dataset(self) -> None:
        """Carga el dataset local"""
        logger.info(f"Cargando dataset desde: {self.dataset_path}")

        # Buscar archivo de dataset
        possible_files = [
            self.dataset_path / "final" / "complete_dataset.pkl",
            self.dataset_path / "embeddings" / "segments_metadata.csv",
            self.dataset_path / "complete_dataset.pkl",
        ]

        dataset_file = None
        for f in possible_files:
            if f.exists():
                dataset_file = f
                break

        if not dataset_file:
            msg = f"No se encontró dataset en: {self.dataset_path}"
            logger.error(msg)
            raise FileNotFoundError(
                f"{msg}. Genera un dataset primero con: "
                "poetry run python src/simple_dataset_pipeline.py --input data/ --output ./dataset"
            )

        # Cargar según tipo de archivo
        if dataset_file.suffix == ".pkl":
            logger.info(f"Cargando pickle: {dataset_file.name}")
            with open(dataset_file, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, pd.DataFrame):
                    self.df = data
                elif isinstance(data, dict) and "dataframe" in data:
                    self.df = data["dataframe"]
                else:
                    msg = "Formato de pickle no reconocido"
                    logger.error(msg)
                    raise ValueError(msg)
        else:
            logger.info(f"Cargando CSV: {dataset_file.name}")
            self.df = pd.read_csv(dataset_file)

        logger.info(f"Dataset cargado: {len(self.df)} segmentos")

        # Cargar embeddings de texto
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Carga los embeddings de texto"""
        embeddings_dir = self.dataset_path / "embeddings" / "text_embeddings"

        # Primero intentar cargar desde columna del DataFrame
        if "text_embedding" in self.df.columns or "embedding" in self.df.columns:
            logger.info("Usando embeddings del DataFrame")
            emb_col = (
                "text_embedding" if "text_embedding" in self.df.columns else "embedding"
            )

            embeddings_list = []
            for idx, row in self.df.iterrows():
                emb = row[emb_col]
                if isinstance(emb, str):
                    # Parsear string a array
                    emb = np.array(ast.literal_eval(emb), dtype=np.float32)
                elif isinstance(emb, (list, np.ndarray)):
                    emb = np.array(emb, dtype=np.float32)
                else:
                    # Embedding vacío
                    emb = np.zeros(384, dtype=np.float32)
                embeddings_list.append(emb)

            self.embeddings = np.vstack(embeddings_list)
            logger.info(f"Embeddings cargados: {self.embeddings.shape}")
            self._build_faiss_index()
            return

        # Buscar archivos de embeddings individuales
        if embeddings_dir.exists():
            logger.info(f"Cargando embeddings desde: {embeddings_dir}")
            embeddings_list = []

            for idx in range(len(self.df)):
                emb_file = embeddings_dir / f"segment_{idx}_embedding.npy"
                if emb_file.exists():
                    emb = np.load(emb_file)
                    embeddings_list.append(emb)
                else:
                    # Embedding vacío si no existe
                    embeddings_list.append(np.zeros(384, dtype=np.float32))

            if embeddings_list:
                self.embeddings = np.vstack(embeddings_list)
                logger.info(f"Embeddings cargados: {self.embeddings.shape}")
                self._build_faiss_index()
                return

        # Buscar índice FAISS existente
        faiss_path = self.dataset_path / "indices" / "text_index.faiss"
        if faiss_path.exists() and FAISS_AVAILABLE:
            logger.info(f"Cargando índice FAISS: {faiss_path}")
            self.faiss_index = faiss.read_index(str(faiss_path))
            logger.info(f"Índice FAISS cargado: {self.faiss_index.ntotal} vectores")
            return

        logger.warning(
            "No se encontraron embeddings pre-calculados. "
            "Se generarán embeddings en tiempo de búsqueda (más lento)"
        )

    def _build_faiss_index(self) -> None:
        """Construye índice FAISS desde embeddings"""
        if self.embeddings is None or not FAISS_AVAILABLE:
            return

        logger.info("Construyendo índice FAISS...")
        dimension = self.embeddings.shape[1]

        # Normalizar para similitud coseno
        normalized = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        normalized = np.nan_to_num(normalized, nan=0.0)  # Manejar NaN

        # Crear índice con producto interno (equivalente a coseno con vectores normalizados)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(normalized.astype(np.float32))
        logger.info(f"Índice FAISS construido: {self.faiss_index.ntotal} vectores")

    def _load_text_model(self) -> None:
        """Carga el modelo de embeddings de texto"""
        try:
            logger.info("Cargando modelo de embeddings de texto...")
            self.text_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Modelo de texto cargado")
        except Exception as e:
            msg = f"Error cargando modelo: {e}"
            logger.error(msg)
            raise ImportError(f"{msg}. Instala: pip install sentence-transformers")

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding vectorial para el texto de búsqueda

        Args:
            text: Texto para generar embedding

        Returns:
            Array numpy con el embedding normalizado
        """
        if self.text_model is None:
            raise RuntimeError("Modelo de texto no cargado")
        embedding = self.text_model.encode(text)
        # Normalizar para similitud coseno
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32)

    def search_semantic(self, query_text: str, k: int = 5) -> list[dict]:
        """
        Realiza búsqueda semántica vectorial

        Args:
            query_text: Texto de búsqueda
            k: Número de resultados a retornar

        Returns:
            Lista de diccionarios con resultados de búsqueda
        """
        logger.info(f"Búsqueda semántica: '{query_text}'")

        # Generar embedding de la consulta
        query_embedding = self.generate_text_embedding(query_text)

        if self.faiss_index is not None and FAISS_AVAILABLE:
            # Búsqueda con FAISS (rápida)
            query_embedding = query_embedding.reshape(1, -1)
            similarities, indices = self.faiss_index.search(query_embedding, k)

            results = []
            for _i, (sim, idx) in enumerate(zip(similarities[0], indices[0], strict=False)):
                if idx >= 0 and idx < len(self.df):
                    row = self.df.iloc[idx]
                    results.append(
                        {
                            "segment": row.to_dict(),
                            "similarity": float(sim),
                            "distance": 1.0 - float(sim),
                        }
                    )

            logger.info(f"Búsqueda FAISS completada: {len(results)} resultados")
            return results

        if self.embeddings is not None:
            # Búsqueda manual con numpy (más lenta pero funciona)
            logger.info("Usando búsqueda numpy (más lenta)...")

            # Normalizar embeddings
            normalized = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )
            normalized = np.nan_to_num(normalized, nan=0.0)

            # Calcular similitudes
            similarities = np.dot(normalized, query_embedding)

            # Obtener top-k
            top_indices = np.argsort(similarities)[::-1][:k]

            results = []
            for idx in top_indices:
                row = self.df.iloc[idx]
                results.append(
                    {
                        "segment": row.to_dict(),
                        "similarity": float(similarities[idx]),
                        "distance": 1.0 - float(similarities[idx]),
                    }
                )

            logger.info(f"Búsqueda completada: {len(results)} resultados")
            return results

        # Generar embeddings en tiempo de ejecución (muy lento)
        logger.warning(
            "Generando embeddings en tiempo real (muy lento)..."
        )

        results = []
        for idx, row in self.df.iterrows():
            text = str(row.get("text", ""))
            if text:
                segment_embedding = self.generate_text_embedding(text)
                similarity = float(np.dot(query_embedding, segment_embedding))
                results.append(
                    {
                        "segment": row.to_dict(),
                        "similarity": similarity,
                        "distance": 1.0 - similarity,
                    }
                )

        # Ordenar por similitud
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def get_segment_info(self, segment_id: int) -> dict | None:
        """
        Obtiene información de un segmento específico

        Args:
            segment_id: ID del segmento

        Returns:
            Diccionario con información del segmento o None si no existe
        """
        if self.df is None:
            return None

        if segment_id < 0 or segment_id >= len(self.df):
            return None

        return self.df.iloc[segment_id].to_dict()
