import logging
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class TextEmbeddingGenerator:
    """
    Clase para generar embeddings de texto usando sentence-transformers
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa el generador de embeddings

        Args:
            model_name: Nombre del modelo de sentence-transformers
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Genera embeddings para una lista de textos

        Args:
            texts: Lista de textos para generar embeddings
            batch_size: Tamaño del lote para procesamiento

        Returns:
            Array numpy con los embeddings
        """
        # Filtrar textos vacíos
        valid_texts = [text for text in texts if text and text.strip()]

        if not valid_texts:
            return np.array([])

        # Generar embeddings en lotes
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings

    def process_transcription_dataframe(self, df: pd.DataFrame,
                                      text_column: str = 'text') -> pd.DataFrame:
        """
        Procesa un DataFrame con transcripciones y añade embeddings

        Args:
            df: DataFrame con transcripciones
            text_column: Nombre de la columna con el texto

        Returns:
            DataFrame con embeddings añadidos
        """
        if text_column not in df.columns:
            raise ValueError(f"Columna '{text_column}' no encontrada en el DataFrame")

        # Filtrar filas con texto válido
        valid_rows = df[df[text_column].notna() & (df[text_column].str.len() > 3)]

        if len(valid_rows) == 0:
            logging.info("No se encontraron textos válidos para procesar")
            return df

        logging.info(f"Generando embeddings para {len(valid_rows)} textos...")

        # Generar embeddings
        texts = valid_rows[text_column].tolist()
        embeddings = self.generate_embeddings(texts)

        # Crear DataFrame resultado
        result_df = valid_rows.copy()

        # Añadir embeddings como lista (para poder guardar en DataFrame)
        result_df['text_embedding'] = [emb.tolist() for emb in embeddings]
        result_df['embedding_model'] = self.model_name
        result_df['embedding_dim'] = self.embedding_dim

        return result_df

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Genera embedding para una consulta

        Args:
            query: Texto de la consulta

        Returns:
            Array numpy con el embedding de la consulta
        """
        if not query or not query.strip():
            return np.array([])

        embedding = self.model.encode([query.strip()], convert_to_numpy=True)
        return embedding[0]

    def calculate_similarity(self, query_embedding: np.ndarray,
                           text_embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula la similitud coseno entre una consulta y un conjunto de embeddings

        Args:
            query_embedding: Embedding de la consulta
            text_embeddings: Array con embeddings de texto

        Returns:
            Array con las similitudes
        """
        # Normalizar embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        text_norms = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        # Calcular similitud coseno
        similarities = np.dot(text_norms, query_norm)

        return similarities

    def save_embeddings(self, df: pd.DataFrame, file_path: str):
        """
        Guarda DataFrame con embeddings en un archivo

        Args:
            df: DataFrame con embeddings
            file_path: Ruta donde guardar el archivo
        """
        with open(file_path, 'wb') as f:
            pickle.dump(df, f)
        logging.info(f"Embeddings guardados en: {file_path}")

    def load_embeddings(self, file_path: str) -> pd.DataFrame:
        """
        Carga DataFrame con embeddings desde un archivo

        Args:
            file_path: Ruta del archivo a cargar

        Returns:
            DataFrame con embeddings
        """
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        logging.info(f"Embeddings cargados desde: {file_path}")
        return df

    def search_similar_texts(self, query: str, df: pd.DataFrame,
                           top_k: int = 5) -> pd.DataFrame:
        """
        Busca los textos más similares a una consulta

        Args:
            query: Consulta en texto natural
            df: DataFrame con embeddings de texto
            top_k: Número de resultados a retornar

        Returns:
            DataFrame con los resultados más similares
        """
        # Generar embedding de la consulta
        query_embedding = self.generate_query_embedding(query)

        if query_embedding.size == 0:
            return pd.DataFrame()

        # Obtener embeddings de los textos
        text_embeddings = np.array(df['text_embedding'].tolist())

        # Calcular similitudes
        similarities = self.calculate_similarity(query_embedding, text_embeddings)

        # Obtener top_k resultados
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Crear DataFrame resultado
        result_df = df.iloc[top_indices].copy()
        result_df['similarity_score'] = similarities[top_indices]
        result_df['query'] = query

        # Ordenar por similitud
        result_df = result_df.sort_values('similarity_score', ascending=False)

        return result_df


class TextPreprocessor:
    """
    Clase para preprocesamiento de texto antes de generar embeddings
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Limpia y normaliza un texto

        Args:
            text: Texto a limpiar

        Returns:
            Texto limpio
        """
        if not text:
            return ""

        # Limpiar espacios extra
        text = " ".join(text.split())

        # Remover caracteres especiales de transcripción
        text = text.replace("[inaudible]", "")
        text = text.replace("[música]", "")
        text = text.replace("[ruido]", "")

        return text.strip()

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Preprocesa un DataFrame con textos

        Args:
            df: DataFrame con textos
            text_column: Nombre de la columna con texto

        Returns:
            DataFrame con textos preprocesados
        """
        df_clean = df.copy()
        df_clean[text_column] = df_clean[text_column].apply(TextPreprocessor.clean_text)

        # Filtrar textos muy cortos
        df_clean = df_clean[df_clean[text_column].str.len() > 3]

        return df_clean


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de uso del generador de embeddings
    embedder = TextEmbeddingGenerator()

    # Datos de ejemplo
    sample_data = {
        'text': [
            "El presidente anunció nuevas medidas económicas",
            "Los mercados financieros mostraron volatilidad",
            "La inflación continúa siendo un desafío",
            "El congreso debate la nueva ley"
        ],
        'start_time': [0, 10, 20, 30],
        'source_file': ['audio1.wav'] * 4
    }

    df = pd.DataFrame(sample_data)

    # Generar embeddings
    # df_with_embeddings = embedder.process_transcription_dataframe(df)

    # Buscar textos similares
    # results = embedder.search_similar_texts("economía y inflación", df_with_embeddings)
    # print(results[['text', 'similarity_score']])

    logging.info("Módulo de embeddings de texto listo. Usar TextEmbeddingGenerator para procesar textos.")
