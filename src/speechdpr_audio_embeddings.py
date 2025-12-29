"""
Implementación de SpeechDPR (Speech Dense Passage Retrieval) para embeddings de audio.
SpeechDPR es un enfoque más simple que el contrastivo, basado en Dense Passage Retrieval
para búsqueda semántica directa sobre audio sin requerir transcripción.

Esta implementación se basa en:
- HuBERT como speech encoder
- Dense Passage Retrieval (DPR) architecture
- Embeddings directos de audio para búsqueda semántica
"""

import logging
import os
import warnings

import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from .audio_embeddings import BaseAudioEmbedding
    from .models_config import SpeechDPRConfig, get_models_config
except ImportError:
    # Para ejecutar como script independiente
    from audio_embeddings import BaseAudioEmbedding
    from models_config import SpeechDPRConfig, get_models_config

logger = logging.getLogger(__name__)

# Verificar disponibilidad de dependencias
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoProcessor, HubertModel, RobertaModel, RobertaTokenizer
    SPEECHDPR_AVAILABLE = True
    logger.info("✅ SpeechDPR dependencies disponibles")
except ImportError as e:
    SPEECHDPR_AVAILABLE = False
    logger.warning(f"⚠️  SpeechDPR no disponible. Dependencias faltantes: {e}")
    logger.warning("Instala con: pip install transformers torch")


class FeatureProcessor(nn.Module):
    """
    Procesador de características que reduce la secuencia temporal
    usando CNNs con stride como en el paper original
    """
    def __init__(self, input_dim: int = 1024, output_dim: int = 768):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(input_dim)

        # CNN de dos capas para reducir longitud de secuencia
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=4, padding=1)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=3, padding=1)

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Tensor de forma (batch_size, seq_len, input_dim)
        Returns:
            Tensor procesado de forma (batch_size, reduced_seq_len, output_dim)
        """
        # Transponer para CNN: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Normalización de instancia
        x = self.instance_norm(x)

        # Primera capa CNN
        x = self.conv1(x)
        x = self.activation(x)

        # Segunda capa CNN
        x = self.conv2(x)
        x = self.activation(x)

        # Volver a transponer: (batch_size, seq_len, output_dim)
        x = x.transpose(1, 2)

        return x


class SpeechDPREmbedding(BaseAudioEmbedding):
    """
    Implementación de embeddings de audio usando arquitectura SpeechDPR

    SpeechDPR es más simple que los enfoques contrastivos porque:
    - No requiere pares audio-texto para entrenamiento
    - Usa arquitectura Dense Passage Retrieval estándar
    - Genera embeddings directos compatibles con búsqueda semántica
    - Más eficiente computacionalmente
    """

    def __init__(self, config: SpeechDPRConfig | None = None):
        """
        Inicializa el generador SpeechDPR

        Args:
            config: Configuración específica de SpeechDPR
        """
        if not SPEECHDPR_AVAILABLE:
            raise RuntimeError("SpeechDPR no disponible. Instala: pip install transformers torch")

        super().__init__()

        # Cargar configuración
        if config is None:
            models_config = get_models_config()
            self.config = models_config.speechdpr_config if models_config else SpeechDPRConfig()
        else:
            self.config = config
        self.device = self._get_device()
        self.model_name = "SpeechDPR"
        self.embedding_dim = self.config.embedding_dim
        self.sample_rate = self.config.sample_rate

        # Componentes del modelo
        self.speech_encoder = None
        self.audio_processor = None
        self.feature_processor = None
        self.passage_encoder = None
        self.tokenizer = None

        self._load_model()

    def _get_device(self) -> str:
        """Determina el dispositivo a usar"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.config.device

    def _load_model(self):
        """Carga todos los componentes del modelo SpeechDPR"""
        try:
            logger.info("🔄 Cargando componentes SpeechDPR...")

            # 1. Speech Encoder (HuBERT)
            self.speech_encoder = HubertModel.from_pretrained(
                self.config.speech_encoder_model
            ).to(self.device)

            # Para HuBERT, usar el processor directamente sin AutoProcessor
            try:
                self.audio_processor = AutoProcessor.from_pretrained(
                    self.config.speech_encoder_model
                )
            except Exception:
                # Fallback: usar el processor de wav2vec2 que es compatible
                from transformers import Wav2Vec2Processor
                self.audio_processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )

            # Congelar parámetros del speech encoder
            for param in self.speech_encoder.parameters():
                param.requires_grad = False

            # 2. Feature Processor
            hubert_dim = self.speech_encoder.config.hidden_size  # 1024 para HuBERT-large
            self.feature_processor = FeatureProcessor(
                input_dim=hubert_dim,
                output_dim=self.config.embedding_dim
            ).to(self.device)

            # 3. Passage Encoder (RoBERTa)
            self.passage_encoder = RobertaModel.from_pretrained(
                self.config.text_encoder_model
            ).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(
                self.config.text_encoder_model
            )

            logger.info(f"✅ SpeechDPR cargado en {self.device}")
            logger.info(f"📐 Dimensión de embeddings: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"❌ Error cargando SpeechDPR: {e}")
            raise RuntimeError(f"No se pudo cargar SpeechDPR: {e}")

    def preprocess_audio(self, audio_path: str, target_sr: int | None = None) -> torch.Tensor:
        """
        Preprocesa audio para SpeechDPR

        Args:
            audio_path: Ruta al archivo de audio
            target_sr: Frecuencia de muestreo objetivo (usa configuración si None)

        Returns:
            Tensor de audio preprocesado
        """
        # Usar frecuencia de muestreo de la configuración si no se especifica
        if target_sr is None:
            target_sr = self.sample_rate

        # Cargar audio
        audio, sr = librosa.load(
            audio_path,
            sr=target_sr,
            mono=True
        )

        # Limitar duración
        max_samples = int(self.config.max_audio_length * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Procesar con HuBERT processor
        inputs = self.audio_processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        )

        return inputs["input_values"].to(self.device)

    def extract_speech_features(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extrae características de audio usando HuBERT

        Args:
            audio_tensor: Tensor de audio preprocesado

        Returns:
            Características extraídas de la capa especificada
        """
        with torch.no_grad():
            # Obtener representaciones de HuBERT
            outputs = self.speech_encoder(
                audio_tensor,
                output_hidden_states=True
            )

            # Extraer de la capa especificada (22 por defecto)
            speech_features = outputs.hidden_states[self.config.feature_layer]

        return speech_features

    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """
        Genera embedding de audio usando SpeechDPR

        Args:
            audio_path: Ruta al archivo de audio

        Returns:
            Array numpy con el embedding de audio
        """
        try:
            # 1. Preprocesar audio
            audio_tensor = self.preprocess_audio(audio_path)

            # 2. Extraer características con HuBERT
            speech_features = self.extract_speech_features(audio_tensor)

            # 3. Procesar características
            processed_features = self.feature_processor(speech_features)

            # 4. Pooling para obtener representación de nivel de pasaje
            # Usar mean pooling sobre la dimensión temporal
            passage_embedding = torch.mean(processed_features, dim=1)  # (batch_size, embedding_dim)

            # 5. Normalización opcional
            if self.config.normalize_embeddings:
                passage_embedding = F.normalize(passage_embedding, p=2, dim=1)

            # Convertir a numpy
            embedding = passage_embedding.cpu().numpy().flatten()

            return embedding

        except Exception as e:
            logger.error(f"❌ Error generando embedding SpeechDPR para {audio_path}: {e}")
            raise RuntimeError(f"Error procesando audio con SpeechDPR: {e}")

    def generate_embeddings_batch(self, audio_paths: list[str], batch_size: int = 4) -> np.ndarray:
        """
        Genera embeddings para múltiples archivos de audio

        Args:
            audio_paths: Lista de rutas de audio
            batch_size: Tamaño de lote para procesamiento

        Returns:
            Array numpy con todos los embeddings
        """
        embeddings = []

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            batch_embeddings = []

            for audio_path in batch_paths:
                try:
                    if os.environ.get('MCP_MODE') != '1':
                        logger.info(f"📊 Procesando {i+len(batch_embeddings)+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")

                    embedding = self.generate_embedding(audio_path)
                    batch_embeddings.append(embedding)

                except Exception as e:
                    logger.warning(f"⚠️  Error procesando {audio_path}: {e}")
                    # Embedding cero para mantener consistencia
                    batch_embeddings.append(np.zeros(self.embedding_dim))

            embeddings.extend(batch_embeddings)

            # Limpieza de memoria
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()

        return np.array(embeddings)

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding de texto usando el passage encoder

        Args:
            text: Texto para generar embedding

        Returns:
            Array numpy con el embedding de texto
        """
        try:
            # Tokenizar texto
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generar embedding con RoBERTa
            with torch.no_grad():
                outputs = self.passage_encoder(**inputs)
                # Usar [CLS] token para representación de nivel de oración
                text_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

            # Normalización opcional
            if self.config.normalize_embeddings:
                text_embedding = F.normalize(text_embedding, p=2, dim=1)

            return text_embedding.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"❌ Error generando embedding de texto: {e}")
            raise RuntimeError(f"Error procesando texto con SpeechDPR: {e}")

    def process_transcription_dataframe(self, df: pd.DataFrame,
                                      temp_audio_dir: str = "temp_audio_speechdpr") -> pd.DataFrame:
        """
        Procesa DataFrame con transcripciones y genera embeddings SpeechDPR

        Args:
            df: DataFrame con información de segmentos
            temp_audio_dir: Directorio temporal para archivos

        Returns:
            DataFrame con embeddings SpeechDPR añadidos
        """
        if not os.path.exists(temp_audio_dir):
            os.makedirs(temp_audio_dir)

        # Generar archivos temporales para cada segmento
        audio_paths = []
        valid_indices = []

        for idx, row in df.iterrows():
            try:
                temp_audio_path = os.path.join(temp_audio_dir, f"speechdpr_segment_{idx}.wav")

                # Cargar y extraer segmento
                audio, sr = librosa.load(row['source_file'], sr=self.config.sample_rate, mono=True)
                start_sample = int(row['start_time'] * sr)
                end_sample = int(row['end_time'] * sr)

                segment_audio = audio[start_sample:end_sample]

                # Guardar segmento
                import soundfile as sf
                sf.write(temp_audio_path, segment_audio, sr)

                audio_paths.append(temp_audio_path)
                valid_indices.append(idx)

            except Exception as e:
                logger.error(f"❌ Error procesando segmento {idx}: {e}")
                continue

        if not audio_paths:
            logger.error("❌ No se pudieron procesar segmentos para SpeechDPR")
            return df

        # Generar embeddings
        if os.environ.get('MCP_MODE') != '1':
            logger.info(f"🔄 Generando embeddings SpeechDPR para {len(audio_paths)} segmentos...")

        embeddings = self.generate_embeddings_batch(audio_paths)

        # Añadir al DataFrame
        result_df = df.copy()
        result_df['audio_embedding_speechdpr'] = None

        for i, idx in enumerate(valid_indices):
            result_df.at[idx, 'audio_embedding_speechdpr'] = embeddings[i].tolist()

        result_df['audio_embedding_model'] = "SpeechDPR"
        result_df['audio_embedding_dim'] = self.embedding_dim

        # Limpiar archivos temporales
        for audio_path in audio_paths:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        # Filtrar filas válidas
        result_df = result_df[result_df['audio_embedding_speechdpr'].notna()]

        if os.environ.get('MCP_MODE') != '1':
            logger.info(f"✅ {len(result_df)} segmentos procesados con SpeechDPR")

        return result_df

    def search_by_text_query(self, query: str, df: pd.DataFrame,
                           top_k: int = 5, embedding_column: str = 'audio_embedding_speechdpr') -> pd.DataFrame:
        """
        Busca audio usando consulta de texto (característica principal de SpeechDPR)

        Args:
            query: Consulta en texto natural
            df: DataFrame con embeddings de audio
            top_k: Número de resultados
            embedding_column: Columna con embeddings

        Returns:
            DataFrame con resultados ordenados por similitud
        """
        if embedding_column not in df.columns:
            raise ValueError(f"Columna {embedding_column} no encontrada")

        # Generar embedding de consulta
        query_embedding = self.generate_text_embedding(query)

        # Obtener embeddings de audio
        audio_embeddings = np.array(df[embedding_column].tolist())

        # Calcular similitudes coseno
        similarities = np.dot(audio_embeddings, query_embedding)

        # Top-k resultados
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Crear DataFrame resultado
        result_df = df.iloc[top_indices].copy()
        result_df['speechdpr_similarity_score'] = similarities[top_indices]
        result_df['query_text'] = query

        return result_df.sort_values('speechdpr_similarity_score', ascending=False)


# Alias para compatibilidad con código existente
SpeechDPRAudioEmbeddingGenerator = SpeechDPREmbedding


def get_speechdpr_embedding_generator(config: SpeechDPRConfig | None = None) -> SpeechDPREmbedding:
    """
    Factory function para obtener generador SpeechDPR

    Args:
        config: Configuración específica

    Returns:
        Instancia del generador SpeechDPR
    """
    return SpeechDPREmbedding(config)


if __name__ == "__main__":
    # Ejemplo de uso y prueba
    if SPEECHDPR_AVAILABLE:
        print("🎤 Generador de Embeddings SpeechDPR")
        print("=" * 50)

        try:
            embedder = get_speechdpr_embedding_generator()
            print("✅ Modelo SpeechDPR inicializado correctamente")
            print(f"📐 Dimensiones de embedding: {embedder.embedding_dim}")
            print(f"🖥️  Device: {embedder.device}")

            # Ejemplo de embedding de texto
            text_query = "discurso político sobre economía"
            text_embedding = embedder.generate_text_embedding(text_query)
            print(f"🔤 Embedding de texto generado: {text_embedding.shape}")

            print("\n💡 Características de SpeechDPR:")
            print("  🎯 Búsqueda semántica directa sin transcripción")
            print("  🔄 Arquitectura más simple que métodos contrastivos")
            print("  📊 Basado en Dense Passage Retrieval")
            print("  🚀 Eficiente para consultas de texto sobre audio")
            print("  📈 No requiere pares audio-texto para entrenamiento")

        except Exception as e:
            print(f"❌ Error inicializando SpeechDPR: {e}")
            print("💡 Asegúrate de tener instalado: pip install transformers torch")
    else:
        print("❌ SpeechDPR no disponible")
        print("💡 Para instalar: pip install transformers torch")
