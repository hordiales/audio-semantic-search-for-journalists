#!/usr/bin/env python3
"""
CLI Interactivo para Búsqueda Semántica de Audio (Local con FAISS)
===================================================================

Interfaz de línea de comandos (CLI) interactiva para buscar contenido en audios
usando un dataset local con índices FAISS.

No requiere Supabase ni conexión a internet. Solo necesita el dataset generado
con simple_dataset_pipeline.py o run_dataset_pipeline.py.

Uso:
    poetry run python examples/demos/cli_audio_search.py [DATASET_PATH]

    # Ejemplo:
    poetry run python examples/demos/cli_audio_search.py ./dataset
"""

import argparse
import contextlib
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import ffmpeg
    FFMPEG_PYTHON_AVAILABLE = True
except ImportError:
    FFMPEG_PYTHON_AVAILABLE = False

# Evitar warning de tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LocalAudioSearch:
    """Sistema de búsqueda semántica local con FAISS"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.text_model = None
        self.df = None
        self.faiss_index = None
        self.embeddings = None

        self._load_dataset()
        self._load_text_model()

    def _load_dataset(self):
        """Carga el dataset local"""
        print(f"📂 Cargando dataset desde: {self.dataset_path}")

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
            print(f"❌ No se encontró dataset en: {self.dataset_path}")
            print("💡 Genera un dataset primero con:")
            print("   poetry run python src/simple_dataset_pipeline.py --input data/ --output ./dataset")
            sys.exit(1)

        # Cargar según tipo de archivo
        if dataset_file.suffix == ".pkl":
            print(f"   📦 Cargando pickle: {dataset_file.name}")
            with open(dataset_file, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, pd.DataFrame):
                    self.df = data
                elif isinstance(data, dict) and "dataframe" in data:
                    self.df = data["dataframe"]
                else:
                    print("❌ Formato de pickle no reconocido")
                    sys.exit(1)
        else:
            print(f"   📊 Cargando CSV: {dataset_file.name}")
            self.df = pd.read_csv(dataset_file)

        print(f"✅ Dataset cargado: {len(self.df)} segmentos")

        # Cargar embeddings de texto
        self._load_embeddings()

    def _load_embeddings(self):
        """Carga los embeddings de texto"""
        embeddings_dir = self.dataset_path / "embeddings" / "text_embeddings"

        # Primero intentar cargar desde columna del DataFrame
        if 'text_embedding' in self.df.columns or 'embedding' in self.df.columns:
            print("   📊 Usando embeddings del DataFrame")
            emb_col = 'text_embedding' if 'text_embedding' in self.df.columns else 'embedding'

            embeddings_list = []
            for idx, row in self.df.iterrows():
                emb = row[emb_col]
                if isinstance(emb, str):
                    # Parsear string a array
                    import ast
                    emb = np.array(ast.literal_eval(emb), dtype=np.float32)
                elif isinstance(emb, (list, np.ndarray)):
                    emb = np.array(emb, dtype=np.float32)
                else:
                    # Embedding vacío
                    emb = np.zeros(384, dtype=np.float32)
                embeddings_list.append(emb)

            self.embeddings = np.vstack(embeddings_list)
            print(f"   ✅ Embeddings cargados: {self.embeddings.shape}")
            self._build_faiss_index()
            return

        # Buscar archivos de embeddings individuales
        if embeddings_dir.exists():
            print(f"   📁 Cargando embeddings desde: {embeddings_dir}")
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
                print(f"   ✅ Embeddings cargados: {self.embeddings.shape}")
                self._build_faiss_index()
                return

        # Buscar índice FAISS existente
        faiss_path = self.dataset_path / "indices" / "text_index.faiss"
        if faiss_path.exists() and FAISS_AVAILABLE:
            print(f"   📁 Cargando índice FAISS: {faiss_path}")
            self.faiss_index = faiss.read_index(str(faiss_path))
            print(f"   ✅ Índice FAISS cargado: {self.faiss_index.ntotal} vectores")
            return

        print("⚠️  No se encontraron embeddings pre-calculados")
        print("   Se generarán embeddings en tiempo de búsqueda (más lento)")

    def _build_faiss_index(self):
        """Construye índice FAISS desde embeddings"""
        if self.embeddings is None or not FAISS_AVAILABLE:
            return

        print("   🔧 Construyendo índice FAISS...")
        dimension = self.embeddings.shape[1]

        # Normalizar para similitud coseno
        normalized = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized = np.nan_to_num(normalized, nan=0.0)  # Manejar NaN

        # Crear índice con producto interno (equivalente a coseno con vectores normalizados)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(normalized.astype(np.float32))
        print(f"   ✅ Índice FAISS construido: {self.faiss_index.ntotal} vectores")

    def _load_text_model(self):
        """Carga el modelo de embeddings de texto"""
        try:
            print("🤖 Cargando modelo de embeddings de texto...")
            self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("✅ Modelo de texto cargado")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            print("💡 Instala: pip install sentence-transformers")
            sys.exit(1)

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Genera embedding vectorial para el texto de búsqueda"""
        embedding = self.text_model.encode(text)
        # Normalizar para similitud coseno
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32)

    def search_semantic(self, query_text: str, k: int = 5) -> list[dict]:
        """Realiza búsqueda semántica vectorial"""
        print(f"🔍 Búsqueda semántica: '{query_text}'")
        print("-" * 50)

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
                    results.append({
                        'segment': row.to_dict(),
                        'similarity': float(sim),
                        'distance': 1.0 - float(sim)
                    })

            print(f"✅ Búsqueda FAISS completada: {len(results)} resultados")
            return results

        if self.embeddings is not None:
            # Búsqueda manual con numpy (más lenta pero funciona)
            print("   🐢 Usando búsqueda numpy (más lenta)...")

            # Normalizar embeddings
            normalized = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            normalized = np.nan_to_num(normalized, nan=0.0)

            # Calcular similitudes
            similarities = np.dot(normalized, query_embedding)

            # Obtener top-k
            top_indices = np.argsort(similarities)[::-1][:k]

            results = []
            for idx in top_indices:
                row = self.df.iloc[idx]
                results.append({
                    'segment': row.to_dict(),
                    'similarity': float(similarities[idx]),
                    'distance': 1.0 - float(similarities[idx])
                })

            print(f"✅ Búsqueda completada: {len(results)} resultados")
            return results

        # Generar embeddings en tiempo de ejecución (muy lento)
        print("   🐌 Generando embeddings en tiempo real (muy lento)...")

        results = []
        for idx, row in self.df.iterrows():
            text = str(row.get('text', ''))
            if text:
                segment_embedding = self.generate_text_embedding(text)
                similarity = float(np.dot(query_embedding, segment_embedding))
                results.append({
                    'segment': row.to_dict(),
                    'similarity': similarity,
                    'distance': 1.0 - similarity
                })

        # Ordenar por similitud
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]

    def display_results(self, results: list[dict], query_text: str):
        """Muestra los resultados de búsqueda"""
        if not results:
            print(f"❌ No se encontraron resultados para '{query_text}'")
            return

        print(f"\n✅ Encontrados {len(results)} resultados más relevantes:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            segment = result['segment']
            similarity = result['similarity']

            # Información del segmento
            segment_id = segment.get('segment_id', i - 1)
            language = segment.get('language', 'N/A')
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            duration = segment.get('duration', end_time - start_time)
            original_file = segment.get('original_file_name', segment.get('source_file', 'N/A'))
            confidence = segment.get('confidence')
            text = segment.get('text', '')

            print(f"🎯 RESULTADO {i}")
            print(f"   📋 ID: {segment_id}")
            print(f"   🌐 Idioma: {language}")
            print(f"   📊 Similitud: {similarity:.4f} ({similarity*100:.1f}%)")
            print(f"   ⏱️  Tiempo: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
            print(f"   📁 Archivo: {original_file}")

            if confidence:
                print(f"   🎯 Confianza transcripción: {confidence:.3f}")

            # Resaltar términos de búsqueda en el texto
            highlighted_text = text
            for word in query_text.lower().split():
                if len(word) > 2 and word in text.lower():
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    highlighted_text = pattern.sub(f"**{word.upper()}**", highlighted_text)

            print(f"   📝 Texto: {highlighted_text}")
            print()

    def find_audio_file(self, original_file_name: str) -> str | None:
        """Encuentra el archivo de audio correspondiente"""
        # Buscar en dataset/converted/
        search_paths = [
            self.dataset_path / "converted" / original_file_name,
            self.dataset_path / "converted" / Path(original_file_name).name,
            Path("dataset/converted") / original_file_name,
        ]

        for audio_path in search_paths:
            if audio_path.exists():
                return str(audio_path)

        # Buscar por nombre similar
        converted_dir = self.dataset_path / "converted"
        if converted_dir.exists():
            base_name = Path(original_file_name).stem
            for f in converted_dir.glob("*.wav"):
                if base_name in f.stem:
                    return str(f)

        print(f"⚠️  Archivo de audio no encontrado: {original_file_name}")
        return None

    def extract_audio_segment(self, audio_file: str, start_time: float, end_time: float, segment_id) -> str | None:
        """Extrae el segmento específico del audio usando ffmpeg"""
        try:
            # Crear directorio temporal
            temp_dir = Path("temp_audio_segments")
            temp_dir.mkdir(exist_ok=True)

            # Archivo temporal para el segmento
            temp_file = temp_dir / f"segment_{segment_id}_{start_time:.1f}s.wav"

            # Usar ffmpeg-python si está disponible
            if FFMPEG_PYTHON_AVAILABLE:
                try:
                    duration = end_time - start_time
                    stream = ffmpeg.input(audio_file, ss=start_time, t=duration)
                    stream = ffmpeg.output(stream, str(temp_file), acodec='pcm_s16le', ar=16000)
                    ffmpeg.run(stream, overwrite_output=True, quiet=True)

                    if temp_file.exists():
                        return str(temp_file)
                    return None
                except ffmpeg.Error:
                    pass  # Fallback a subprocess

            # Fallback a subprocess
            cmd = [
                'ffmpeg', '-y',
                '-i', audio_file,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                str(temp_file)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and temp_file.exists():
                return str(temp_file)
            return None

        except Exception as e:
            print(f"❌ Error extrayendo segmento: {e}")
            return None

    def play_audio_segment(self, segment_file: str) -> bool:
        """Reproduce el segmento de audio"""
        try:
            print(f"🔊 Reproduciendo: {segment_file}")

            # Detectar reproductor disponible
            players = ['afplay', 'ffplay', 'vlc', 'mpv', 'mplayer']

            for player in players:
                if subprocess.run(['which', player], capture_output=True).returncode == 0:
                    print(f"   🎵 Usando {player}")

                    if player == 'ffplay':
                        cmd = [player, '-nodisp', '-autoexit', segment_file]
                    else:
                        cmd = [player, segment_file]

                    # Ejecutar en hilo separado
                    def play():
                        subprocess.run(cmd, capture_output=True)

                    thread = threading.Thread(target=play)
                    thread.start()
                    return True

            print("❌ No se encontró reproductor de audio")
            print("💡 Instala: brew install ffmpeg (macOS) o apt install ffmpeg (Linux)")
            return False

        except Exception as e:
            print(f"❌ Error reproduciendo audio: {e}")
            return False

    def search_and_play(self, query_text: str, k: int = 5):
        """Función principal: busca y reproduce resultados"""
        print("\n🎵 BÚSQUEDA SEMÁNTICA DE AUDIO")
        print("=" * 70)

        # Realizar búsqueda
        results = self.search_semantic(query_text, k)

        if not results:
            return

        # Mostrar resultados
        self.display_results(results, query_text)

        # Preguntar qué resultado reproducir
        while True:
            print("🎵 Opciones de reproducción:")
            for i, result in enumerate(results, 1):
                segment = result['segment']
                text = segment.get('text', '')[:50]
                print(f"   {i}. ID {segment.get('segment_id', i-1)} - {text}...")

            print("   0. Volver al menú de búsqueda")
            print("   r. Mostrar resultados nuevamente")

            try:
                choice = input(f"\n🔊 ¿Qué segmento reproducir? (1-{len(results)}, r, 0): ").strip().lower()

                if choice == '0':
                    print("🔄 Volviendo al menú de búsqueda...")
                    return

                if choice == 'r':
                    print("\n" + "="*70)
                    self.display_results(results, query_text)
                    continue

                choice_num = int(choice)

                if 1 <= choice_num <= len(results):
                    selected_result = results[choice_num - 1]
                    segment = selected_result['segment']

                    segment_id = segment.get('segment_id', choice_num - 1)
                    start_time = segment.get('start_time', 0)
                    end_time = segment.get('end_time', 0)
                    original_file = segment.get('original_file_name', segment.get('source_file', ''))

                    print(f"\n🎯 Reproduciendo segmento {segment_id}...")
                    print(f"   📝 Texto: {segment.get('text', '')}")
                    print(f"   ⏱️  Tiempo: {start_time:.1f}s - {end_time:.1f}s")

                    # Encontrar archivo de audio
                    audio_file = self.find_audio_file(original_file)
                    if not audio_file:
                        continue

                    # Extraer segmento
                    print("✂️  Extrayendo segmento de audio...")
                    segment_file = self.extract_audio_segment(
                        audio_file, start_time, end_time, segment_id
                    )

                    if segment_file:
                        success = self.play_audio_segment(segment_file)

                        if success:
                            print("✅ Reproducción iniciada")
                            input("   Presiona Enter para continuar...")

                            # Limpiar archivo temporal
                            with contextlib.suppress(OSError):
                                os.remove(segment_file)
                        else:
                            print("❌ Error en reproducción")
                    else:
                        print("❌ No se pudo extraer el segmento")

                    print("\n" + "-"*50)
                    continue

                print("❌ Opción inválida")

            except ValueError:
                print("❌ Entrada inválida")
            except KeyboardInterrupt:
                print("\n🔄 Volviendo al menú de búsqueda...")
                return

    def interactive_search(self):
        """Interfaz interactiva de búsqueda"""
        print("\n🔍 BÚSQUEDA SEMÁNTICA INTERACTIVA DE AUDIO")
        print("=" * 70)
        print(f"📂 Dataset: {self.dataset_path}")
        print(f"📊 Segmentos: {len(self.df)}")
        print(f"🔧 Backend: {'FAISS' if self.faiss_index else 'NumPy'}")
        print()
        print("💡 Ejemplos: 'política económica', 'entrevista', 'música de fondo'")
        print()

        while True:
            try:
                query = input("🔍 Ingresa tu búsqueda (o 'salir' para terminar): ").strip()

                if query.lower() in ['salir', 'exit', 'quit', '']:
                    print("👋 ¡Hasta luego!")
                    break

                if len(query) < 2:
                    print("⚠️  Búsqueda muy corta, intenta con al menos 2 caracteres")
                    continue

                self.search_and_play(query)
                print("\n" + "="*70 + "\n")

            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="CLI interactivo para búsqueda semántica de audio (local con FAISS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s ./dataset
  %(prog)s /path/to/my/dataset

Requisitos:
  - Dataset generado con simple_dataset_pipeline.py
  - ffmpeg instalado para reproducción de audio
  - sentence-transformers para embeddings
        """
    )
    parser.add_argument(
        'dataset_path',
        nargs='?',
        default='./dataset',
        help='Ruta al directorio del dataset (default: ./dataset)'
    )

    args = parser.parse_args()

    print("🎵 CLI de Búsqueda Semántica de Audio (Local)")
    print("=" * 50)

    # Verificar dependencias
    try:
        import sentence_transformers
    except ImportError:
        print("❌ sentence-transformers no está instalado")
        print("💡 Instala: pip install sentence-transformers")
        return 1

    if not FAISS_AVAILABLE:
        print("⚠️  FAISS no disponible - la búsqueda será más lenta")
        print("💡 Instala: pip install faiss-cpu")

    # Verificar ffmpeg
    if subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode != 0:
        print("⚠️  ffmpeg no encontrado - funcionalidad de reproducción limitada")
        print("💡 Instala: brew install ffmpeg (macOS) o apt install ffmpeg (Linux)")

    # Verificar que existe el dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset no encontrado: {dataset_path}")
        print("💡 Genera un dataset primero con:")
        print("   poetry run python src/simple_dataset_pipeline.py --input data/ --output ./dataset")
        return 1

    # Crear sistema de búsqueda
    try:
        search_system = LocalAudioSearch(str(dataset_path))
        search_system.interactive_search()
        return 0

    except Exception as e:
        print(f"❌ Error iniciando sistema: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
