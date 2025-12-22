#!/usr/bin/env python3
"""
Sistema de búsqueda semántica con reproducción de audio
Busca términos en Supabase y reproduce el segmento de audio correspondiente
"""

import os
import sys
import ast
import time
import numpy as np
import subprocess
import threading
from pathlib import Path
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

try:
    import ffmpeg
    FFMPEG_PYTHON_AVAILABLE = True
except ImportError:
    FFMPEG_PYTHON_AVAILABLE = False

# Evitar warning de tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SemanticAudioSearch:
    """Sistema de búsqueda semántica con reproducción de audio"""

    def __init__(self):
        self.supabase = None
        self.text_model = None
        self.connect_to_supabase()
        self.load_text_model()

    def connect_to_supabase(self):
        """Conecta a Supabase"""
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')

        if not all([supabase_url, supabase_key]):
            print("❌ Variables de entorno no encontradas")
            print("Ejecuta: source .supabase")
            sys.exit(1)

        try:
            self.supabase: Client = create_client(supabase_url, supabase_key)
            print(f"✅ Conectado a Supabase: {supabase_url}")
        except Exception as e:
            print(f"❌ Error conectando: {e}")
            sys.exit(1)

    def load_text_model(self):
        """Carga el modelo de embeddings de texto"""
        try:
            print("🤖 Cargando modelo de embeddings de texto...")
            self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("✅ Modelo de texto cargado")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            print("💡 Instala: pip install sentence-transformers")
            sys.exit(1)

    def generate_text_embedding(self, text):
        """Genera embedding vectorial para el texto de búsqueda"""
        try:
            embedding = self.text_model.encode(text)
            # Normalizar para similitud coseno
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            print(f"❌ Error generando embedding: {e}")
            return None

    def parse_embedding(self, embedding_data):
        """Convierte embedding de Supabase a numpy array"""
        try:
            if isinstance(embedding_data, str):
                # Si es string, convertir a lista
                embedding_list = ast.literal_eval(embedding_data)
            elif isinstance(embedding_data, list):
                # Si ya es lista, usar directamente
                embedding_list = embedding_data
            else:
                return None

            # Convertir a numpy array
            embedding_array = np.array(embedding_list, dtype=np.float32)
            return embedding_array

        except Exception as e:
            print(f"⚠️  Error parseando embedding: {e}")
            return None

    def search_semantic(self, query_text, k=5):
        """Realiza búsqueda semántica vectorial"""
        print(f"🔍 Búsqueda semántica: '{query_text}'")
        print("-" * 50)

        # Generar embedding de la consulta
        query_embedding = self.generate_text_embedding(query_text)
        if query_embedding is None:
            return []

        try:
            # Búsqueda vectorial usando RPC (función personalizada)
            # Como Supabase Python no soporta directamente operador <=>, usamos alternativa

            # Primero obtener todos los segmentos con embeddings
            all_segments = self.supabase.table('audio_segments').select(
                'segment_id, text, language, duration, start_time, end_time, ' +
                'source_file, original_file_name, text_embedding, confidence'
            ).execute()

            if not all_segments.data:
                print("❌ No hay segmentos en la base de datos")
                return []

            # Calcular similitud coseno manualmente
            results = []
            valid_results = 0

            for segment in all_segments.data:
                if segment.get('text_embedding'):
                    # Parsear embedding del segmento
                    segment_embedding = self.parse_embedding(segment['text_embedding'])

                    if segment_embedding is not None and len(segment_embedding) > 0:
                        try:
                            # Normalizar embeddings
                            segment_norm = segment_embedding / np.linalg.norm(segment_embedding)
                            query_norm = query_embedding / np.linalg.norm(query_embedding)

                            # Calcular similitud coseno
                            similarity = np.dot(query_norm, segment_norm)

                            result = {
                                'segment': segment,
                                'similarity': float(similarity),
                                'distance': 1.0 - float(similarity)  # Para compatibilidad
                            }
                            results.append(result)
                            valid_results += 1

                        except Exception as e:
                            print(f"⚠️  Error calculando similitud para segmento {segment.get('segment_id', 'unknown')}: {e}")
                            continue

            print(f"✅ Procesados {valid_results} segmentos válidos de {len(all_segments.data)} totales")

            # Ordenar por similitud (mayor es mejor)
            results.sort(key=lambda x: x['similarity'], reverse=True)

            # Retornar top-k resultados
            return results[:k]

        except Exception as e:
            print(f"❌ Error en búsqueda vectorial: {e}")
            return []

    def display_results(self, results, query_text):
        """Muestra los resultados de búsqueda"""
        if not results:
            print(f"❌ No se encontraron resultados para '{query_text}'")
            return

        print(f"✅ Encontrados {len(results)} resultados más relevantes:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            segment = result['segment']
            similarity = result['similarity']

            # Información del segmento
            print(f"🎯 RESULTADO {i}")
            print(f"   📋 ID: {segment['segment_id']}")
            print(f"   🌐 Idioma: {segment['language']}")
            print(f"   📊 Similitud: {similarity:.4f} ({similarity*100:.1f}%)")
            print(f"   ⏱️  Tiempo: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s ({segment['duration']:.1f}s)")
            print(f"   📁 Archivo: {segment['original_file_name']}")

            if segment.get('confidence'):
                print(f"   🎯 Confianza transcripción: {segment['confidence']:.3f}")

            # Resaltar términos de búsqueda en el texto
            text = segment['text']
            query_words = query_text.lower().split()

            # Resaltar palabras encontradas
            highlighted_text = text
            for word in query_words:
                if word in text.lower():
                    # Encontrar la palabra en el texto original (manteniendo mayúsculas)
                    import re
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    highlighted_text = pattern.sub(f"**{word.upper()}**", highlighted_text)

            print(f"   📝 Texto: {highlighted_text}")
            print()

    def find_audio_file(self, original_file_name):
        """Encuentra el archivo de audio correspondiente"""
        # Buscar en dataset/converted/
        audio_path = Path("dataset/converted") / original_file_name

        if audio_path.exists():
            return str(audio_path)

        print(f"⚠️  Archivo de audio no encontrado: {original_file_name}")
        return None

    def extract_audio_segment(self, audio_file, start_time, end_time, segment_id):
        """Extrae el segmento específico del audio usando ffmpeg"""
        try:
            # Crear directorio temporal
            temp_dir = Path("temp_audio_segments")
            temp_dir.mkdir(exist_ok=True)

            # Archivo temporal para el segmento
            temp_file = temp_dir / f"segment_{segment_id}_{start_time:.1f}s.wav"

            # Usar ffmpeg-python si está disponible (más limpio y mantenible)
            if FFMPEG_PYTHON_AVAILABLE:
                try:
                    duration = end_time - start_time
                    stream = ffmpeg.input(audio_file, ss=start_time, t=duration)
                    stream = ffmpeg.output(stream, str(temp_file), codec='copy')
                    # Ejecutar de forma silenciosa (captura stderr automáticamente)
                    ffmpeg.run(stream, overwrite_output=True, quiet=True)
                    
                    if temp_file.exists():
                        return str(temp_file)
                    else:
                        print("❌ Error: archivo de salida no fue creado")
                        return None
                except ffmpeg.Error as e:
                    error_message = e.stderr.decode() if e.stderr else str(e)
                    print(f"❌ Error extrayendo segmento con ffmpeg-python: {error_message}")
                    return None
            else:
                # Fallback a subprocess si ffmpeg-python no está disponible
                cmd = [
                    'ffmpeg', '-y',  # -y para sobrescribir
                    '-i', audio_file,
                    '-ss', str(start_time),
                    '-t', str(end_time - start_time),
                    '-c', 'copy',  # Copiar sin recodificar
                    str(temp_file)
                ]

                # Ejecutar ffmpeg silenciosamente
                result = subprocess.run(cmd,
                                      capture_output=True,
                                      text=True,
                                      timeout=30)

                if result.returncode == 0 and temp_file.exists():
                    return str(temp_file)
                else:
                    print(f"❌ Error extrayendo segmento: {result.stderr}")
                    return None

        except subprocess.TimeoutExpired:
            print("❌ Timeout extrayendo segmento")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def play_audio_segment(self, segment_file):
        """Reproduce el segmento de audio"""
        try:
            print(f"🔊 Reproduciendo: {segment_file}")

            # Detectar reproductor disponible
            players = ['afplay', 'ffplay', 'vlc', 'mpv', 'mplayer']

            for player in players:
                if subprocess.run(['which', player], capture_output=True).returncode == 0:
                    print(f"   🎵 Usando {player}")

                    if player == 'ffplay':
                        # ffplay con opciones para ocultar ventana
                        cmd = [player, '-nodisp', '-autoexit', segment_file]
                    else:
                        cmd = [player, segment_file]

                    # Ejecutar en hilo separado para no bloquear
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

    def search_and_play(self, query_text, k=3):
        """Función principal: busca y reproduce resultados"""
        print(f"🎵 BÚSQUEDA SEMÁNTICA DE AUDIO PERIODÍSTICO")
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
                print(f"   {i}. ID {segment['segment_id']} - {segment['text'][:50]}...")

            print(f"   0. Volver al menú de búsqueda")
            print(f"   r. Mostrar resultados nuevamente")

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

                    print(f"\n🎯 Reproduciendo segmento {segment['segment_id']}...")
                    print(f"   📝 Texto: {segment['text']}")
                    print(f"   ⏱️  Tiempo: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s")

                    # Encontrar archivo de audio
                    audio_file = self.find_audio_file(segment['original_file_name'])
                    if not audio_file:
                        continue

                    # Extraer segmento
                    print("✂️  Extrayendo segmento de audio...")
                    segment_file = self.extract_audio_segment(
                        audio_file,
                        segment['start_time'],
                        segment['end_time'],
                        segment['segment_id']
                    )

                    if segment_file:
                        # Reproducir
                        success = self.play_audio_segment(segment_file)

                        if success:
                            print("✅ Reproducción completada")
                            input("   Presiona Enter para continuar...")

                            # Limpiar archivo temporal
                            try:
                                os.remove(segment_file)
                            except:
                                pass
                        else:
                            print("❌ Error en reproducción")
                    else:
                        print("❌ No se pudo extraer el segmento")

                    # Preguntar si quiere reproducir otro segmento
                    print("\n" + "-"*50)
                    continue

                else:
                    print("❌ Opción inválida")
                    continue

            except ValueError:
                print("❌ Entrada inválida")
                continue
            except KeyboardInterrupt:
                print("\n🔄 Volviendo al menú de búsqueda...")
                return

    def interactive_search(self):
        """Interfaz interactiva de búsqueda"""
        print("🔍 BÚSQUEDA SEMÁNTICA INTERACTIVA DE AUDIO PERIODÍSTICO")
        print("=" * 70)
        print("💡 Busca contenido en debates políticos y programas de TV argentinos")
        print("📝 Ejemplos: 'política económica', 'debate', 'discusión', 'fantino'")
        print()

        while True:
            try:
                query = input("🔍 Ingresa tu búsqueda (o 'salir' para terminar): ").strip()

                if query.lower() in ['salir', 'exit', 'quit', '']:
                    print("👋 ¡Hasta luego!")
                    break

                if len(query) < 3:
                    print("⚠️  Búsqueda muy corta, intenta con al menos 3 caracteres")
                    continue

                print()
                self.search_and_play(query)
                print("\n" + "="*70 + "\n")

            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Función principal"""
    print("Loading semantic audio search system...")

    # Verificar dependencias
    try:
        import sentence_transformers
    except ImportError:
        print("❌ sentence-transformers no está instalado")
        print("💡 Instala: pip install sentence-transformers")
        return False

    # Verificar variables de entorno
    if not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_KEY'):
        print("❌ Variables de entorno no encontradas")
        print("Ejecuta: source .supabase")
        return False

    # Verificar ffmpeg para extracción de audio
    if not FFMPEG_PYTHON_AVAILABLE:
        print("⚠️  ffmpeg-python no está instalado - usando subprocess como fallback")
        print("💡 Instala: pip install ffmpeg-python (recomendado)")
    
    if subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode != 0:
        print("⚠️  ffmpeg no encontrado en el sistema - funcionalidad de extracción limitada")
        print("💡 Instala: brew install ffmpeg (macOS) o apt install ffmpeg (Linux)")

    # Crear sistema de búsqueda
    try:
        search_system = SemanticAudioSearch()

        # Ejecutar interfaz interactiva
        search_system.interactive_search()

        return True

    except Exception as e:
        print(f"❌ Error iniciando sistema: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)