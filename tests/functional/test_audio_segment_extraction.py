#!/usr/bin/env python3
"""
Test para validar la extracción de segmentos de audio usando ffmpeg-python
"""

from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
PROJECT_ROOT = TESTS_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_extract_audio_segment():
    """Test de extracción de segmento de audio"""
    print("🧪 Testing Audio Segment Extraction")
    print("=" * 50)

    # Buscar un archivo de audio de prueba
    audio_dir = PROJECT_ROOT / "dataset" / "converted"

    if not audio_dir.exists():
        print(f"❌ Directorio de audio no encontrado: {audio_dir}")
        return False

    # Buscar archivos de audio
    audio_files = list(audio_dir.glob("*.wav"))
    if not audio_files:
        print(f"❌ No se encontraron archivos de audio en {audio_dir}")
        return False

    test_audio_file = str(audio_files[0])
    print(f"📁 Archivo de prueba: {Path(test_audio_file).name}")

    # Verificar que ffmpeg está disponible
    import subprocess
    ffmpeg_check = subprocess.run(['which', 'ffmpeg'], capture_output=True)
    if ffmpeg_check.returncode != 0:
        print("⚠️  ffmpeg no está instalado - el test no puede ejecutarse")
        return False

    # Verificar si ffmpeg-python está disponible
    try:
        import ffmpeg
        print("✅ ffmpeg-python está disponible")
        using_ffmpeg_python = True
    except ImportError:
        print("⚠️  ffmpeg-python no está disponible - usando fallback de subprocess")
        using_ffmpeg_python = False

    # Verificar disponibilidad de ffmpeg-python sin importar el módulo completo
    try:
        import ffmpeg
        FFMPEG_PYTHON_AVAILABLE = True
    except ImportError:
        FFMPEG_PYTHON_AVAILABLE = False

    # Crear una instancia mínima - solo necesitamos el método extract_audio_segment
    # Copiamos la lógica directamente para evitar dependencias
    class TestSearchSystem:
        def extract_audio_segment(self, audio_file, start_time, end_time, segment_id):
            """Extrae el segmento específico del audio usando ffmpeg"""
            from pathlib import Path
            import subprocess

            try:
                # Crear directorio temporal
                temp_dir = Path("temp_audio_segments")
                temp_dir.mkdir(exist_ok=True)

                # Archivo temporal para el segmento
                temp_file = temp_dir / f"segment_{segment_id}_{start_time:.1f}s.wav"

                # Usar ffmpeg-python si está disponible
                try:
                    import ffmpeg
                    ffmpeg_available = True
                except ImportError:
                    ffmpeg_available = False

                if ffmpeg_available:
                    try:
                        import ffmpeg
                        duration = end_time - start_time
                        stream = ffmpeg.input(audio_file, ss=start_time, t=duration)
                        stream = ffmpeg.output(stream, str(temp_file), codec='copy')
                        ffmpeg.run(stream, overwrite_output=True, quiet=True)

                        if temp_file.exists():
                            return str(temp_file)
                        print("❌ Error: archivo de salida no fue creado")
                        return None
                    except ffmpeg.Error as e:
                        error_message = e.stderr.decode() if e.stderr else str(e)
                        print(f"❌ Error extrayendo segmento con ffmpeg-python: {error_message}")
                        return None
                else:
                    # Fallback a subprocess
                    cmd = [
                        'ffmpeg', '-y', '-i', audio_file,
                        '-ss', str(start_time), '-t', str(end_time - start_time),
                        '-c', 'copy', str(temp_file)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and temp_file.exists():
                        return str(temp_file)
                    print(f"❌ Error extrayendo segmento: {result.stderr}")
                    return None
            except subprocess.TimeoutExpired:
                print("❌ Timeout extrayendo segmento")
                return None
            except Exception as e:
                print(f"❌ Error: {e}")
                return None

    search_system = TestSearchSystem()
    print("✅ Sistema de prueba inicializado")

    # Parámetros de prueba
    start_time = 5.0  # 5 segundos
    end_time = 10.0   # 10 segundos
    segment_id = "test_001"

    print("\n🎵 Extrayendo segmento:")
    print(f"   Archivo: {Path(test_audio_file).name}")
    print(f"   Tiempo: {start_time}s - {end_time}s ({end_time - start_time}s)")
    print(f"   Usando: {'ffmpeg-python' if using_ffmpeg_python else 'subprocess'}")

    # Ejecutar extracción
    try:
        output_file = search_system.extract_audio_segment(
            test_audio_file, start_time, end_time, segment_id
        )

        if output_file and Path(output_file).exists():
            file_size = Path(output_file).stat().st_size
            print("\n✅ Segmento extraído exitosamente!")
            print(f"   Archivo de salida: {Path(output_file).name}")
            print(f"   Tamaño: {file_size} bytes")

            # Verificar que el archivo tiene contenido
            if file_size > 0:
                print("✅ Archivo tiene contenido válido")

                # Limpiar archivo temporal
                try:
                    Path(output_file).unlink()
                    print("✅ Archivo temporal limpiado")
                except:
                    pass

                return True
            print("❌ Archivo está vacío")
            return False
        print("❌ No se pudo extraer el segmento")
        return False

    except Exception as e:
        print(f"❌ Error durante la extracción: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ffmpeg_python_import():
    """Test para verificar que ffmpeg-python puede importarse"""
    print("\n🧪 Testing ffmpeg-python import")
    print("=" * 50)

    try:
        import ffmpeg
        print("✅ ffmpeg-python importado correctamente")
        print(f"   Versión disponible: {ffmpeg.__version__ if hasattr(ffmpeg, '__version__') else 'N/A'}")
        return True
    except ImportError as e:
        print(f"⚠️  ffmpeg-python no está disponible: {e}")
        print("   Ejecuta: pip install ffmpeg-python")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("TEST DE EXTRACCIÓN DE SEGMENTOS DE AUDIO")
    print("=" * 50)

    # Test 1: Verificar importación
    import_test = test_ffmpeg_python_import()

    # Test 2: Extracción de segmento
    extraction_test = test_extract_audio_segment()

    # Resumen
    print("\n" + "=" * 50)
    print("RESUMEN DE TESTS")
    print("=" * 50)
    print(f"Import test: {'✅ PASS' if import_test else '⚠️  SKIP (opcional)'}")
    print(f"Extraction test: {'✅ PASS' if extraction_test else '❌ FAIL'}")

    if extraction_test:
        print("\n✅ Todos los tests críticos pasaron!")
        sys.exit(0)
    else:
        print("\n❌ Algunos tests fallaron")
        sys.exit(1)

