"""
Utilidades para conversión de formatos de audio
"""

import logging
from pathlib import Path

try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioConverter:
    """Conversor de formatos de audio usando pydub y librosa"""

    def __init__(self):
        """Inicializa el conversor y verifica dependencias"""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Verificar dependencias
        if not PYDUB_AVAILABLE:
            self.logger.warning("pydub no disponible. Funcionalidad limitada.")

        if not LIBROSA_AVAILABLE:
            self.logger.warning("librosa no disponible. Funcionalidad limitada.")

        # Verificar ffmpeg para pydub
        if PYDUB_AVAILABLE:
            ffmpeg_path = which("ffmpeg")
            if not ffmpeg_path:
                self.logger.warning("ffmpeg no encontrado. Algunos formatos pueden no funcionar.")

        self.supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}

    def convert(self,
                input_path: str | Path,
                output_path: str | Path,
                sample_rate: int = 16000,
                channels: int = 1,
                format: str = "wav") -> bool:
        """
        Convierte archivo de audio a formato específico
        
        Args:
            input_path: Ruta del archivo de entrada
            output_path: Ruta del archivo de salida
            sample_rate: Frecuencia de muestreo objetivo
            channels: Número de canales (1=mono, 2=estéreo)
            format: Formato de salida
            
        Returns:
            True si la conversión fue exitosa
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            self.logger.error(f"Archivo de entrada no existe: {input_path}")
            return False

        # Crear directorio de salida si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Intentar con librosa primero (mejor calidad)
            if LIBROSA_AVAILABLE:
                return self._convert_with_librosa(input_path, output_path, sample_rate, channels)

            # Fallback a pydub
            if PYDUB_AVAILABLE:
                return self._convert_with_pydub(input_path, output_path, sample_rate, channels, format)

            self.logger.error("No hay librerías de conversión disponibles")
            return False

        except Exception as e:
            self.logger.error(f"Error convirtiendo {input_path}: {e!s}")
            return False

    def _convert_with_librosa(self,
                             input_path: Path,
                             output_path: Path,
                             sample_rate: int,
                             channels: int) -> bool:
        """Convierte usando librosa + soundfile"""
        try:
            # Cargar audio
            audio, sr = librosa.load(str(input_path), sr=sample_rate, mono=(channels == 1))

            # Ajustar canales si es necesario
            if channels == 2 and audio.ndim == 1:
                # Convertir mono a estéreo duplicando el canal
                audio = np.stack([audio, audio])
            elif channels == 1 and audio.ndim == 2:
                # Convertir estéreo a mono
                audio = librosa.to_mono(audio)

            # Guardar
            sf.write(str(output_path), audio, sample_rate)

            self.logger.debug(f"Convertido con librosa: {input_path.name} → {output_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error en conversión con librosa: {e!s}")
            return False

    def _convert_with_pydub(self,
                           input_path: Path,
                           output_path: Path,
                           sample_rate: int,
                           channels: int,
                           format: str) -> bool:
        """Convierte usando pydub"""
        try:
            # Cargar audio
            audio = AudioSegment.from_file(str(input_path))

            # Ajustar propiedades
            if audio.frame_rate != sample_rate:
                audio = audio.set_frame_rate(sample_rate)

            if audio.channels != channels:
                if channels == 1:
                    audio = audio.set_channels(1)  # Convertir a mono
                elif channels == 2:
                    audio = audio.set_channels(2)  # Convertir a estéreo

            # Exportar
            audio.export(str(output_path), format=format)

            self.logger.debug(f"Convertido con pydub: {input_path.name} → {output_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error en conversión con pydub: {e!s}")
            return False

    def get_audio_info(self, file_path: str | Path) -> dict | None:
        """
        Obtiene información del archivo de audio
        
        Args:
            file_path: Ruta del archivo de audio
            
        Returns:
            Diccionario con información del audio o None si hay error
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return None

        try:
            if LIBROSA_AVAILABLE:
                # Usar librosa para obtener información
                audio, sr = librosa.load(str(file_path), sr=None)
                duration = len(audio) / sr

                return {
                    "duration": duration,
                    "sample_rate": sr,
                    "channels": 1 if audio.ndim == 1 else audio.shape[0],
                    "samples": len(audio) if audio.ndim == 1 else audio.shape[1],
                    "format": file_path.suffix.lower(),
                    "file_size": file_path.stat().st_size
                }

            if PYDUB_AVAILABLE:
                # Usar pydub para obtener información
                audio = AudioSegment.from_file(str(file_path))

                return {
                    "duration": len(audio) / 1000.0,  # pydub usa millisegundos
                    "sample_rate": audio.frame_rate,
                    "channels": audio.channels,
                    "samples": audio.frame_count(),
                    "format": file_path.suffix.lower(),
                    "file_size": file_path.stat().st_size
                }

            # Solo información básica del archivo
            return {
                "format": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size
            }

        except Exception as e:
            self.logger.error(f"Error obteniendo info de {file_path}: {e!s}")
            return None

    def validate_audio_file(self, file_path: str | Path) -> bool:
        """
        Valida si un archivo es un audio válido
        
        Args:
            file_path: Ruta del archivo a validar
            
        Returns:
            True si el archivo es válido
        """
        file_path = Path(file_path)

        # Verificar extensión
        if file_path.suffix.lower() not in self.supported_formats:
            return False

        # Verificar que el archivo existe y no está vacío
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False

        # Intentar cargar el archivo para verificar que es válido
        info = self.get_audio_info(file_path)
        return info is not None and info.get("duration", 0) > 0

    def batch_convert(self,
                     input_dir: str | Path,
                     output_dir: str | Path,
                     sample_rate: int = 16000,
                     channels: int = 1,
                     format: str = "wav") -> dict:
        """
        Convierte múltiples archivos en lote
        
        Args:
            input_dir: Directorio con archivos de entrada
            output_dir: Directorio de salida
            sample_rate: Frecuencia de muestreo objetivo
            channels: Número de canales
            format: Formato de salida
            
        Returns:
            Diccionario con estadísticas de conversión
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Directorio de entrada no existe: {input_dir}")

        # Crear directorio de salida
        output_dir.mkdir(parents=True, exist_ok=True)

        # Encontrar archivos de audio
        audio_files = []
        for ext in self.supported_formats:
            audio_files.extend(input_dir.rglob(f"*{ext}"))
            audio_files.extend(input_dir.rglob(f"*{ext.upper()}"))

        stats = {
            "total_files": len(audio_files),
            "converted": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }

        for audio_file in audio_files:
            # Generar ruta de salida manteniendo estructura de directorios
            relative_path = audio_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix(f'.{format}')

            # Verificar si ya existe
            if output_file.exists():
                self.logger.info(f"Saltando (ya existe): {audio_file.name}")
                stats["skipped"] += 1
                continue

            # Convertir
            if self.convert(audio_file, output_file, sample_rate, channels, format):
                stats["converted"] += 1
                self.logger.info(f"Convertido: {audio_file.name}")
            else:
                stats["failed"] += 1
                stats["errors"].append(str(audio_file))
                self.logger.error(f"Falló conversión: {audio_file.name}")

        self.logger.info(f"Conversión en lote completada: {stats['converted']}/{stats['total_files']} exitosos")
        return stats


# Importar numpy si librosa está disponible
if LIBROSA_AVAILABLE:
    import numpy as np
