# Audio Playback Feature

## Funcionalidad de Reproducción de Audio

Se ha agregado una nueva herramienta MCP que permite reproducir segmentos de audio directamente desde Claude Desktop.

### Herramienta: `play_audio_segment`

Reproduce un segmento específico de audio del dataset usando el reproductor de audio apropiado para el sistema operativo.

#### Parámetros:

- **`source_file`** (requerido): Nombre del archivo de audio (sin ruta)
- **`start_time`** (requerido): Tiempo de inicio en segundos
- **`end_time`** (requerido): Tiempo de fin en segundos  
- **`segment_index`** (opcional): Índice del segmento de los resultados de búsqueda

#### Ejemplo de uso:

```json
{
  "source_file": "audio_file.wav",
  "start_time": 10.5,
  "end_time": 15.2,
  "segment_index": 0
}
```

### Reproductores Soportados por SO:

#### macOS (Darwin)
- **ffplay** (preferido) - Parte de ffmpeg, soporta tiempo específico
- **afplay** (fallback) - Nativo de macOS, reproduce archivo completo

#### Linux
- **ffplay** (preferido) - Mejor soporte para segmentos temporales
- **cvlc** - VLC en línea de comandos
- **mplayer** - Reproductor multimedia
- **aplay/paplay** - Para audio PCM

#### Windows
- **ffplay** (preferido) - Si ffmpeg está instalado
- **wmplayer** - Windows Media Player

### Funcionalidades:

1. **Detección automática de SO**: Identifica el sistema operativo y selecciona el reproductor apropiado
2. **Búsqueda inteligente de archivos**: Busca en múltiples directorios del dataset
3. **Soporte de extensiones múltiples**: .wav, .mp3, .opus, .m4a, .flac
4. **Reproducción de segmentos precisos**: Con ffplay, reproduce exactamente el tiempo especificado
5. **Ejecución asíncrona**: No bloquea la interfaz de Claude Desktop

### Ubicaciones de búsqueda de archivos:

1. `dataset/converted/`
2. `dataset/audio/`
3. `data/`
4. `temp_audio/`

### Instalación de Reproductores:

#### macOS:
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt install ffmpeg
```

#### Windows:
Descargar ffmpeg desde: https://ffmpeg.org/download.html

### Comandos de ejemplo generados:

#### ffplay (recomendado):
```bash
ffplay -ss 10.5 -t 4.7 -autoexit -nodisp audio_file.wav
```

#### afplay (macOS fallback):
```bash
afplay audio_file.wav
```

#### VLC (Linux):
```bash
cvlc --play-and-exit --start-time 10.5 --stop-time 15.2 audio_file.wav
```

### Casos de uso:

1. **Verificación de resultados de búsqueda**: Escuchar segmentos encontrados
2. **Análisis cualitativo**: Evaluar la relevancia del contenido audio
3. **Investigación periodística**: Revisar fragmentos específicos de entrevistas
4. **Control de calidad**: Verificar la precisión de transcripciones
5. **Contextualización**: Entender el tono y contexto del audio

### Integración con otras herramientas:

La herramienta se integra perfectamente con las búsquedas existentes:

```
1. semantic_search "política" -> obtener resultados
2. play_audio_segment con los parámetros del primer resultado
```

### Limitaciones:

- **afplay**: No soporta reproducción de segmentos específicos, reproduce el archivo completo
- **Dependencias**: Requiere un reproductor de audio instalado en el sistema
- **Formatos**: Limitado a formatos soportados por el reproductor seleccionado

### Testing:

Usar el script de prueba para verificar compatibilidad:

```bash
python test_audio_player.py
```

Este script verifica:
- Detección del sistema operativo
- Disponibilidad de reproductores
- Capacidad de ejecución

¡La funcionalidad está lista para usar desde Claude Desktop!