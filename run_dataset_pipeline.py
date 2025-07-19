#!/usr/bin/env python3
"""
Script para ejecutar el pipeline de generación de dataset
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from dataset_orchestrator import DatasetOrchestrator, DatasetConfig
from config_loader import get_config


def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Pipeline de generación de dataset desde archivos de audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

# Procesamiento básico
python run_dataset_pipeline.py --input /data --output ./my_dataset

# Con configuración personalizada
python run_dataset_pipeline.py --input /data --output ./dataset \\
    --whisper-model medium --workers 8 --batch-size 8

# Solo transcripción (sin embeddings)
python run_dataset_pipeline.py --input /data --output ./dataset \\
    --skip-embeddings

# Procesamiento rápido para desarrollo
python run_dataset_pipeline.py --input /data --output ./dataset \\
    --whisper-model tiny --mock-audio --workers 2
        """
    )
    
    # Directorios principales
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Directorio con archivos de audio de entrada"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="./dataset",
        help="Directorio de salida para el dataset (default: ./dataset)"
    )
    
    parser.add_argument(
        "--temp-dir",
        default="./temp_processing",
        help="Directorio temporal para procesamiento (default: ./temp_processing)"
    )
    
    # Configuración de audio
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Frecuencia de muestreo objetivo en Hz (default: 16000)"
    )
    
    parser.add_argument(
        "--channels",
        type=int,
        choices=[1, 2],
        default=1,
        help="Número de canales (1=mono, 2=estéreo) (default: 1)"
    )
    
    # Configuración de transcripción
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Modelo Whisper a usar (default: base)"
    )
    
    parser.add_argument(
        "--language",
        default="es",
        help="Idioma para transcripción (default: es)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Tamaño de lote para procesamiento (default: 4)"
    )
    
    # Configuración de segmentación
    parser.add_argument(
        "--segmentation",
        choices=["silence", "time"],
        default="silence",
        help="Método de segmentación (default: silence)"
    )
    
    parser.add_argument(
        "--min-silence",
        type=int,
        default=500,
        help="Duración mínima de silencio en ms (default: 500)"
    )
    
    parser.add_argument(
        "--silence-thresh",
        type=int,
        default=-40,
        help="Umbral de silencio en dB (default: -40)"
    )
    
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=10.0,
        help="Duración de segmento para método 'time' en segundos (default: 10.0)"
    )
    
    # Configuración de embeddings
    parser.add_argument(
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Modelo para embeddings de texto"
    )
    
    parser.add_argument(
        "--mock-audio",
        action="store_true",
        help="Usar embeddings de audio mock (más rápido)"
    )
    
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Omitir generación de embeddings (solo transcripción)"
    )
    
    # Configuración de paralelización
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Número de workers paralelos (default: 4)"
    )
    
    parser.add_argument(
        "--no-multiprocessing",
        action="store_true",
        default=True,  # Desactivado por defecto
        help="Deshabilitar multiprocesamiento"
    )
    
    # Opciones de salida
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="No guardar archivos intermedios"
    )
    
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Comprimir salida final"
    )
    
    # Opciones de ejecución
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Intentar reanudar procesamiento previo"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar qué se haría sin ejecutar"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar salida detallada"
    )
    
    # Configuración desde archivo
    parser.add_argument(
        "--config-file",
        help="Cargar configuración desde archivo JSON"
    )
    
    return parser.parse_args()


def load_config_from_file(config_file: str) -> dict:
    """Carga configuración desde archivo JSON"""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_file}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_dataset_config(args) -> DatasetConfig:
    """Crea configuración del dataset desde argumentos"""
    
    # Cargar configuración base del sistema
    system_config = get_config()
    
    # Cargar desde archivo si se especifica
    file_config = {}
    if args.config_file:
        file_config = load_config_from_file(args.config_file)
    
    # Crear configuración combinada (prioridad: args > archivo > sistema)
    config = DatasetConfig(
        # Directorios
        input_dir=args.input,
        output_dir=args.output,
        temp_dir=args.temp_dir,
        
        # Audio
        target_format="wav",
        sample_rate=file_config.get("sample_rate", args.sample_rate),
        channels=file_config.get("channels", args.channels),
        
        # Transcripción
        whisper_model=file_config.get("whisper_model", args.whisper_model),
        language=file_config.get("language", args.language),
        batch_size=file_config.get("batch_size", args.batch_size),
        
        # Segmentación
        segmentation_method=file_config.get("segmentation_method", args.segmentation),
        min_silence_len=file_config.get("min_silence_len", args.min_silence),
        silence_thresh=file_config.get("silence_thresh", args.silence_thresh),
        segment_duration=file_config.get("segment_duration", args.segment_duration),
        
        # Embeddings
        text_model=file_config.get("text_model", args.text_model),
        use_mock_audio=file_config.get("use_mock_audio", args.mock_audio),
        
        # Paralelización
        max_workers=file_config.get("max_workers", args.workers),
        use_multiprocessing=not args.no_multiprocessing,
        
        # Output
        save_intermediate=not args.no_intermediate,
        compress_output=args.compress
    )
    
    return config


def print_config_summary(config: DatasetConfig):
    """Imprime resumen de la configuración"""
    print("🔧 Configuración del Pipeline")
    print("=" * 50)
    print(f"📁 Input:  {config.input_dir}")
    print(f"📁 Output: {config.output_dir}")
    print(f"🎤 Whisper: {config.whisper_model}")
    print(f"📝 Text Model: {config.text_model}")
    print(f"🔊 Audio Mock: {config.use_mock_audio}")
    print(f"📊 Segmentación: {config.segmentation_method}")
    print(f"👥 Workers: {config.max_workers}")
    print(f"💾 Intermediate: {config.save_intermediate}")
    print()


def dry_run_analysis(config: DatasetConfig):
    """Analiza qué se haría en un dry run"""
    print("🔍 Análisis de Dry Run")
    print("=" * 30)
    
    input_path = Path(config.input_dir)
    if not input_path.exists():
        print(f"❌ Directorio de entrada no existe: {input_path}")
        return False
    
    # Contar archivos de audio
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_path.rglob(f"*{ext}"))
        audio_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    print(f"📊 Archivos encontrados: {len(audio_files)}")
    
    # Estimar tiempo de procesamiento
    if audio_files:
        print("📁 Ejemplos de archivos:")
        for i, file in enumerate(audio_files[:5]):
            print(f"  {i+1}. {file.name}")
        if len(audio_files) > 5:
            print(f"  ... y {len(audio_files) - 5} más")
    
    # Estimar espacio en disco
    total_size = sum(f.stat().st_size for f in audio_files)
    print(f"💾 Tamaño total: {total_size / (1024**2):.1f} MB")
    
    # Directorios que se crearían
    print("\n📁 Directorios que se crearán:")
    dirs = [
        config.output_dir,
        f"{config.output_dir}/converted",
        f"{config.output_dir}/transcriptions",
        f"{config.output_dir}/embeddings",
        f"{config.output_dir}/indices",
        f"{config.output_dir}/final"
    ]
    
    for dir_path in dirs:
        exists = "✅" if Path(dir_path).exists() else "📁"
        print(f"  {exists} {dir_path}")
    
    return len(audio_files) > 0


def main():
    """Función principal"""
    args = parse_arguments()
    
    print("🎵 Pipeline de Generación de Dataset de Audio")
    print("=" * 60)
    print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Crear configuración
        config = create_dataset_config(args)
        
        # Mostrar configuración
        if args.verbose:
            print_config_summary(config)
        
        # Dry run si se solicita
        if args.dry_run:
            success = dry_run_analysis(config)
            if success:
                print("\n✅ Dry run completado. Todo listo para procesamiento.")
                return 0
            else:
                print("\n❌ Dry run falló. Revisa la configuración.")
                return 1
        
        # Verificar directorio de entrada
        if not Path(config.input_dir).exists():
            print(f"❌ Error: Directorio de entrada no existe: {config.input_dir}")
            return 1
        
        # Crear orquestador
        orchestrator = DatasetOrchestrator(config)
        
        # Ejecutar pipeline
        if args.skip_embeddings:
            print("⚠️  Modo solo transcripción - embeddings omitidos")
            # TODO: Implementar modo solo transcripción
            print("❌ Modo solo transcripción no implementado aún")
            return 1
        else:
            result = orchestrator.run_full_pipeline()
        
        # Mostrar resultados
        if result["success"]:
            print(f"\n✅ ¡Pipeline completado exitosamente!")
            print(f"📊 Estadísticas:")
            stats = result["stats"]
            print(f"  • Archivos procesados: {stats['converted_files']}/{stats['total_files']}")
            print(f"  • Segmentos generados: {stats['total_segments']}")
            print(f"  • Tiempo total: {stats['processing_time']:.2f} segundos")
            print(f"  • Archivos fallidos: {stats['failed_files']}")
            
            print(f"\n📁 Dataset creado en: {result['dataset']['dataset_dir']}")
            print(f"📋 Manifiesto: {result['dataset']['manifest']}")
            
            if args.verbose:
                print(f"\n📄 Archivos generados:")
                for key, path in result['dataset'].items():
                    if key != 'dataset_dir':
                        print(f"  • {key}: {path}")
            
            return 0
        else:
            print(f"\n❌ Pipeline falló: {result['error']}")
            print(f"⏱️  Tiempo transcurrido: {result['stats']['processing_time']:.2f} segundos")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Procesamiento interrumpido por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Limpieza
        if 'orchestrator' in locals():
            orchestrator.cleanup()


if __name__ == "__main__":
    exit(main())