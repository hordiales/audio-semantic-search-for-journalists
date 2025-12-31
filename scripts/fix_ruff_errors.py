#!/usr/bin/env python3
"""
Script para aplicar correcciones automáticas de Ruff de forma segura.

Este script aplica correcciones automáticas de ruff de forma incremental,
permitiendo revisar los cambios antes de aplicarlos todos.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Intentar cargar python-dotenv para leer archivos .env
try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def load_env_file() -> None:
    """Carga el archivo .env si está disponible."""
    if not DOTENV_AVAILABLE:
        return

    # Buscar archivo .env en el directorio del proyecto
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        # También buscar en el directorio actual
        current_env = Path.cwd() / ".env"
        if current_env.exists():
            load_dotenv(current_env)


def setup_logging(log_level: str | None = None) -> logging.Logger:
    """
    Configura el sistema de logging.

    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   Si es None, se usa la variable de entorno LOG_LEVEL del .env
                   o DEBUG por defecto.

    Returns:
        Logger configurado.
    """
    # Cargar archivo .env primero
    load_env_file()

    # Obtener nivel de logging desde argumento, variable de entorno o por defecto
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

    # Validar nivel
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Configurar formato
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configurar logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configurado con nivel: {log_level}")

    return logger


def run_ruff_check(logger: logging.Logger) -> tuple[int, str]:
    """
    Ejecuta ruff check y retorna el código de salida y la salida.

    Args:
        logger: Logger para mensajes de debug.

    Returns:
        Tupla con (código_de_salida, salida_stdout).
    """
    logger.debug("Ejecutando: ruff check --output-format=concise")
    result = subprocess.run(
        ["ruff", "check", "--output-format=concise"],
        capture_output=True,
        text=True,
    )
    logger.debug(f"Ruff check terminó con código: {result.returncode}")
    return result.returncode, result.stdout


def run_ruff_fix(logger: logging.Logger, dry_run: bool = False) -> tuple[int, str]:
    """
    Ejecuta ruff fix y retorna el código de salida y la salida.

    Args:
        logger: Logger para mensajes de debug.
        dry_run: Si es True, solo muestra diferencias sin aplicar cambios.

    Returns:
        Tupla con (código_de_salida, salida_stdout).
    """
    cmd = ["ruff", "check", "--fix"]
    if dry_run:
        cmd.append("--diff")
        logger.debug("Ejecutando en modo dry-run: ruff check --fix --diff")
    else:
        logger.debug("Ejecutando: ruff check --fix")

    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.debug(f"Ruff fix terminó con código: {result.returncode}")
    return result.returncode, result.stdout


def run_ruff_format(logger: logging.Logger, dry_run: bool = False) -> tuple[int, str]:
    """
    Ejecuta ruff format y retorna el código de salida y la salida.

    Args:
        logger: Logger para mensajes de debug.
        dry_run: Si es True, solo muestra diferencias sin aplicar cambios.

    Returns:
        Tupla con (código_de_salida, salida_stdout).
    """
    cmd = ["ruff", "format"]
    if dry_run:
        cmd.append("--check")
        cmd.append("--diff")
        logger.debug("Ejecutando en modo dry-run: ruff format --check --diff")
    else:
        logger.debug("Ejecutando: ruff format")

    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.debug(f"Ruff format terminó con código: {result.returncode}")
    return result.returncode, result.stdout


def main():
    """Función principal."""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Script para aplicar correcciones automáticas de Ruff"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL). "
        "Por defecto lee LOG_LEVEL del archivo .env (valor por defecto: DEBUG). "
        "También se puede configurar con la variable de entorno LOG_LEVEL.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Ejecutar en modo no interactivo (aplica solo formateo)",
    )

    args = parser.parse_args()

    # Configurar logging
    logger = setup_logging(args.log_level)

    logger.info("🔍 Verificando errores de Ruff...")
    logger.debug("Iniciando verificación de errores de Ruff")
    exit_code, output = run_ruff_check(logger)

    if exit_code == 0:
        logger.info("✅ No se encontraron errores de Ruff!")
        return 0

    logger.info(f"\n📊 Se encontraron errores. Mostrando primeros 20:\n")
    lines = output.split("\n")[:20]
    for line in lines:
        logger.info(line)
    total_lines = len(output.split("\n"))
    logger.debug(f"Total de líneas de salida: {total_lines}")

    if total_lines > 20:
        logger.info(f"\n... y {total_lines - 20} más")

    # Modo no interactivo: solo formatear
    if args.non_interactive:
        logger.info("🎨 Modo no interactivo: formateando código...")
        exit_code, output = run_ruff_format(logger, dry_run=False)
        logger.info(output)
        if exit_code == 0:
            logger.info("✅ Código formateado exitosamente")
        return exit_code

    logger.info("\n" + "=" * 70)
    logger.info("Opciones:")
    logger.info("1. Ver diferencias de formateo (dry-run)")
    logger.info("2. Aplicar correcciones automáticas seguras")
    logger.info("3. Aplicar correcciones automáticas (incluyendo unsafe)")
    logger.info("4. Solo formatear código (sin linting)")
    logger.info("5. Salir sin cambios")

    choice = input("\nSelecciona una opción (1-5): ").strip()
    logger.debug(f"Opción seleccionada: {choice}")

    if choice == "1":
        logger.info("\n📝 Mostrando diferencias de formateo...")
        exit_code, output = run_ruff_format(logger, dry_run=True)
        logger.info(output)
        if exit_code == 0:
            logger.info("✅ El código ya está formateado correctamente")
        return exit_code

    elif choice == "2":
        logger.info("\n🔧 Aplicando correcciones automáticas seguras...")
        exit_code, output = run_ruff_fix(logger, dry_run=False)
        logger.info(output)
        if exit_code == 0:
            logger.info("✅ Correcciones aplicadas exitosamente")
        return exit_code

    elif choice == "3":
        logger.info("\n⚠️  Aplicando correcciones automáticas (incluyendo unsafe)...")
        logger.debug("Ejecutando: ruff check --fix --unsafe-fixes")
        result = subprocess.run(
            ["ruff", "check", "--fix", "--unsafe-fixes"],
            capture_output=True,
            text=True,
        )
        logger.debug(f"Ruff check --fix --unsafe-fixes terminó con código: {result.returncode}")
        logger.info(result.stdout)
        if result.returncode == 0:
            logger.info("✅ Correcciones aplicadas exitosamente")
        return result.returncode

    elif choice == "4":
        logger.info("\n🎨 Formateando código...")
        exit_code, output = run_ruff_format(logger, dry_run=False)
        logger.info(output)
        if exit_code == 0:
            logger.info("✅ Código formateado exitosamente")
        return exit_code

    elif choice == "5":
        logger.info("👋 Saliendo sin cambios")
        return 0

    else:
        logger.error("❌ Opción inválida")
        return 1


if __name__ == "__main__":
    sys.exit(main())
