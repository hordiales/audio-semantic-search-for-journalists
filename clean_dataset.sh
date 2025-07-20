#!/bin/bash

# Script para limpiar el dataset construido
# Elimina todos los archivos generados manteniendo solo los archivos fuente

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar ayuda
show_help() {
    echo "🧹 Script de limpieza del dataset"
    echo ""
    echo "Uso: $0 [OPCIONES] [DIRECTORIO_DATASET]"
    echo ""
    echo "Opciones:"
    echo "  -h, --help          Mostrar esta ayuda"
    echo "  -f, --force         Eliminar sin confirmación"
    echo "  -v, --verbose       Mostrar detalles de lo que se elimina"
    echo "  --backup            Crear backup antes de eliminar"
    echo "  --dry-run           Mostrar qué se eliminaría sin hacerlo"
    echo ""
    echo "Ejemplos:"
    echo "  $0 ./dataset                    # Limpiar dataset con confirmación"
    echo "  $0 -f ./dataset                # Limpiar sin confirmación"
    echo "  $0 --backup ./dataset          # Crear backup antes de limpiar"
    echo "  $0 --dry-run ./dataset         # Solo mostrar qué se eliminaría"
    echo ""
    echo "El script eliminará:"
    echo "  - Directorio 'converted/' (archivos convertidos)"
    echo "  - Directorio 'transcriptions/' (transcripciones)"
    echo "  - Directorio 'embeddings/' (embeddings generados)"
    echo "  - Directorio 'indices/' (índices vectoriales)"
    echo "  - Directorio 'final/' (dataset final)"
    echo "  - Directorio 'temp_audio/' (archivos temporales)"
    echo "  - Archivos de log (*.log)"
    echo ""
    echo "Mantendrá:"
    echo "  - Archivos de audio originales"
    echo "  - Archivos de configuración (.env, *.json)"
    echo "  - Scripts y código fuente"
}

# Variables por defecto
DATASET_DIR=""
FORCE=false
VERBOSE=false
BACKUP=false
DRY_RUN=false

# Parsear argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --backup)
            BACKUP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo -e "${RED}❌ Opción desconocida: $1${NC}"
            echo "Usa -h o --help para ver las opciones disponibles"
            exit 1
            ;;
        *)
            if [[ -z "$DATASET_DIR" ]]; then
                DATASET_DIR="$1"
            else
                echo -e "${RED}❌ Demasiados argumentos${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Usar directorio por defecto si no se especifica
if [[ -z "$DATASET_DIR" ]]; then
    DATASET_DIR="./dataset"
fi

# Verificar que el directorio existe
if [[ ! -d "$DATASET_DIR" ]]; then
    echo -e "${RED}❌ Error: El directorio '$DATASET_DIR' no existe${NC}"
    exit 1
fi

echo -e "${BLUE}🧹 Limpieza del Dataset${NC}"
echo -e "${BLUE}=====================${NC}"
echo ""
echo -e "📁 Directorio: ${YELLOW}$DATASET_DIR${NC}"

# Función para verbose logging
log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "  ${BLUE}ℹ️  $1${NC}"
    fi
}

# Función para dry run logging
log_dry_run() {
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "  ${YELLOW}🔍 [DRY RUN] $1${NC}"
    fi
}

# Directorios y archivos a eliminar
DIRS_TO_CLEAN=(
    "converted"
    "transcriptions" 
    "embeddings"
    "indices"
    "final"
    "temp_audio"
    "backup"
)

FILES_TO_CLEAN=(
    "*.log"
    "pipeline.log"
    "sentiment_processing.log"
    "dataset_processing.log"
)

# Función para crear backup
create_backup() {
    if [[ "$BACKUP" == true ]]; then
        local backup_dir="$DATASET_DIR/backup_$(date +%Y%m%d_%H%M%S)"
        echo -e "${YELLOW}💾 Creando backup en: $backup_dir${NC}"
        
        if [[ "$DRY_RUN" == false ]]; then
            mkdir -p "$backup_dir"
            for dir in "${DIRS_TO_CLEAN[@]}"; do
                if [[ -d "$DATASET_DIR/$dir" ]]; then
                    log_verbose "Copiando $dir/ al backup"
                    cp -r "$DATASET_DIR/$dir" "$backup_dir/" 2>/dev/null || true
                fi
            done
            echo -e "${GREEN}✅ Backup creado exitosamente${NC}"
        else
            log_dry_run "Se crearía backup en: $backup_dir"
        fi
        echo ""
    fi
}

# Función para obtener el tamaño de un directorio
get_dir_size() {
    if [[ -d "$1" ]]; then
        du -sh "$1" 2>/dev/null | cut -f1 || echo "0B"
    else
        echo "0B"
    fi
}

# Mostrar qué se va a eliminar
echo -e "${YELLOW}📋 Elementos a eliminar:${NC}"
total_size=0

for dir in "${DIRS_TO_CLEAN[@]}"; do
    full_path="$DATASET_DIR/$dir"
    if [[ -d "$full_path" ]]; then
        size=$(get_dir_size "$full_path")
        file_count=$(find "$full_path" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo -e "  📂 $dir/ (${size}, ${file_count} archivos)"
        log_verbose "Ruta completa: $full_path"
    fi
done

for pattern in "${FILES_TO_CLEAN[@]}"; do
    files_found=$(find "$DATASET_DIR" -maxdepth 1 -name "$pattern" 2>/dev/null || true)
    if [[ -n "$files_found" ]]; then
        echo -e "  📄 Archivos $pattern"
        if [[ "$VERBOSE" == true ]]; then
            echo "$files_found" | while read -r file; do
                if [[ -n "$file" ]]; then
                    size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "0B")
                    log_verbose "$(basename "$file") (${size})"
                fi
            done
        fi
    fi
done

echo ""

# Verificar si hay algo que limpiar
has_content=false
for dir in "${DIRS_TO_CLEAN[@]}"; do
    if [[ -d "$DATASET_DIR/$dir" ]]; then
        has_content=true
        break
    fi
done

if [[ "$has_content" == false ]]; then
    echo -e "${GREEN}✅ El dataset ya está limpio${NC}"
    exit 0
fi

# Mostrar advertencia
echo -e "${RED}⚠️  ADVERTENCIA:${NC}"
echo -e "   Esta operación eliminará todos los datos procesados"
echo -e "   Los archivos de audio originales NO se eliminarán"
echo ""

# Crear backup si se solicita
create_backup

# Pedir confirmación (a menos que sea --force o --dry-run)
if [[ "$FORCE" == false && "$DRY_RUN" == false ]]; then
    echo -e "${YELLOW}❓ ¿Continuar con la limpieza? [y/N]:${NC} "
    read -r response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo ""
            ;;
        *)
            echo -e "${BLUE}ℹ️  Operación cancelada${NC}"
            exit 0
            ;;
    esac
fi

# Realizar la limpieza
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}🔍 MODO DRY RUN - Solo mostrando qué se eliminaría:${NC}"
else
    echo -e "${GREEN}🧹 Iniciando limpieza...${NC}"
fi

cleaned_items=0

# Eliminar directorios
for dir in "${DIRS_TO_CLEAN[@]}"; do
    full_path="$DATASET_DIR/$dir"
    if [[ -d "$full_path" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            log_dry_run "Eliminaría directorio: $dir/"
        else
            log_verbose "Eliminando directorio: $dir/"
            rm -rf "$full_path"
            if [[ $? -eq 0 ]]; then
                echo -e "  ${GREEN}✅ Eliminado: $dir/${NC}"
            else
                echo -e "  ${RED}❌ Error eliminando: $dir/${NC}"
            fi
        fi
        ((cleaned_items++))
    fi
done

# Eliminar archivos
for pattern in "${FILES_TO_CLEAN[@]}"; do
    files_found=$(find "$DATASET_DIR" -maxdepth 1 -name "$pattern" 2>/dev/null || true)
    if [[ -n "$files_found" ]]; then
        echo "$files_found" | while read -r file; do
            if [[ -n "$file" && -f "$file" ]]; then
                if [[ "$DRY_RUN" == true ]]; then
                    log_dry_run "Eliminaría archivo: $(basename "$file")"
                else
                    log_verbose "Eliminando archivo: $(basename "$file")"
                    rm -f "$file"
                    if [[ $? -eq 0 ]]; then
                        echo -e "  ${GREEN}✅ Eliminado: $(basename "$file")${NC}"
                    else
                        echo -e "  ${RED}❌ Error eliminando: $(basename "$file")${NC}"
                    fi
                fi
            fi
        done
        ((cleaned_items++))
    fi
done

echo ""

# Mostrar resumen
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}🔍 RESUMEN (DRY RUN):${NC}"
    echo -e "   Se eliminarían aproximadamente $cleaned_items elementos"
    echo -e "   Para ejecutar la limpieza real, quita la opción --dry-run"
else
    if [[ $cleaned_items -gt 0 ]]; then
        echo -e "${GREEN}✅ Limpieza completada exitosamente${NC}"
        echo -e "   Elementos eliminados: $cleaned_items"
        
        # Mostrar espacio liberado
        echo ""
        echo -e "${BLUE}📊 Estado actual del directorio:${NC}"
        remaining_size=$(get_dir_size "$DATASET_DIR")
        echo -e "   Tamaño actual: $remaining_size"
        
        # Listar lo que queda
        echo -e "   Archivos restantes:"
        find "$DATASET_DIR" -maxdepth 2 -type f 2>/dev/null | head -10 | while read -r file; do
            echo -e "     📄 $(basename "$file")"
        done
        
        remaining_count=$(find "$DATASET_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [[ $remaining_count -gt 10 ]]; then
            echo -e "     ... y $((remaining_count - 10)) archivos más"
        fi
    else
        echo -e "${BLUE}ℹ️  No había elementos para limpiar${NC}"
    fi
fi

echo ""
echo -e "${GREEN}🎉 Proceso terminado${NC}"

# Mostrar siguiente paso sugerido
if [[ "$DRY_RUN" == false ]]; then
    echo ""
    echo -e "${BLUE}💡 Siguiente paso sugerido:${NC}"
    echo -e "   Para regenerar el dataset ejecuta:"
    echo -e "   ${YELLOW}./build_corpus_dataset.sh${NC}"
fi