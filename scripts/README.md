# Scripts - Utilidades Generales

Este directorio contiene scripts de utilidad general del proyecto.

## Estructura

### sql/

Scripts SQL para configuración de bases de datos:

- **supabase_setup.sql** - Setup inicial de Supabase
- **create_embeddings_table_manual.sql** - Creación manual de tabla de embeddings
- **audio_embeddings_comparison_setup.sql** - Setup para comparación de embeddings
- **insert_audio_embeddings.sql** - Inserciones de embeddings de audio

### shell/

Scripts bash para automatización:

- **build_corpus_dataset.sh** - Construir corpus de dataset
- **clean_dataset.sh** - Limpiar dataset
- **download-from-youtbe.sh** - Descargar audio de YouTube
- **play_file.sh** - Reproducir archivo de audio
- **quick_install.sh** - Instalación rápida
- **run_clap_full.sh** - Ejecutar CLAP completo
- **run_clap_test.sh** - Test de CLAP
- **run_cmd_line.sh** - Ejecutar línea de comandos

### Python

- **fix_ruff_errors.py** - Script para corregir errores de Ruff

## Uso

### Scripts SQL

Ejecutar en Supabase Dashboard o usando psql:

```bash
psql $DATABASE_URL < scripts/sql/supabase_setup.sql
```

### Scripts Shell

Desde la raíz del proyecto:

```bash
bash scripts/shell/build_corpus_dataset.sh
```

O hacerlos ejecutables:

```bash
chmod +x scripts/shell/*.sh
./scripts/shell/build_corpus_dataset.sh
```

### Scripts Python

```bash
poetry run python scripts/fix_ruff_errors.py
```
