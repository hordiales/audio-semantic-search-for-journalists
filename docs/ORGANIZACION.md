# Organización de la Documentación

Este documento explica la organización de la documentación del proyecto.

## Estructura

```
.
├── README.md                    # README principal (raíz)
└── doc/
    ├── README.md               # Índice de documentación
    ├── INSTALLATION.md         # Guía de instalación consolidada
    ├── QUICK_START.md          # Inicio rápido
    ├── TROUBLESHOOTING.md      # Solución de problemas
    ├── DATASET.md              # Guía de datasets
    ├── ARCHITECTURE_long.md    # Arquitectura del sistema
    ├── API_README.md           # API REST
    ├── MCP.md                  # Servidor MCP
    ├── REQUIREMENTS_PYTHON.md  # Requisitos de Python
    ├── CHANGELOG_POETRY.md     # Changelog Poetry
    └── archive/                # Archivos archivados/obsoletos
```

## Consolidaciones Realizadas

### Instalación
- **INSTALL.md** + **INSTALL_POETRY.md** + **REQUIREMENTS_PYTHON.md** → **INSTALLATION.md**
  - Consolidado en un solo archivo con todas las opciones
  - Mantiene información de Poetry y pip
  - Incluye requisitos de Python

### MCP
- **MCP_SETUP.md** → **doc/MCP.md**
  - Renombrado y movido a doc/

### Datasets
- **DATASET.md** (raíz) → **doc/DATASET.md**
  - Movido a doc/ para mantener organización

## Archivos Archivados

Los siguientes archivos fueron movidos a `doc/archive/` por ser:
- Temporales/obsoletos
- Resultados de análisis históricos
- Configuraciones antiguas
- Resúmenes de implementación históricos

### Análisis y Planes
- ANALISIS_Y_PLAN_LIMPIEZA.md
- RESUMEN_EJECUTIVO.md
- CODEX_REVIEW.md
- TODO.md

### Benchmarks y Resultados
- CHUNKING_STRATEGIES_REPORT.md
- EMBEDDINGS_EXECUTION_RESULTS.md
- REALISTIC_BENCHMARK_RESULTS.md
- BENCHMARK_RESULTS_FINAL.md
- EMBEDDING_BENCHMARK_README.md

### Configuraciones y Setup Históricos
- EMBEDDING_MODELS_FINAL_SETUP.md
- EMBEDDING_MODELS_SETUP.md
- MODELS_CONFIGURATION.md
- FINAL_SUPABASE_SUMMARY.md
- SUPABASE_MIGRATION_GUIDE.md
- VECTOR_DATABASE_SYSTEM.md
- SYSTEM_STATUS_FINAL.md
- YAMNET_FIXES.md
- CLAUDE.md

## Principios de Organización

1. **README.md en raíz**: Punto de entrada principal, conciso
2. **doc/README.md**: Índice completo de toda la documentación
3. **Consolidación**: Agrupar contenido relacionado en un solo archivo
4. **Archive**: Mover archivos obsoletos pero mantenerlos para referencia
5. **Nombres claros**: Usar nombres descriptivos y consistentes

## Actualización de Referencias

Si encuentras referencias a archivos movidos, actualízalas:

- `INSTALL.md` → `doc/INSTALLATION.md`
- `INSTALL_POETRY.md` → `doc/INSTALLATION.md` (consolidado)
- `REQUIREMENTS_PYTHON.md` → `doc/REQUIREMENTS_PYTHON.md`
- `MCP_SETUP.md` → `doc/MCP.md`
- `DATASET.md` → `doc/DATASET.md`

## Mantenimiento

Para mantener la organización:

1. **Nuevos documentos**: Añadir a `doc/` con nombre descriptivo
2. **Actualizar índice**: Actualizar `doc/README.md` cuando se añadan documentos
3. **Archivar obsoletos**: Mover a `doc/archive/` en lugar de eliminar
4. **Consolidar**: Agrupar contenido relacionado cuando sea posible
