# Test Suite Overview

## Resumen

El repositorio ahora cuenta con una jerarquía unificada de pruebas dentro de la carpeta `tests/`, organizada por tipo de verificación (unitarias, funcionales, de integración y heredadas). Cada script `test_*.py` fue migrado a esta estructura y utiliza utilidades compartidas que normalizan las rutas hacia `src/`, recursos y artefactos de salida.

## Estructura principal

```
tests/
├── common/                 # Utilidades compartidas (path utils, etc.)
├── unit/                   # Pruebas de componentes aislados
├── functional/             # Flujos funcionales ligeros sin infra completa
├── integration/            # Escenarios de mayor alcance (Supabase, MCP, CLAP…)
├── legacy/                 # Scripts históricos o exploratorios
├── artifacts/              # Carpeta común para salidas generadas por las pruebas
├── resources/              # Configuración y datos auxiliares compartidos
├── test_catalog.json       # Catálogo de pruebas por categoría
└── run.py                  # Orquestador CLI para ejecutar suites
```

Las utilidades en `tests/common/path_utils.py` resuelven automáticamente el directorio raíz del proyecto, añaden `src/` al `sys.path`, localizan recursos compartidos y crean subcarpetas bajo `tests/artifacts/<nombre>` para mantener los outputs ordenados.

## Artefactos y recursos

- Todos los scripts escriben resultados en subdirectorios de `tests/artifacts/` (por ejemplo `components/`, `semantic_heatmap/`, `benchmark_quick/`).
- Los archivos de configuración reutilizables (p.ej. `test_config.json`, `test_supabase_config.json`) se invocan desde `tests/resources/`.
- Los scripts antiguos (`tests/legacy/`) permanecen disponibles, pero ahora usan las mismas utilidades de rutas para garantizar compatibilidad con el nuevo layout.

## Ejecución de pruebas

El orquestador `tests/run.py` permite listar y ejecutar categorías completas:

```bash
# Mostrar el catálogo completo
python tests/run.py --list

# Ejecutar categorías específicas
python tests/run.py unit
python tests/run.py functional integration

# Ejecutar todo el catálogo (incluye legacy)
python tests/run.py all
```

También se puede ejecutar cualquier script individual utilizando `python -m <módulo>`; por ejemplo:

```bash
python -m tests.functional.test_audio_embeddings_simple
python -m tests.integration.test_supabase_integration
```

El runner utiliza `subprocess.run` con `sys.executable` para que cada módulo se ejecute en un proceso independiente, lo cual aísla fallos y simplifica el análisis de logs.

## Dependencias y notas

- Algunas pruebas requieren dependencias pesadas o acceso a modelos/descargas externos (TensorFlow, `laion-clap`, `sentence-transformers`, etc.). Verifica que estén instaladas antes de ejecutar las suites funcionales e integraciones que las utilizan.
- En entornos sandbox o sin red las descargas de modelos pueden fallar; se recomienda ejecutar estas pruebas en una máquina con las credenciales/modelos ya cacheados.
- Los módulos heredados (`tests/legacy/tmp/*`) dependen de utilidades adicionales ubicadas en la carpeta `tmp/` del proyecto principal. El helper de rutas los expone automáticamente.

## Próximos pasos sugeridos

1. Revisar `tests/test_catalog.json` para ajustar la agrupación de pruebas si se agregan nuevos escenarios.
2. Integrar `tests/run.py` con el sistema de CI/CD deseado para automatizar suites críticas (por ejemplo, ejecutar `unit` y `functional` en cada commit).
3. Documentar en los scripts las dependencias específicas (modelos, variables de entorno) para facilitar la ejecución en entornos limpios.

