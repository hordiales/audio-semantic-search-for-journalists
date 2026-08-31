# Resultados de la evaluación de CLAP en Clotho

## Propósito

Este documento registra el resultado de correr el benchmark público Clotho con la
implementación de CLAP del proyecto. Sirve como evidencia de que el *wiring* del
modelo (carga, resampleo, normalización y ranking) está funcionando.

> **No es evidencia de transferencia al dominio periodístico en español.** Para
> esa conclusión ver `docs/reporte-comparativo-texto-clap.md`.

---

## Configuración

| Parámetro | Valor |
| --- | --- |
| Fecha | 2026-08-23 |
| Modelo | `laion/clap-htsat-unfused` |
| `enable_fusion` | `False` |
| Dispositivo | `cpu` |
| Dataset | Clotho v2.1 split `evaluation` |
| Clips de audio | 1045 |
| Captions (queries) | 5225 (5 captions por clip) |
| Ventana de audio CLAP | 10 s con 2 s de solapamiento |
| Embedding dimensión | 512 |

El audio se convirtió internamente por LAION-CLAP a 48 kHz y 480.000 muestras
(10 s). El pipeline del proyecto genera ventanas de 10 s, así que cada clip de
Clotho (15–30 s) se dividió en ventanas superpuestas cuyos embeddings se
promediaron y normalizaron L2.

---

## Resultados agregados

| Métrica | Valor obtenido | Referencia publicada (Wu et al., 2023) |
| --- | --- | --- |
| Recall@1 | 0.1458 | ~0.167 |
| Recall@5 | 0.3684 | ~0.411 |
| Recall@10 | 0.4945 | ~0.541 |
| MRR | 0.2593 | — |
| nDCG@1 | 0.1458 | — |
| nDCG@5 | 0.2607 | — |
| nDCG@10 | 0.3017 | — |
| Precision@1 | 0.1458 | — |
| Precision@5 | 0.0737 | — |
| Precision@10 | 0.0495 | — |

## Interpretación

Los números obtenidos están dentro del mismo orden de magnitud que los
publicados para CLAP-HTSAT entrenado en AudioCaps+Clotho+WT5K (Wu et al., 2023,
Tabla 3). Esto permite concluir que:

- El modelo se carga correctamente.
- El resampleo a 48 kHz y la ventana de 10 s funcionan como espera LAION-CLAP.
- La normalización L2 antes de la búsqueda y el uso de `IndexFlatIP` sobre
  embeddings normalizados son correctos.
- El text encoder recibe captions en inglés sin pasar por traducción.

La diferencia respecto a los valores de referencia puede deberse a que el
publicado puede haber usado el checkpoint *fusion*, un protocolo ligeramente
distinto de evaluación o media sobre más splits. Para los fines de este proyecto
la coincidencia de orden de magnitud es suficiente.

---

## Archivos generados

- `evaluation/results/clap_clotho_eval.json` — resultado completo con
  configuración, métricas agregadas y resultados por query.
- `evaluation/results/.clap_clotho_audio_embeddings.npy` — cache de los
  embeddings de audio de los 1045 clips (reutilizable para otras corridas).

---

## Cómo reproducir

```bash
uv run python -m benchmarks.evaluate_clap_clotho \
    --audio-dir data/clotho/evaluation \
    --captions-csv data/clotho/clotho_captions_evaluation.csv \
    --output evaluation/results/clap_clotho_eval.json \
    --cache evaluation/results/.clap_clotho_audio_embeddings.npy
```

Si Clotho no está descargado, seguir primero los pasos de descarga en
`docs/evaluar-clap-clotho.md` §1.
