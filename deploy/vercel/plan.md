# Plan de Evolución: Ejecución de CLAP e Infraestructura de Inferencia para el Stack Vercel + Neon

Ver también el documento raíz [`../../plan.md`](../../plan.md).

Este resumen recoge las decisiones de diseño futuras para ejecutar **CLAP (Contrastive Language-Audio Pretraining)** dentro del despliegue **Vercel + Neon (`pgvector`) + Eve Agents**:

1. **Separación Web vs. Inferencia:** Vercel sirve el frontend y las rutas `/api/*` + `/eve/v1/*`, delegando la inferencia de CLAP (`512` dims) por HTTP (`CUSTOM_EMBEDDING_API_URL`) a un contenedor dedicado en **Google Cloud Run** (o **Modal** / **Hugging Face Endpoints**) y persistiendo/consultando los vectores en **Neon PostgreSQL (`pgvector`)**.
2. **Vercel Large Functions (5 GB con Fluid Compute):** Evaluar `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` para cargas bajas o para ejecutar únicamente el **text encoder** de CLAP en CPU sin infraestructura externa.
3. **Optimización Audio Encoder vs. Text Encoder:** Procesar los archivos de audio una sola vez en batch (con GPU en Cloud Run Jobs o Modal), guardar los embeddings en Neon y ejecutar en tiempo de consulta únicamente el *text encoder* de CLAP contra `pgvector`.
