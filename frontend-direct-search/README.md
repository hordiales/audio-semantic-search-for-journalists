# Frontend — búsqueda directa de archivo sonoro

Interfaz de búsqueda estructurada para periodistas que trabajan con corpus de audio. Es una aplicación independiente del widget conversacional: consulta la API REST, muestra el plan efectivo y permite escuchar cada segmento sin salir de la página.

## Qué incluye

- Búsqueda de transcripciones por defecto y dos fuentes acústicas opt-in e
  independientes: el índice vectorial CLAP y las clases AudioSet de YAMNet.
- Con `Reformular con IA`, el proxy pide por A2A al agente un plan estructurado
  antes del retrieval. El planner reformula las queries, pero respeta los
  fuentes elegidas: texto siempre, más CLAP y/o YAMNet si se activaron.
- Plan efectivo visible, editable y copiable para repetir el retrieval.
- Resultados separados por índice: nunca mezcla scores de espacios vectoriales distintos.
- Cada tarjeta identifica explícitamente su origen: texto/transcripciones,
  audio/CLAP o etiquetas YAMNet/AudioSet.
- Reproductor inline por segmento, con contexto opcional de ±10 segundos.
- Atajos: `Espacio` reproduce/pausa el resultado enfocado; `J` y `K` recorren resultados.
- Reproducción continua del ranking, descarga de clips y copia de citas.
- Filtros locales por archivo y por timestamp.
- Renovación preventiva de URLs firmadas antes de que expiren.

## Requisitos

- Node.js 20 o superior.
- La API estructurada del backend disponible según [la especificación](../spec/08-frontend-busqueda-directa.md), en particular `POST /api/search`.

## Desarrollo local

```bash
cd frontend-direct-search
npm install
npm run dev
```

Vite inicia normalmente en `http://localhost:5173` y redirige las peticiones `/api/*` a `http://localhost:8080`. Por ello el backend debe estar disponible en ese puerto durante el desarrollo.

En producción, `VITE_API_BASE_URL` debe apuntar al **proxy** Cloud Run, no a
`audio-search-service`: éste último conserva IAM y sólo es invocado por el
proxy. El proxy expone `/api/search` al navegador y devuelve las URLs firmadas
de audio.

## Configuración

Copiá el archivo de ejemplo si necesitás cambiar los valores por defecto:

```bash
cp .env.example .env
```

| Variable | Uso |
| --- | --- |
| `VITE_API_BASE_URL` | Origen del backend cuando frontend y API viven en dominios distintos. No incluir `/` al final. Vacío usa rutas relativas `/api`. |
| `VITE_WEAK_TEXT_SIMILARITY` | Umbral opcional para advertir coincidencias débiles del índice de texto. |
| `VITE_WEAK_AUDIO_SIMILARITY` | Umbral opcional para advertir coincidencias débiles de CLAP. |
| `VITE_WEAK_YAMNET_SCORE` | Umbral opcional para advertir coincidencias débiles del ranking por clases YAMNet. |

Los umbrales deben medirse sobre el corpus real; no se configuran por defecto para no presentar valores arbitrarios como criterio editorial.

### Activar o desactivar CLAP y YAMNet desde el frontend

La búsqueda CLAP está **desactivada por defecto**. El control **Incluir índice
CLAP** aparece junto al formulario:

- Desmarcado: `POST /api/search` recibe `include_clap: false` y sólo consulta
  las transcripciones.
- Marcado: recibe `include_clap: true`, mantiene la búsqueda textual y agrega
  el ranking texto→audio de CLAP en una segunda columna.

El estado se puede compartir en la URL. `?clap=1` lo activa; si el parámetro no
está presente, permanece desactivado. Por ejemplo:

```text
/direct-search/?q=aplausos&clap=1&k=10
```

Al habilitar CLAP, el servicio traduce la descripción acústica al inglés,
calcula su embedding en el espacio compartido y devuelve la traducción efectiva
en `plan.audio_query_en` y `indexes.audio.translated_query`.

YAMNet tiene un segundo control, **Incluir clases YAMNet**, también
**desactivado por defecto** e independiente de CLAP:

- Desmarcado: se envía `include_yamnet: false` y no se consulta esta fuente.
- Marcado: se envía `include_yamnet: true`; el servicio traduce la descripción
  al inglés y compara sus términos con las etiquetas AudioSet almacenadas por
  segmento. El ranking combina cobertura de clases y scores del clasificador.

YAMNet no crea un índice vectorial FAISS: es una búsqueda/filtro sobre las
clases producidas durante la ingesta. Para que esté disponible, el release debe
haberse generado con `EMBEDDINGS_ACTIVE=...,yamnet`. Cuando CLAP está activo,
sus tarjetas muestran además las clases YAMNet disponibles como señal auxiliar.

El estado compartible es `?yamnet=1`; las tres fuentes se habilitan con:

```text
/direct-search/?q=aplausos&clap=1&yamnet=1&k=10
```

## Compilar y previsualizar

```bash
npm run build
npm run preview
```

El bundle se genera en `dist/`. Puede servirse desde estáticos de FastAPI o desde un servicio web separado. Si se despliega por separado, configurá CORS del backend para el origen del frontend y definí `VITE_API_BASE_URL` al compilar.

En este repositorio, el workflow `Deploy GitHub Pages` lo publica bajo
`/direct-search/` dentro del mismo sitio que la portada y el widget existente.
Antes de ejecutarlo, creá la variable de repositorio `DIRECT_SEARCH_API_URL`
con la URL HTTPS del proxy, por ejemplo `https://audio-search-copilotkit-proxy-…a.run.app`.

## Contrato esperado de API

La aplicación realiza una petición a:

```text
POST /api/search
```

El body recomendado contiene `query`, `include_clap` e `include_yamnet` (ambos
default `false`), `k`, `rewrite` y, opcionalmente, `plan`. `indexes` sigue aceptándose para clientes
anteriores y para reproducir planes guardados. La respuesta incluye el plan de
búsqueda, resultados separados por índice, URLs firmadas de clip y sus tiempos
de expiración.

El frontend no usa CopilotKit, A2A ni el endpoint conversacional. Si la URL de un clip está a punto de vencer, vuelve a ejecutar el mismo plan para recibir referencias renovadas.

## Estructura

```text
frontend-direct-search/
├── src/
│   ├── App.tsx       # Estado, búsqueda y componentes de interfaz
│   ├── api.ts        # Cliente REST
│   ├── types.ts      # Contrato TypeScript de la API
│   └── styles.css    # Diseño responsive
├── vite.config.ts    # Proxy local /api
└── .env.example      # Configuración de despliegue
```
