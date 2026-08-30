# Intenciones de usuario del agente

El agente está pensado para periodistas que necesitan localizar y verificar
contenido dentro de un corpus de audio ya procesado. Responde en español por
defecto y fundamenta los hallazgos con archivo de origen, timestamps e índice
consultado.

## Resumen de capacidades

| Intención | Qué resuelve | Herramienta | Evidencia esperada |
|---|---|---|---|
| Buscar contenido dicho | Frases, temas, nombres o conceptos presentes en transcripciones. | `buscar_audio` | Texto del segmento, archivo, inicio/fin y similitud del índice textual. |
| Buscar un sonido | Eventos que pueden no figurar en la transcripción, como aplausos o música. | `buscar_evento_acustico` | Segmento, archivo, inicio/fin y similitud del índice CLAP. |
| Pedir detalle de un segmento | Metadatos completos de un segmento conocido. | `obtener_info_segmento` | Datos del segmento identificado por `segment_id`. |
| Interpretar etiquetas de audio | Clases AudioSet detectadas para un segmento recuperado. | `obtener_clases_audio` | Etiquetas YAMNet en inglés y scores del clasificador. |

## Intenciones soportadas

### 1. Buscar contenido en transcripciones

**Objetivo:** encontrar declaraciones o contenido semánticamente relacionado
con una consulta periodística.

**Ejemplos de mensajes:**

- “¿Qué se dijo sobre inflación en las entrevistas de enero?”
- “Buscá menciones a subsidios energéticos.”
- “¿En qué parte hablan de la renuncia del ministro?”
- “Mostrame hasta 10 fragmentos sobre educación pública.”

**Comportamiento esperado:** el agente usa el índice de texto de
transcripciones y devuelve los segmentos más relevantes. Cada resultado debe
indicar `Índice de texto (transcripciones)`, el archivo de origen y el rango
temporal.

**Límites:** admite entre 1 y 20 resultados. Una consulta vacía devuelve un
error de validación. Si el índice textual no está disponible, el agente debe
informarlo en lugar de inventar resultados.

### 2. Buscar eventos acústicos

**Objetivo:** encontrar sonidos descritos en lenguaje natural, incluso cuando
no fueron transcriptos.

**Ejemplos de mensajes:**

- “Buscá aplausos durante un discurso político.”
- “¿Dónde se escucha música de fondo?”
- “Encontrá risas o gritos del público.”
- “Mostrame sonidos de sirena.”

**Comportamiento esperado:** el agente consulta el índice de audio CLAP con la
descripción del sonido. En resultados en español, la consulta puede traducirse
internamente para el encoder de CLAP; la respuesta se mantiene en español.
Cada hallazgo debe citar `Índice de audio (CLAP)`, archivo y timestamps.

**Límites:** la similitud de CLAP no es una certeza ni una clasificación de
audio. Debe presentarse como relevancia del resultado y conviene revisar el
fragmento original antes de publicarlo.

### 3. Consultar detalle de un segmento

**Objetivo:** recuperar información ampliada de un segmento que ya fue
identificado en una búsqueda.

**Ejemplos de mensajes:**

- “Dame más información del segmento 42.”
- “¿Cuál es el archivo y duración del segmento 17?”
- “Mostrame los metadatos del resultado con id 9.”

**Comportamiento esperado:** el agente busca el `segment_id` y devuelve sus
metadatos completos. Si no existe, informa claramente que no fue encontrado.

### 4. Consultar clases YAMNet de un segmento

**Objetivo:** inspeccionar etiquetas acústicas estandarizadas detectadas para
un segmento recuperado.

**Ejemplos de mensajes:**

- “¿Qué clases de audio detectó YAMNet en el segmento 42?”
- “¿El segmento 17 tiene etiquetas de aplausos o música?”
- “Explicame las clases acústicas del resultado 9.”

**Comportamiento esperado:** el agente devuelve las etiquetas AudioSet en
inglés, junto a sus scores. Debe aclarar que los scores son probabilidades del
clasificador YAMNet, no porcentajes de similitud CLAP.

**Disponibilidad:** esta intención requiere que el dataset se haya procesado
con YAMNet. Si no contiene esas clases, el agente informa cómo reprocesarlo;
no intenta inferirlas.

## Intenciones compuestas

El usuario puede encadenar intenciones. Por ejemplo:

1. “Buscá aplausos después de hablar de inflación.”
2. “Dame los datos completos del primer resultado.”
3. “¿Qué clases YAMNet tiene ese segmento?”

El agente primero recupera candidatos, luego usa el `segment_id` de un
resultado para ampliar metadatos o consultar YAMNet. Las referencias deben
mantenerse vinculadas al segmento original.

## Comportamientos no soportados o condicionados

- **Subir, transcribir o indexar audio nuevo:** no es una intención del agente
  de consulta. Se realiza previamente mediante el pipeline de ingesta.
- **Afirmar hechos no presentes en los resultados:** el agente debe limitarse
  a la evidencia recuperada.
- **Consultar contenido fuera del dataset configurado:** no tiene acceso a
  fuentes externas ni debe exponer rutas locales o credenciales.
- **Búsqueda de texto o audio deshabilitada:** `AGENT_MODALITY=text` expone
  sólo la búsqueda textual; `AGENT_MODALITY=audio`, sólo la acústica; `both`
  habilita ambas.
- **Resultados vacíos:** debe decir que no encontró coincidencias y sugerir una
  reformulación, sin fabricar segmentos.

## Contrato de respuesta periodística

Para cada hallazgo, la respuesta debe incluir cuando estén disponibles:

- archivo de origen;
- timestamp de inicio y fin;
- texto del segmento;
- identificador de segmento;
- índice usado, con su etiqueta exacta;
- una aclaración sobre la naturaleza de la similitud cuando corresponda.

Este contrato permite verificar el audio fuente antes de utilizar un resultado
en una nota, informe o publicación.
