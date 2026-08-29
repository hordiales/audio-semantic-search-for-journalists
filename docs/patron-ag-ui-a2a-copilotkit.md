# Patrón de diseño: AG-UI, A2A, CopilotKit y Cloud Run

## Propósito

Exponer el agente de búsqueda semántica de audio a una interfaz web sin dar al
navegador acceso directo a Agent Runtime ni a credenciales de Google Cloud.

```text
Navegador + widget CopilotKit
        │ AG-UI (HTTP streaming)
        ▼
Proxy en Cloud Run
        │ A2A JSON-RPC + IAM/ADC
        ▼
Agente ADK en Agent Runtime
        │
        ▼
Índices FAISS y dataset de audio
```

## Responsabilidades

| Componente | Responsabilidad | No debe hacer |
|---|---|---|
| Widget | Renderizar el chat y enviar la conversación usando AG-UI. | Conocer credenciales o URL interna de Agent Runtime. |
| Proxy | Traducir AG-UI a A2A, aplicar CORS/rate limit y obtener credenciales ADC. | Contener lógica de retrieval ni claves de modelos. |
| Agent Runtime | Ejecutar ADK, herramientas de búsqueda y respuestas con evidencia. | Exponer una API pública de navegador. |

## Elección de protocolo

CopilotKit entiende AG-UI, mientras que el agente publica A2A. El proxy es un
adaptador deliberado: evita acoplar la interfaz al contrato interno de ADK y
permite reemplazar el agente sin reescribir el widget.

Para el único agente actual se usa un puente determinista: toma el último
mensaje de usuario, invoca `message/send` A2A y lo emite como
`TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT` y `TEXT_MESSAGE_END`. No añade
un LLM orquestador, por lo que no incorpora coste, latencia ni respuestas
ajenas a la búsqueda periodística.

## Autenticación y seguridad

Agent Runtime exige IAM también en la ruta A2A. El SDK A2A no permite inyectar
headers de autenticación; por eso el proxy intercepta `fetch` hacia hosts
`*-aiplatform.googleapis.com` y agrega un token de Application Default
Credentials de la service account de Cloud Run.

La service account necesita `roles/aiplatform.user` en el proyecto que aloja
el agente. El widget no recibe ese token. CORS limita los orígenes autorizados
pero no reemplaza autenticación de usuarios; si el chat contiene datos no
públicos, agregar IAP, Firebase Auth o un mecanismo equivalente antes de
exponer el proxy.

## Operación

- Configurar `A2A_AGENT_URL` con la URL completa del endpoint del agente.
- Configurar `ALLOWED_ORIGINS` con los dominios reales del widget; no usar `*`.
- Usar `--update-env-vars` al modificar Cloud Run para no borrar otras
  variables de entorno.
- Publicar el bundle del widget en CDN/hosting estático, no desde el proxy.
- Mantener el agente en Agent Runtime y el proxy como un servicio Cloud Run
  independiente; cada uno tiene ciclo de despliegue y escalado propios.

## Extensión a varios agentes

Si se agregan agentes, el proxy necesita una política explícita de routing.
Puede ser determinista (por ruta/intención fija) o un orquestador LLM. No se
debe habilitar un orquestador LLM por defecto: requiere una evaluación propia,
añade coste y puede enviar consultas a un agente incorrecto.
