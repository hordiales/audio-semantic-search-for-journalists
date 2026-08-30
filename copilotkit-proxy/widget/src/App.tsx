import { CopilotKit } from "@copilotkit/react-core";
import { CopilotPopup } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import type { AnchorHTMLAttributes, ReactNode } from "react";

declare const __RUNTIME_URL__: string | undefined;

const runtimeUrl = __RUNTIME_URL__ || import.meta.env.VITE_COPILOT_RUNTIME_URL || "http://localhost:8080/api/copilotkit";

function isSignedClipUrl(value: string | undefined): value is string {
  if (!value) return false;
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname.endsWith("storage.googleapis.com") && url.pathname.endsWith(".opus");
  } catch {
    return false;
  }
}

function ClipLink({ href, children, ...props }: AnchorHTMLAttributes<HTMLAnchorElement> & { children?: ReactNode }) {
  if (!isSignedClipUrl(href)) {
    return <a {...props} href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
  }
  return (
    <audio controls preload="none" src={href}>
      <a {...props} href={href} target="_blank" rel="noopener noreferrer">{children}</a>
    </audio>
  );
}

export default function App() {
  return (
    <CopilotKit runtimeUrl={runtimeUrl} agent="default" useSingleEndpoint showDevConsole={false}>
      <CopilotPopup
        instructions="Respondé en español y citá el archivo y timestamps que entregue el agente."
        labels={{ title: "Búsqueda de audio", initial: "¿Qué querés encontrar en el archivo de audio?" }}
        markdownTagRenderers={{ a: ClipLink }}
      />
    </CopilotKit>
  );
}
