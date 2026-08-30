import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

const containerId = "audio-search-widget";
function mount(): void {
  let container = document.getElementById(containerId);
  if (!container) {
    container = document.createElement("div");
    container.id = containerId;
    document.body.appendChild(container);
  }
  createRoot(container).render(<React.StrictMode><App /></React.StrictMode>);
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
else mount();
