import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, Vite serves the screen and forwards API calls to FastAPI on :8000.
// In production, FastAPI serves the built files from web/dist itself.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000", "/health": "http://127.0.0.1:8000" },
  },
});
