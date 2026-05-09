import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    host: true,
    // Proxy backend calls in dev so the browser doesn't need CORS on the API.
    // In production, set VITE_API_URL to point directly at your backend host.
    proxy: {
      "/stations": { target: "http://127.0.0.1:9000", changeOrigin: true },
      "/predict": { target: "http://127.0.0.1:9000", changeOrigin: true },
    },
  },
});
