import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === "build" ? "/ui/" : "/",
  server: { port: 3000, proxy: { "/api": "http://127.0.0.1:9000" } },
}));
