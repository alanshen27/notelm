import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command }) => {
  const electron = process.env.ELECTRON === "1";
  return {
    plugins: [react()],
    base: electron ? "./" : command === "build" ? "/app/" : "/",
    server: {
      port: 5174,
      proxy: {
        "/api": "http://127.0.0.1:8000",
      },
    },
  };
});
