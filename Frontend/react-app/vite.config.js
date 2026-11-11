import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  publicDir: false,
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8100", changeOrigin: true } },
  },
  build: {
    outDir: "dist",
    rollupOptions: {
      output: {
        manualChunks: {
          maps: ["leaflet", "react-leaflet"],
          charts: ["chart.js", "react-chartjs-2"],
        },
      },
    },
  },
});
