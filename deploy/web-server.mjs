#!/usr/bin/env node
/**
 * Serve the Next static export and proxy /api to the inference service.
 * Same public origin (notelm.onrender.com) so the UI can keep calling /api/...
 */
import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, process.env.STATIC_ROOT || "out");
const PORT = Number(process.env.PORT || 3000);
const UPSTREAM = (process.env.API_UPSTREAM || "https://notelm-api.onrender.com").replace(
  /\/+$/,
  "",
);

const ALIASES = {
  "/afterbar": "/clavier/",
  "/afterbar/": "/clavier/",
};

const MIME = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".map": "application/json",
  ".midi": "audio/midi",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
  ".woff2": "font/woff2",
  ".xml": "application/xml",
};

function safeJoin(root, reqPath) {
  const decoded = decodeURIComponent(reqPath.split("?")[0]);
  const rel = path.posix.normalize(decoded).replace(/^(\.\.(\/|$))+/, "");
  const abs = path.join(root, rel);
  if (!abs.startsWith(root)) return null;
  return abs;
}

function sendFile(res, filePath, extraHeaders = {}, status = 200) {
  const ext = path.extname(filePath).toLowerCase();
  res.writeHead(status, {
    "Content-Type": MIME[ext] || "application/octet-stream",
    ...extraHeaders,
  });
  fs.createReadStream(filePath).pipe(res);
}

function proxy(req, res) {
  const dest = new URL(req.url, UPSTREAM);
  const lib = dest.protocol === "https:" ? https : http;
  const headers = { ...req.headers, host: dest.host };
  delete headers.connection;
  const upstream = lib.request(dest, { method: req.method, headers }, (up) => {
    res.writeHead(up.statusCode || 502, up.headers);
    up.pipe(res);
  });
  upstream.on("error", (err) => {
    if (res.headersSent) {
      res.destroy();
      return;
    }
    res.writeHead(502, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(`API proxy failed: ${err.message}`);
  });
  upstream.setTimeout(10 * 60 * 1000, () => {
    upstream.destroy();
    if (!res.headersSent) {
      res.writeHead(504, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("API timeout");
    }
  });
  req.pipe(upstream);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url || "/", "http://localhost");
  const pathname = url.pathname;

  if (pathname === "/api" || pathname.startsWith("/api/")) {
    proxy(req, res);
    return;
  }

  const alias = ALIASES[pathname];
  if (alias) {
    res.writeHead(307, { Location: alias + url.search });
    res.end();
    return;
  }

  if (pathname !== "/" && !pathname.endsWith("/")) {
    const leaf = pathname.split("/").pop();
    if (leaf && !leaf.includes(".")) {
      const asDir = path.join(ROOT, pathname, "index.html");
      if (fs.existsSync(asDir)) {
        res.writeHead(308, { Location: `${pathname}/${url.search}` });
        res.end();
        return;
      }
    }
  }

  let filePath = safeJoin(ROOT, pathname);
  if (!filePath) {
    res.writeHead(400, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("bad path");
    return;
  }
  if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
    filePath = path.join(filePath, "index.html");
  }
  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    const extra =
      pathname.startsWith("/_next/static/")
        ? { "Cache-Control": "public, max-age=31536000, immutable" }
        : {};
    sendFile(res, filePath, extra);
    return;
  }

  const fallback = path.join(ROOT, "404.html");
  if (fs.existsSync(fallback)) {
    sendFile(res, fallback, {}, 404);
    return;
  }
  res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
  res.end("Not found");
});

server.listen(PORT, "0.0.0.0", () => {
  const { port } = server.address();
  console.log(`notelm site :${port}  static=${ROOT}  /api → ${UPSTREAM}`);
});
