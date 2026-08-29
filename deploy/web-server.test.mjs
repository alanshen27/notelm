#!/usr/bin/env node
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

function listen(server) {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
}

function fetchText(url, opts = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(url, { method: opts.method || "GET", headers: opts.headers }, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        resolve({
          status: res.statusCode,
          headers: res.headers,
          body: Buffer.concat(chunks).toString("utf8"),
        });
      });
    });
    req.on("error", reject);
    if (opts.body) req.write(opts.body);
    req.end();
  });
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "notelm-web-"));
fs.mkdirSync(path.join(tmp, "app"), { recursive: true });
fs.writeFileSync(path.join(tmp, "index.html"), "<html><body>home</body></html>");
fs.writeFileSync(path.join(tmp, "app", "index.html"), "<html><body>clavier</body></html>");
fs.writeFileSync(path.join(tmp, "404.html"), "<html><body>missing</body></html>");

const api = http.createServer((req, res) => {
  if (req.url === "/api/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ ok: true }));
    return;
  }
  if (req.method === "POST" && req.url === "/api/continue") {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ echoed: Buffer.concat(chunks).toString("utf8") }));
    });
    return;
  }
  res.writeHead(404);
  res.end();
});

const apiPort = await listen(api);
const site = spawn(process.execPath, [path.join(here, "web-server.mjs")], {
  env: {
    ...process.env,
    PORT: "0",
    STATIC_ROOT: tmp,
    API_UPSTREAM: `http://127.0.0.1:${apiPort}`,
  },
  stdio: ["ignore", "pipe", "pipe"],
});

let sitePort = 0;
await new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error("site did not start")), 5000);
  const onExit = (code) => {
    clearTimeout(timer);
    reject(new Error(`site exited ${code}`));
  };
  const onData = (buf) => {
    const text = buf.toString("utf8");
    const m = text.match(/site :(\d+)/);
    if (m) {
      sitePort = Number(m[1]);
      clearTimeout(timer);
      site.off("exit", onExit);
      site.stdout.off("data", onData);
      resolve();
    }
  };
  site.stdout.on("data", onData);
  site.stderr.on("data", (b) => process.stderr.write(b));
  site.on("exit", onExit);
});

// PORT=0 lets the OS pick; Node's http.Server with listen(0) works, but our
// script uses Number(process.env.PORT || 3000) and listen(PORT). 0 is valid.
if (!sitePort) {
  throw new Error("could not parse site port");
}

try {
  const home = await fetchText(`http://127.0.0.1:${sitePort}/`);
  assert.equal(home.status, 200);
  assert.match(home.body, /home/);

  const slash = await fetchText(`http://127.0.0.1:${sitePort}/app`);
  assert.equal(slash.status, 308);
  assert.equal(slash.headers.location, "/app/");

  const app = await fetchText(`http://127.0.0.1:${sitePort}/app/`);
  assert.equal(app.status, 200);
  assert.match(app.body, /clavier/);

  const alias = await fetchText(`http://127.0.0.1:${sitePort}/afterbar`);
  assert.equal(alias.status, 307);
  assert.equal(alias.headers.location, "/clavier/");

  const health = await fetchText(`http://127.0.0.1:${sitePort}/api/health`);
  assert.equal(health.status, 200);
  assert.match(health.body, /"ok":true/);

  const cont = await fetchText(`http://127.0.0.1:${sitePort}/api/continue`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ notes: [{ pitch: 60, start: 0, duration: 1 }] }),
  });
  assert.equal(cont.status, 200);
  assert.match(cont.body, /pitch\\":60/);

  const missing = await fetchText(`http://127.0.0.1:${sitePort}/nope`);
  assert.equal(missing.status, 404);
  assert.match(missing.body, /missing/);

  console.log("web-server tests passed");
} finally {
  site.kill("SIGTERM");
  api.close();
  fs.rmSync(tmp, { recursive: true, force: true });
}
