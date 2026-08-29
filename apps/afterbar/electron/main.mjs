import { app, BrowserWindow } from "electron";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dir = path.dirname(fileURLToPath(import.meta.url));

function createWindow() {
  const win = new BrowserWindow({
    width: 1320,
    height: 820,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: "#0c0c0e",
    title: "clavier",
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 14, y: 12 },
    webPreferences: {
      contextIsolation: true,
      sandbox: true,
    },
  });
  const dev = process.env.CLAVIER_DEV === "1" || process.env.AFTERBAR_DEV === "1";
  if (dev) {
    win.loadURL("http://127.0.0.1:5174");
  } else {
    win.loadFile(path.join(dir, "../dist/index.html"));
  }
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => app.quit());
