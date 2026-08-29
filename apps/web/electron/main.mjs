import { app, BrowserWindow } from "electron";

function createWindow() {
  const win = new BrowserWindow({
    width: 1320,
    height: 820,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: "#171717",
    title: "clavier",
    webPreferences: {
      contextIsolation: true,
      sandbox: true,
    },
  });
  const url =
    process.env.CLAVIER_URL ||
    process.env.AFTERBAR_URL ||
    (process.env.ELECTRON_DEV === "1"
      ? "http://127.0.0.1:3000/app/"
      : "http://127.0.0.1:8000/app/");
  win.loadURL(url);
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => app.quit());
