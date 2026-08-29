/** Public paths. Trailing slashes match Next `output: "export"`. */
export const routes = {
  home: "/",
  models: "/models/",
  modelsHash: "/#models",
  research: "/research/",
  clavier: "/clavier/",
  app: "/app/",
  dev: "/dev/",
} as const;

export const aliases: Record<string, string> = {
  "/afterbar": routes.clavier,
  "/afterbar/": routes.clavier,
  "/playground": routes.app,
  "/playground/": routes.app,
};

export function siteOrigin(): string {
  const raw = process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/+$/, "");
  if (raw) return raw;
  return process.env.NODE_ENV === "production"
    ? "http://localhost:8000"
    : "http://localhost:3000";
}

/** Empty base = same origin (`/api/...`). Set NEXT_PUBLIC_API_URL to split hosts. */
export function apiUrl(path: string): string {
  const base = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/+$/, "");
  return `${base}${path}`;
}
