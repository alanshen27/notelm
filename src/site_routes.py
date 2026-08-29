"""Public site URLs for the Next static export (must match apps/web/src/lib/routes.ts)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

PAGES = {
    "home": "/",
    "models": "/models/",
    "research": "/research/",
    "clavier": "/clavier/",
    "app": "/app/",
    "dev": "/dev/",
}

ALIASES = {
    "/afterbar": PAGES["clavier"],
    "/afterbar/": PAGES["clavier"],
    "/playground": PAGES["app"],
    "/playground/": PAGES["app"],
}


def public_origin() -> str:
    raw = (
        os.environ.get("SITE_URL")
        or os.environ.get("RENDER_EXTERNAL_URL")
        or os.environ.get("NEXT_PUBLIC_SITE_URL")
        or ""
    ).strip()
    return raw.rstrip("/")


def _redirect(url: str, request: Request, status: int = 307) -> RedirectResponse:
    qs = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(url + qs, status_code=status)


def mount_exported_site(app: FastAPI, web_out: Path) -> None:
    @app.middleware("http")
    async def site_urls(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api"):
            return await call_next(request)
        dest = ALIASES.get(path)
        if dest:
            return _redirect(dest, request)
        if path != "/" and not path.endswith("/"):
            leaf = path.rsplit("/", 1)[-1]
            if "." not in leaf:
                index = web_out / path.lstrip("/") / "index.html"
                if index.is_file():
                    return _redirect(path + "/", request, status=308)
        return await call_next(request)

    if web_out.is_dir() and (web_out / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web_out, html=True), name="web")
