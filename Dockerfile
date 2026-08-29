# CPU inference image: Next static export + FastAPI + PyTorch CPU.
# CUDA wheels are huge and useless on Render (no GPU).
FROM node:22-bookworm-slim AS ui
WORKDIR /ui
ENV NEXT_TELEMETRY_DISABLED=1
ENV npm_config_registry=https://registry.npmjs.org
ARG NEXT_PUBLIC_SITE_URL=https://notelm.onrender.com
ARG NEXT_PUBLIC_API_URL=
ENV NEXT_PUBLIC_SITE_URL=$NEXT_PUBLIC_SITE_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
COPY apps/web/package.json ./
RUN npm install --omit=optional
COPY apps/web ./
RUN npm run build

FROM python:3.13-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir \
        "pretty_midi>=0.2.11" \
        "tqdm>=4.66" \
        "fastapi>=0.115" \
        "uvicorn>=0.32" \
        "python-multipart>=0.0.12"

COPY src ./src
COPY --from=ui /ui/out ./apps/web/out

WORKDIR /app/src
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
