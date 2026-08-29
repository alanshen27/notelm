# Public site: Next static export + /api reverse-proxy to notelm-api.
# Electron is optional; lightningcss's native binary is also optional — we need the latter.
FROM node:22-bookworm-slim AS ui
WORKDIR /ui
ENV NEXT_TELEMETRY_DISABLED=1
ENV npm_config_registry=https://registry.npmjs.org
ARG NEXT_PUBLIC_SITE_URL=https://notelm.onrender.com
ARG NEXT_PUBLIC_API_URL=
ENV NEXT_PUBLIC_SITE_URL=$NEXT_PUBLIC_SITE_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
COPY apps/web/package.json ./
RUN npm install --omit=optional \
    && VER="$(node -p "JSON.parse(require('fs').readFileSync('node_modules/lightningcss/package.json','utf8')).version")" \
    && npm install --no-save --omit=optional "lightningcss-linux-x64-gnu@$VER"
COPY apps/web ./
RUN npm run build

FROM node:22-bookworm-slim
WORKDIR /app
ENV NODE_ENV=production
ENV API_UPSTREAM=https://notelm-api.onrender.com
COPY --from=ui /ui/out ./out
COPY deploy/web-server.mjs ./web-server.mjs
EXPOSE 3000
CMD ["node", "web-server.mjs"]
