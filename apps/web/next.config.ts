import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV !== "production";

const nextConfig: NextConfig = {
  trailingSlash: true,
  images: { unoptimized: true },
  ...(isDev
    ? {
        async redirects() {
          return [
            { source: "/afterbar", destination: "/clavier/", permanent: false },
            { source: "/afterbar/", destination: "/clavier/", permanent: false },
          ];
        },
        async rewrites() {
          return [
            {
              source: "/api/:path*",
              destination: "http://127.0.0.1:8000/api/:path*",
            },
          ];
        },
      }
    : { output: "export" as const }),
};

export default nextConfig;
