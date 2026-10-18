import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // A second dev server (e.g. a preview next to `npm run dev`) needs its own build folder.
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
};

export default config;
