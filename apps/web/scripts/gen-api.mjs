// Regenerate lib/api-types.ts from the running API's OpenAPI schema (cross-platform:
// npm runs scripts with cmd.exe on Windows, where ${VAR:-default} doesn't expand).
import { execFileSync } from "node:child_process";

const api = process.env.API_URL ?? "http://127.0.0.1:8000";
execFileSync(
  process.execPath,
  ["node_modules/openapi-typescript/bin/cli.js", `${api}/openapi.json`, "-o", "lib/api-types.ts"],
  { stdio: "inherit" },
);
