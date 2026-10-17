// Typed API client. Types come from the FastAPI OpenAPI schema (`npm run gen:api`),
// so a backend contract change shows up as a TypeScript error here, not at runtime.
import createClient from "openapi-fetch";

import type { components, paths } from "./api-types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export const api = createClient<paths>({ baseUrl: API_URL });

export type Job = components["schemas"]["JobOut"];
export type Notes = components["schemas"]["NotesOut"];
export type Unit = Notes["units"][number];
export type UnitEdit = components["schemas"]["UnitEdit"];
export type StageProgress = components["schemas"]["StageProgress"];
export type SubjectPreset = components["schemas"]["JobCreate"]["subject_preset"];
export type QaLevel = NonNullable<components["schemas"]["QaCreate"]["level"]>;
export type QuestionSet = components["schemas"]["QuestionSetOut"];

/** The API's uniform error body ({"error": {code, message, request_id}}) or a network error. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status: number | null,
  ) {
    super(message);
  }
}

export function toApiError(error: unknown, status: number | null): ApiError {
  const body = error as { error?: { code?: string; message?: string } } | undefined;
  if (body?.error?.message) {
    return new ApiError(body.error.message, body.error.code ?? "error", status);
  }
  const detail = (error as { detail?: { msg?: string }[] } | undefined)?.detail?.[0]?.msg;
  return new ApiError(detail ?? "Unexpected response from the server.", "error", status);
}

export const exportUrl = (notesId: string) => `${API_URL}/v1/notes/${notesId}/export?format=md`;
