// Typed API client. Types come from the FastAPI OpenAPI schema (`npm run gen:api`),
// so a backend contract change shows up as a TypeScript error here, not at runtime.
import createClient from "openapi-fetch";

import type { components, paths } from "./api-types";
import { accessToken } from "./supabase";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export const api = createClient<paths>({ baseUrl: API_URL });

// Every request carries the signed-in user's token (the API scopes data and limits by it).
api.use({
  async onRequest({ request }) {
    const token = await accessToken();
    if (token && !request.headers.has("Authorization")) {
      request.headers.set("Authorization", `Bearer ${token}`);
    }
    return request;
  },
});

type Schemas = components["schemas"];
export type Job = Schemas["JobOut"];
export type Notes = Schemas["NotesOut"];
export type Unit = Notes["units"][number];
export type UnitEdit = Schemas["UnitEdit"];
export type StageProgress = Schemas["StageProgress"];
export type SubjectPreset = NonNullable<Schemas["LessonCreate"]["subject_preset"]>;
export type Me = Schemas["MeOut"];
export type QaLevel = NonNullable<Schemas["QaCreate"]["level"]>;
export type QuestionSet = Schemas["QuestionSetOut"];
export type Question = Schemas["QuestionOut"];
export type Chapter = Schemas["ChapterOut"];
export type Lesson = Schemas["LessonOut"];

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

/** Unwrap an openapi-fetch result: the data, or a thrown ApiError. */
export async function call<T>(
  request: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await request;
  if (error !== undefined || data === undefined) throw toApiError(error, response.status);
  return data;
}

/** Same as `call` for endpoints that answer 204 No Content. */
export async function callEmpty(
  request: Promise<{ error?: unknown; response: Response }>,
): Promise<void> {
  const { error, response } = await request;
  if (!response.ok) throw toApiError(error, response.status);
}

export const exportUrl = (notesId: string) => `${API_URL}/v1/notes/${notesId}/export?format=md`;

export function formatTime(seconds: number): string {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}

export const youtubeAt = (videoId: string, seconds: number) =>
  `https://youtu.be/${videoId}?t=${Math.floor(seconds)}`;
