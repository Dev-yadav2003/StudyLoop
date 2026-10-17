# Testing everything on your laptop

Everything runs locally against your real services (Supabase, Gemini, YouTube Data API,
Langfuse). Deploying to Render/Vercel comes later.

## 0. One-time setup

```bash
uv sync                          # Python deps
uv run alembic upgrade head      # Supabase schema → latest (0006)
cd apps/web && npm install       # frontend deps (Node ≥ 20.9)
cp .env.example .env.local       # NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

`.env` (repo root) must have `GEMINI_API_KEY`, `YOUTUBE_API_KEY`, `DATABASE_URL` (Supabase
**session pooler**), `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`. Optional: `ADMIN_TOKEN`.

## 1. Automated checks (≈ 30 s, no network)

```bash
uv run ruff check . && uv run pyright && uv run pytest     # 379 tests incl. real Postgres
cd apps/web && npm run typecheck && npm run build
```

## 2. Start the two processes (two terminals)

```bash
uv run lecnotes serve            # API + job runner → http://127.0.0.1:8000
cd apps/web && npm run dev       # UI → http://localhost:3000
```

Check the API first:

- http://127.0.0.1:8000/readyz → `{"status":"ok","checks":{"database":"ok","worker":"ok"}}`
- http://127.0.0.1:8000/docs → interactive API docs

## 3. Happy path in the browser

1. Open http://localhost:3000, paste e.g. `https://www.youtube.com/watch?v=Ek49cGiAOwo`,
   pick **STEM**, click **Make notes**.
2. You land on `/jobs/<id>`: stages tick from *waiting* to *done* every few seconds
   (a 5-10 min lecture takes ~2-4 min; "Watching the video" and "Writing notes" are the slow
   ones).
3. When done it opens `/notes/<id>`: sections, code blocks, `[mm:ss]` links (open YouTube at
   that moment in a new tab), **Copy Markdown**, **Download .md**.
4. Submit the same video again (even as `youtu.be/...`): you go straight to the same job — no
   new work (idempotent).
5. **Practice questions** (under the notes): pick a level and count, **Generate**. Each
   answer is behind *Show answer*, with links to where it is taught. Generating again with the
   same settings is instant (cached).
6. **Edit notes**: fix a concept name or explanation, **Save** → the notes re-render at once
   (no AI call; an *edited* tag marks the unit). Reload: the edit is still there. Timestamps
   and sources can't be edited. An admin `rerun` replaces the notes, edits included.

## 4. Things worth trying

| Try | Expected |
|---|---|
| A playlist link (`youtube.com/playlist?list=…`) | Inline error: "Playlists aren't supported…" |
| A non-YouTube link | Inline error: "That doesn't look like a link to a YouTube video." |
| A private/unlisted video | Job page: "This video can't be processed…" |
| Stop the API mid-job (Ctrl+C), start it again | The job continues by itself after ~2 min (lease expiry); finished stages are reused from the cache |
| Stop the API, reload a job page | "The server is waking up…" after 5 s, recovers when it's back |
| `uv run lecnotes run <url>` | Same pipeline from the CLI, prints per-stage cost/latency |

Admin (needs `ADMIN_TOKEN` in `.env`):

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" http://127.0.0.1:8000/v1/admin/jobs/<id>/stages
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  "http://127.0.0.1:8000/v1/admin/jobs/<id>/rerun?from=extract"
```

## 5. Where to look when something is off

- API terminal: JSON logs with `job_id`, `stage`, `request_id` on every line.
- Langfuse: one trace per job → one span per stage → one generation per Gemini call.
- Supabase: `jobs` (status, error, attempts), `stage_runs` (per-stage latency/tokens/cost),
  `notes` / `units`.
- The `X-Request-ID` shown in an API error body matches the API log lines.
