# Testing everything on your laptop

Everything runs locally against your real services (Supabase, Gemini, YouTube Data API,
Langfuse).

## 0. One-time setup

```bash
uv sync                          # Python deps
cp .env.example .env              # fill in the REQUIRED section (see comments in the file)
uv run alembic upgrade head      # create/upgrade the schema
cd apps/web && npm install       # frontend deps (Node ≥ 20.9)
cp .env.example .env.local       # API URL + Supabase URL + publishable key
```

Supabase → **Authentication → URL Configuration**: Site URL `http://localhost:3000`, and
`http://localhost:3000/**` under Redirect URLs — the sign-in email links back there.

## 1. Automated checks (≈ 30 s, no network)

```bash
uv run ruff check . && uv run pyright && uv run pytest     # 455 tests incl. real Postgres
cd apps/web && npm run typecheck && npm run build
```

## 2. Start the two processes (two terminals)

```bash
uv run lecnotes serve            # API + job runner → http://127.0.0.1:8000
cd apps/web && npm run dev       # UI → http://localhost:3000
```

Check the API first:

- http://127.0.0.1:8000/readyz → `{"status":"ok","checks":{"database":"ok","schema":"ok","worker":"ok"}}`
  (`schema` names the missing migration if `alembic upgrade head` wasn't run)
- http://127.0.0.1:8000/docs → interactive API docs

## 3. Happy path in the browser

0. **Sign in:** enter your email → open the link in the email (same browser). With your
   email in `OWNER_EMAILS` the sidebar shows *Owner · unlimited*; any other email is a guest
   with 60 minutes of video in total (shown as a bar in the sidebar).
1. **Chapters** (left sidebar): **+** next to *Chapters* → type a name (e.g. `Python`),
   Enter. Hover a chapter → **+** (add a lesson to it), ✏️ (rename), 🗑 (delete it and its
   lessons — the notes themselves are kept).
2. **New lesson**: paste e.g. `https://www.youtube.com/watch?v=Ek49cGiAOwo`, pick the chapter
   (or *New chapter…*), optionally a lesson name, the subject type (General / STEM / Theory),
   **Make notes**.
3. You land on the lesson: a progress bar and each step ticking off (a 5-10 min lecture takes
   ~2-4 min). The sidebar dot pulses while it works, turns green when done, red if it failed
   (**Try again** on the lesson page).
4. **Notes** tab: numbered sections, one card per concept (definition, bullets, code,
   examples), ▶ `mm:ss` chips open YouTube at that moment, *On this page* outline on wide
   screens, **Copy Markdown**, **Download**. No "On the board" dumps.
5. Click the lesson title (or ✏️ in the sidebar) to rename it; Enter saves, Esc cancels.
6. **Practice** tab: pick a level and count, **Generate questions**. *Show answer* reveals it
   with where it's taught; mark **Got it** / **Review this** — the score counts up. The same
   settings again are instant (cached).
7. **Edit a note**: hover a concept card → ✏️ → change it → **Save**: the notes re-render at
   once (no AI call; an *edited* tag appears). Timestamps and sources can't be edited. An admin
   `rerun` replaces the notes, edits included.
8. **Feedback** (bottom of the notes): 👍 sends at once; 👎 asks what was wrong. It lands in
   Supabase (`feedback`) and in Langfuse as a `user_feedback` score on the job's trace.
9. Add the same video to another chapter: it reuses the finished notes instantly.
10. Theme: the sun/moon at the bottom of the sidebar. On a phone, ☰ opens the library.

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
curl -H "Authorization: Bearer $ADMIN_TOKEN" "http://127.0.0.1:8000/v1/admin/feedback?rating=down"
```

On Windows PowerShell use `curl.exe` (plain `curl` is a different command there).

## 5. Where to look when something is off

- API terminal: JSON logs with `job_id`, `stage`, `request_id` on every line.
- Langfuse: one trace per job → one span per stage → one generation per Gemini call.
- Supabase: `jobs` (status, error, attempts), `stage_runs` (per-stage latency/tokens/cost),
  `notes` / `units`.
- The `X-Request-ID` shown in an API error body matches the API log lines.
