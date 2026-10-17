# Lecture-to-Notes

Turn a public YouTube lecture (or playlist) into faithful, timestamped study notes covering
what the teacher **said** and what they **wrote or showed**.

See [`plan.md`](plan.md) for the architecture and build plan, and
[`docs/devlog.md`](docs/devlog.md) for progress notes and measured results.

## Setup

With plain pip (Python 3.12):

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt   # app + tests/lint/evals (runtime only: requirements.txt)
```

Or with [uv](https://docs.astral.sh/uv/) — same pinned versions, faster:

```bash
uv sync                      # core + dev + eval dependencies
cp .env.example .env         # fill in secrets — never put them in .env.example
uv run alembic upgrade head  # create/upgrade the schema in DATABASE_URL
uv run pre-commit install
```

`DATABASE_URL` for Supabase must be the **Session pooler** string (the direct host is
IPv6-only); URL-encode special characters in the password (`@` → `%40`).

## Run

Full local walkthrough (API + UI + things to try): [`docs/local-testing.md`](docs/local-testing.md).

```bash
uv run lecnotes serve                               # API + job runner → http://127.0.0.1:8000/docs
cd apps/web && npm install && npm run dev           # UI → http://localhost:3000
uv run lecnotes run <youtube-url> --preset stem     # notes/<video_id>.md + per-stage cost
uv run lecnotes run <url> --rerun-from extract      # recompute from a stage (rest cached)
uv run lecnotes run <url> --no-db --no-trace        # in-memory cache, no Langfuse
```

## Evaluate

```bash
# Golden set (hand-made ground truth) — one entry per lecture:
uv run python scripts/new_golden_video.py <url> --type code --terms "zip,enumerate" --captions
uv run python scripts/try_perceive.py <url> --duration MM:SS   # perception fixture

uv run python -m evals.perception_report    # WER, board recall, boundary F1
uv run python -m evals.run --label "what changed" [--stage extract]
uv run python -m evals.compare --latest     # before/after: metrics, stages, latency, cost
```

PRs labelled `eval` run the eval suite in CI and fail if faithfulness or keypoint recall
drops by more than 5 points versus `evals/baseline.json`.

## Dependencies

`pyproject.toml` declares them, `uv.lock` pins every version, and `requirements*.txt` are
generated from the lock (never edit them by hand):

```bash
uv add <package>                                  # or edit pyproject.toml, then: uv lock
uv run python scripts/export_requirements.py      # regenerate requirements*.txt
```

CI fails if `requirements*.txt` don't match `uv.lock`.

## Develop

```bash
uv run ruff format . && uv run ruff check .   # format + lint
uv run pyright                                # type check
uv run pytest                                 # unit + integration (never touches the network)
```

Integration tests use `TEST_DATABASE_URL`, or create a throwaway PostgreSQL cluster from
local binaries (`PG_BIN`), or are skipped.
