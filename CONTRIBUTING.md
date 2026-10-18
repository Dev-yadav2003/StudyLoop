# Contributing

## Set up

Follow **Getting started** in the [README](README.md), then turn on the commit hooks:

```bash
uv run pre-commit install      # ruff + pyright run on every commit
```

## Workflow

1. Start from the latest `main`:
   ```bash
   git checkout main && git pull
   ```
2. Create a branch named by type:
   `feature/…`, `fix/…`, `docs/…`, `refactor/…`, `test/…`, `chore/…`
3. Make a focused change: one idea per pull request, ideally under ~400 changed lines.
4. Commit with [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add a timer to practice sessions
   fix: keep one timestamp per distinct link in answers
   docs: explain Supabase redirect URLs
   ```
   First line under ~72 characters; explain *why* in the body when it isn't obvious.
5. Push and open a pull request. Fill in the template. CI (`check`, `web`) must be green.
   `main` is protected: every change arrives through a pull request.

## Before you push

```bash
uv run ruff format . && uv run ruff check .
uv run pyright
uv run pytest
cd apps/web && npm run typecheck
```

## Guidelines

- **Tests:** new behaviour needs a test; a bug fix needs a test that fails without it.
  Tests never call real services (a network guard enforces it): use `FakeLLM` and the
  fakes in `tests/`.
- **Layers:** the dependency direction is enforced by `tests/unit/test_architecture.py`.
  Only `src/lecnotes/llm/gateway.py` calls a language model.
- **Prompts:** never edit a released prompt file — add `prompts/<name>/v<N+1>.yaml`, bump the
  default in `config.py` and `.env.example`, and show the eval comparison in the PR.
- **Stages:** if a stage's output changes for the same input, bump its `version` so cached
  results are recomputed.
- **Database:** schema changes need an Alembic migration
  (`uv run alembic revision --autogenerate -m "…"`); the schema test checks it matches the
  models.
- **API:** after changing it, regenerate the web client: `cd apps/web && npm run gen:api`.
- **Dependencies:** add with `uv add`, then `uv run python scripts/export_requirements.py`.
- **Decisions:** record choices others might question as an ADR in `docs/adr/`.
