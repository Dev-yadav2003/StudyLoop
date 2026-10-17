# Lecture-to-Notes

Turn a public YouTube lecture (or playlist) into faithful, timestamped study notes covering
what the teacher **said** and what they **wrote or showed**.

See [`plan.md`](plan.md) for the architecture and build plan, and
[`docs/devlog.md`](docs/devlog.md) for progress notes.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install core + dev dependencies
cp .env.example .env         # fill in secrets
uv run pre-commit install    # enable git hooks

uv run ruff check .          # lint
uv run pyright               # type check
uv run pytest                # tests (never touch the network)
```
