"""ASGI entrypoint: `uvicorn apps.api.main:app` (Linux / Render).

On Windows use `uv run lecnotes serve`, which runs uvicorn on a SelectorEventLoop (psycopg's
async mode doesn't support the default ProactorEventLoop).
"""

from apps.api.app import create_app

app = create_app()
