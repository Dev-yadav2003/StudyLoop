"""Generate requirements.txt / requirements-dev.txt from uv.lock (the source of truth).

    uv run python scripts/export_requirements.py           # regenerate both files
    uv run python scripts/export_requirements.py --check   # CI: fail if they're stale

requirements.txt      runtime deps + the project itself (`-e .`) — what Render installs
requirements-dev.txt  `-r requirements.txt` + test/lint/eval tools

Exact versions come from uv.lock, so `pip install -r ...` gets the same packages as
`uv sync`. No hashes: pip's hash-checking mode rejects the editable `-e .` install.
Never edit the generated files by hand — change pyproject.toml, run `uv lock`, re-export.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    "# Generated from uv.lock by scripts/export_requirements.py — do not edit by hand.\n"
    "# To change dependencies: edit pyproject.toml, run `uv lock`, then re-run the script.\n"
)


def uv_export(*args: str) -> str:
    out = subprocess.run(
        ["uv", "export", "--format", "requirements-txt", "--no-hashes", "--quiet", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # Drop uv's own header comment (it names the exact command) and keep ours.
    body = "\n".join(line for line in out.splitlines() if not line.startswith("#"))
    return body.strip() + "\n"


def render() -> dict[str, str]:
    runtime = uv_export("--no-default-groups")
    dev = uv_export("--only-group", "dev", "--only-group", "eval", "--no-emit-project")
    return {
        "requirements.txt": HEADER + runtime,
        "requirements-dev.txt": HEADER + "-r requirements.txt\n" + dev,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="exit 1 if the files are stale")
    args = parser.parse_args()

    stale = []
    for name, content in render().items():
        path = ROOT / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        stale.append(name)
        if not args.check:
            path.write_text(content, encoding="utf-8", newline="\n")
    if args.check and stale:
        print(
            f"Out of date: {', '.join(stale)}. Run: uv run python scripts/export_requirements.py",
            file=sys.stderr,
        )
        return 1
    print("up to date" if not stale else f"wrote {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
