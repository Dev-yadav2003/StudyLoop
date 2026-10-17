"""Import-lint: enforce the dependency direction from plan §4.1 by parsing imports with ast.

Add a rule when a new layer gets real code. Rules are checked statically, so a violating
import fails here even if it sits inside a function body or TYPE_CHECKING block.
"""

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "lecnotes"

# layer -> (allowed lecnotes.* prefixes, allowed third-party top-level packages or None = any)
RULES: dict[str, tuple[set[str], set[str] | None]] = {
    "domain": ({"lecnotes.domain"}, {"pydantic"}),
    # ports may also use the pure cache-key value object
    "ports": ({"lecnotes.domain", "lecnotes.ports", "lecnotes.cache"}, {"pydantic"}),
    "cache": ({"lecnotes.cache"}, {"pydantic"}),
    "prompts": ({"lecnotes.prompts"}, {"pydantic", "jinja2", "yaml", "langchain_core"}),
    "observability": (
        {
            "lecnotes",
            "lecnotes.config",
            "lecnotes.domain",
            "lecnotes.observability",
            "lecnotes.ports",
        },
        # langchain_core only for the Langfuse<->LangChain callback bridge.
        {"structlog", "langfuse", "langchain_core"},
    ),
    "llm": (
        {
            "lecnotes.config",
            "lecnotes.domain",
            "lecnotes.errors",
            "lecnotes.llm",
            "lecnotes.observability",
            "lecnotes.prompts",
        },
        {"pydantic", "langchain_core", "langchain_google_genai", "google"},
    ),
}

# Stages get concrete dependencies through ports: never import adapters or storage.
RULES["pipeline"] = (
    {
        "lecnotes.cache",
        "lecnotes.config",
        "lecnotes.domain",
        "lecnotes.errors",
        "lecnotes.llm",
        "lecnotes.observability",
        "lecnotes.pipeline",
        "lecnotes.ports",
        "lecnotes.prompts",
        "lecnotes.timefmt",
    },
    {"pydantic", "rapidfuzz"},
)

RULES["storage"] = (
    {"lecnotes.config", "lecnotes.domain", "lecnotes.storage"},
    {"sqlalchemy", "alembic", "psycopg"},
)

# Plan §4.3: no LLM call outside the gateway (or the perception adapter, Phase 3).
LLM_SDKS = {"langchain_google_genai", "google.genai", "google.generativeai"}
LLM_SDK_ALLOWED = {"llm/gateway.py", "adapters/perception/gemini_url.py"}


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def violations(layer: str) -> list[str]:
    internal_ok, third_party_ok = RULES[layer]
    bad: list[str] = []
    for path in sorted((SRC / layer).rglob("*.py")):
        for mod in imports_of(path):
            top = mod.split(".")[0]
            if top == "lecnotes":
                # The bare package matches only itself (`from lecnotes import __version__`);
                # as a prefix it would allow every lecnotes.* import.
                ok = any(
                    mod == p or (p != "lecnotes" and mod.startswith(p + ".")) for p in internal_ok
                )
            elif top in sys.stdlib_module_names or top == "__future__":
                ok = True
            else:
                ok = third_party_ok is None or top in third_party_ok
            if not ok:
                bad.append(f"{path.relative_to(SRC)} imports {mod}")
    return bad


@pytest.mark.parametrize("layer", sorted(RULES))
def test_layer_dependencies(layer: str) -> None:
    assert violations(layer) == []


def test_llm_sdks_only_imported_by_gateway() -> None:
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in LLM_SDK_ALLOWED:
            continue
        for mod in imports_of(path):
            if any(mod == sdk or mod.startswith(sdk + ".") for sdk in LLM_SDKS):
                offenders.append(f"{rel} imports {mod}")
    assert offenders == []
