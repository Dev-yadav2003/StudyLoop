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
    "ports": ({"lecnotes.domain", "lecnotes.ports"}, {"pydantic"}),
}


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
                ok = any(mod == p or mod.startswith(p + ".") for p in internal_ok)
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
