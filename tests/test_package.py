from __future__ import annotations

import re
from importlib.metadata import version as dist_version
from pathlib import Path

import pytest

import hotdata_langchain as hl

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "hotdata_langchain"
_RUNTIME_SUBMODULE = re.compile(
    r"(?m)^\s*(?:from\s+hotdata_framework\.(client|env|result|health)\s+import"
    r"|import\s+hotdata_framework\.(client|env|result|health)(?:\s|$|,|as))"
)
# `langchain`, but not `langchain_core` or any other langchain_* distribution.
_LANGCHAIN_IMPORT = re.compile(r"(?m)^\s*(?:from|import)\s+langchain(?=[.\s,]|$)")


def test_version_is_pep440_core() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\+.*)?", hl.__version__)


def test_version_matches_distribution_metadata() -> None:
    assert dist_version("hotdata-langchain") == hl.__version__


@pytest.mark.parametrize("name", hl.__all__)
def test_public_export_is_importable(name: str) -> None:
    assert hasattr(hl, name), f"missing export: {name}"
    assert getattr(hl, name) is not None


def test_source_uses_hotdata_framework_root_imports() -> None:
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if _RUNTIME_SUBMODULE.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(REPO_ROOT)))
    assert not violations, (
        "Use `from hotdata_framework import ...` in package source; "
        f"found submodule imports in: {', '.join(violations)}"
    )


def test_source_imports_only_langchain_core() -> None:
    """`langchain` is an extra, which only holds while nothing in the runtime imports it."""
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in SOURCE_ROOT.rglob("*.py")
        if _LANGCHAIN_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert not violations, (
        "package source imports `langchain`, which a bare install does not provide; "
        f"either use `langchain_core` or promote the extra to a dependency: {violations}"
    )


def test_the_agents_extra_provides_the_package_create_agent_lives_in() -> None:
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert any(req.startswith("langchain>") for req in extras["agents"]), extras
