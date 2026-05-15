from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = [
    Path("docs/tutorials/vqe_h2_mitigation.ipynb"),
    Path("docs/tutorials/qaoa_maxcut_mitigation.ipynb"),
]

STALE_OR_OVERCLAIMED_TEXT = [
    "20%",
    "10 to 40",
    "minimum threshold of 10",
    "hardware validation",
    "dominant risk",
    "ready to connect to your simulator or hardware executor",
    "state-of-the-art",
    "revolutionary",
    "battle-tested",
    "enterprise-grade",
    "production-ready",
    "optimal",
    "\u2014",
]


def _read_notebook(path: Path) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return notebook


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_tutorial_notebooks_are_valid_v4(path: Path) -> None:
    notebook = _read_notebook(path)

    assert notebook.nbformat == 4


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_tutorial_notebooks_keep_colab_badge(path: Path) -> None:
    notebook = _read_notebook(path)
    first_cell = notebook.cells[0]

    assert first_cell.cell_type == "markdown"
    assert "colab-badge.svg" in first_cell.source
    assert "colab.research.google.com" in first_cell.source


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_tutorial_code_cells_are_python_syntax(path: Path) -> None:
    notebook = _read_notebook(path)

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source.lstrip()
        if not source or source.startswith("!"):
            continue
        ast.parse(cell.source)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_tutorial_outputs_are_cleared(path: Path) -> None:
    notebook = _read_notebook(path)

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is None
        assert cell.outputs == []


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_tutorial_notebooks_avoid_stale_claims(path: Path) -> None:
    notebook = _read_notebook(path)
    text = "\n".join(cell.source for cell in notebook.cells)
    text_lower = text.lower()

    for phrase in STALE_OR_OVERCLAIMED_TEXT:
        assert phrase.lower() not in text_lower
