"""
Regression tests for GenerationSource Literal type + List→list modernization.

Cherry-picked from upstream PR #631 (commit 560c79f — security fix removing
client-writable `source` field from GenerationRequest). Verifies:

1. ``GenerationSource`` Literal type rejects unknown source values at validation
   time on the response model.
2. ``GenerationRequest`` does NOT accept a client-supplied ``source`` field —
   clients must not be able to spoof request provenance.
3. ``GenerationResponse.source`` is typed as ``GenerationSource``.
4. ``list[...]`` (PEP 585) built-in generic syntax is used in the touched
   models (no ``typing.List``).
"""

import sys
from pathlib import Path

import pytest

# tests/ lives at backend/tests/ — make backend/ importable so we can do
# ``from backend.models import ...`` (which keeps the package's relative
# imports in models.py working).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.models import (  # noqa: E402
    GenerationRequest,
    GenerationResponse,
    GenerationSource,
)
from datetime import datetime


VALID_SOURCES = ("mcp", "rest", "manual", "import", "personality_speak")


def test_generation_source_literal_accepts_all_valid_values():
    """All documented source values should round-trip through the Literal type."""
    for value in VALID_SOURCES:
        resp = GenerationResponse(
            id="g-1",
            profile_id="p-1",
            text="hello",
            language="en",
            source=value,
            created_at=datetime.now(),
        )
        assert resp.source == value


def test_generation_source_literal_rejects_unknown_value():
    """Pydantic must reject unknown source values — closed Literal set."""
    with pytest.raises(Exception):
        GenerationResponse(
            id="g-1",
            profile_id="p-1",
            text="hello",
            language="en",
            source="spoofed-source",
            created_at=datetime.now(),
        )


def test_generation_request_does_not_accept_source_field():
    """Clients must NOT be able to spoof request provenance by sending source.

    This is the security fix from upstream commit 560c79f: ``source`` was
    previously exposed on the public API model, allowing clients to lie about
    whether a request came from MCP/REST. Now source is server-controlled only.

    Pydantic v2 silently drops unknown fields by default, so we verify two
    things: (1) the ``source`` field is not in the declared schema, and
    (2) even if a client supplies it, it does not become a model attribute.
    """
    # (1) The schema must not declare a ``source`` field at all.
    assert "source" not in GenerationRequest.model_fields

    # (2) A request with a client-supplied source must not retain it.
    req = GenerationRequest(
        profile_id="p-1",
        text="hello",
        language="en",
        source="rest",  # client-supplied — must be silently dropped
    )
    # Pydantic drops unknown fields; check the model_dump doesn't include one.
    dumped = req.model_dump()
    assert "source" not in dumped, (
        "GenerationRequest.model_dump() leaked a client-supplied source — "
        "the upstream security fix (560c79f) regressed"
    )


def test_generation_request_without_source_is_valid():
    """A normal /generate call without source is fine — server fills it in."""
    req = GenerationRequest(
        profile_id="p-1",
        text="hello",
        language="en",
    )
    assert "source" not in GenerationRequest.model_fields
    assert "source" not in req.model_dump()


def test_generation_source_is_typed_as_literal():
    """Type annotation on GenerationResponse.source should be the Literal alias."""
    field = GenerationResponse.model_fields["source"]
    # Pydantic stores the annotation as a string; check it references the alias.
    annotation = str(field.annotation)
    assert "GenerationSource" in annotation or "Literal" in annotation


def test_generation_source_literal_has_five_members():
    """The Literal should have exactly the 5 documented source values."""
    # GenerationSource is ``Literal["mcp", "rest", "manual", "import", "personality_speak"]``
    # Pydantic exposes the args via __args__ when the alias is resolved.
    args = getattr(GenerationSource, "__args__", None)
    assert args is not None, "GenerationSource must be a Literal type"
    assert set(args) == set(VALID_SOURCES)


def test_models_use_builtin_list_syntax():
    """list[...] (PEP 585) should be used in touched models — no typing.List."""
    from backend import models

    # Read the source file and check the typing import line excludes List.
    source_text = Path(models.__file__).read_text()
    assert "from typing import" in source_text
    typing_imports = [
        line for line in source_text.splitlines()
        if line.strip().startswith("from typing import")
    ]
    for line in typing_imports:
        # Allow Optional, Literal; forbid List.
        assert "List" not in line.replace("Literal", ""), (
            f"models.py still imports typing.List: {line!r}"
        )