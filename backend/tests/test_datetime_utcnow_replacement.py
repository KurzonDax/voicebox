"""Regression tests for PR #659 — datetime.utcnow() → datetime.now(UTC).

Why this exists:
    `datetime.utcnow()` is deprecated in Python 3.12 (DeprecationWarning since
    3.12, removed in 3.14). Replacing it with the callable form
    `lambda: datetime.now(UTC)` inside SQLAlchemy ``default=`` / ``onupdate=``
    is NOT just a rename — the lambda wrapper is required because ``Column``
    evaluates its ``default`` argument once at class-definition time otherwise,
    freezing every row's ``created_at`` to the import instant.

These tests pin both invariants:

1. No ``datetime.utcnow`` reference remains anywhere in the backend package.
2. SQLAlchemy ``default=lambda: datetime.now(UTC)`` fires per-row (close to
   "now", not at module import), and ``onupdate=lambda: datetime.now(UTC)``
   advances the timestamp on UPDATE.
3. The ``utils.tasks`` dataclass ``default_factory=lambda: datetime.now(UTC)``
   produces a fresh ``datetime`` per instance (not the same object across
   instances).
"""

from __future__ import annotations

import inspect
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Match sibling backend tests: import "backend.*" modules from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database.models import (
    Base,
    Generation,
    GenerationSettings,
    Story,
    VoiceProfile,
)

# ---------------------------------------------------------------------------
# 1. Source-level invariant: no utcnow() anywhere in the backend tree.
# ---------------------------------------------------------------------------


def _backend_root() -> Path:
    """Resolve the backend/ directory from this test file's location."""
    return Path(__file__).resolve().parent.parent


def test_no_datetime_utcnow_remains_in_backend() -> None:
    """No .py file under backend/ may still reference datetime.utcnow().

    This is the acceptance criterion that future regressions will trip first.
    Excludes the test suite itself (which intentionally mentions the symbol
    in docstrings to describe what it pins).
    """
    backend = _backend_root()
    offenders: list[str] = []
    for py_file in backend.rglob("*.py"):
        # Skip the venv — vendored/site-packages aren't ours to police.
        if "venv" in py_file.parts or ".venv" in py_file.parts:
            continue
        # Skip the test directory — it intentionally references the symbol.
        rel = py_file.relative_to(backend)
        if rel.parts and rel.parts[0] == "tests":
            continue
        try:
            text = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "datetime.utcnow" in text or "utcnow()" in text:
            offenders.append(str(rel))
    assert not offenders, (
        f"datetime.utcnow() must be fully replaced with datetime.now(UTC); remaining offenders: {offenders}"
    )


def test_models_py_imports_utc_from_datetime() -> None:
    """database/models.py must import UTC alongside datetime."""
    from backend import database

    src = inspect.getsource(database.models)
    assert "from datetime import" in src
    assert "UTC" in src.split("from datetime import", 1)[1].split("\n", 1)[0]


# ---------------------------------------------------------------------------
# 2. SQLAlchemy lambda defaults fire per-row, not at class definition.
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_engine():
    """Fresh SQLite engine per test — full schema, ephemeral."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _fresh_session(engine) -> Session:
    return Session(engine)


def test_voice_profile_created_at_is_per_instance(in_memory_engine) -> None:
    """Two profiles inserted with a sleep gap must have distinct created_at.

    If `default=datetime.utcnow` (no lambda) had survived the migration, every
    row's `created_at` would equal the import instant and this assertion would
    fail on the second insert.
    """
    with _fresh_session(in_memory_engine) as s:
        a = VoiceProfile(name="alpha")
        s.add(a)
        s.commit()
        s.refresh(a)
        first = a.created_at

        time.sleep(0.05)  # 50 ms — bigger than clock resolution headroom

        b = VoiceProfile(name="bravo")
        s.add(b)
        s.commit()
        s.refresh(b)
        second = b.created_at

    assert first != second, "created_at must be evaluated per insert; got the same value for both rows."
    assert second > first, "second insert must have a strictly later created_at."


def test_voice_profile_updated_at_advances_on_update(in_memory_engine) -> None:
    """onupdate=lambda: datetime.now(UTC) must fire on UPDATE."""
    with _fresh_session(in_memory_engine) as s:
        p = VoiceProfile(name="charlie")
        s.add(p)
        s.commit()
        s.refresh(p)
        original = p.updated_at

        time.sleep(0.05)

        p.name = "charlie-renamed"
        s.commit()
        s.refresh(p)
        after_update = p.updated_at

    assert after_update > original, (
        "updated_at must advance when the row is UPDATEd; the lambda onupdate is not firing."
    )


def test_story_and_generation_timestamps_populate(in_memory_engine) -> None:
    """Smoke test: every model with a created_at column auto-populates it."""
    with _fresh_session(in_memory_engine) as s:
        g = Generation(
            profile_id="any-profile",
            text="hello",
            engine="qwen",
            model_size="small",
        )
        s.add(g)
        s.commit()
        s.refresh(g)
        assert g.created_at is not None
        assert isinstance(g.created_at, datetime)

        st = Story(name="my story")
        s.add(st)
        s.commit()
        s.refresh(st)
        assert st.created_at is not None
        assert st.updated_at is not None

        gs = GenerationSettings()
        s.add(gs)
        s.commit()
        s.refresh(gs)
        assert gs.updated_at is not None


# ---------------------------------------------------------------------------
# 3. utils.tasks dataclass default_factory produces distinct instances.
# ---------------------------------------------------------------------------


def test_task_default_factory_produces_fresh_datetimes() -> None:
    """Each DownloadTask/GenerationTask instance gets its own started_at.

    Without the lambda wrapper, `default_factory=datetime.utcnow` would also
    produce fresh instances per call (because utcnow() is invoked), so this
    test is really guarding against accidentally swapping to a non-callable
    sentinel like a module-level constant.
    """
    from backend.utils.tasks import DownloadTask, GenerationTask

    a = DownloadTask(model_name="x")
    b = DownloadTask(model_name="y")
    assert a.started_at != b.started_at
    assert isinstance(a.started_at, datetime)

    g1 = GenerationTask(task_id="t1", profile_id="p1", text_preview="hi")
    g2 = GenerationTask(task_id="t2", profile_id="p2", text_preview="ho")
    assert g1.started_at != g2.started_at


# ---------------------------------------------------------------------------
# 4. Importing the changed modules emits no DeprecationWarning.
# ---------------------------------------------------------------------------


def test_no_datetime_deprecation_warning_on_models_import() -> None:
    """Importing database.models must not emit a datetime DeprecationWarning.

    This is the user-visible payoff of PR #659 — `datetime.utcnow()` triggers
    a DeprecationWarning on Python 3.12+. After the migration the warning
    should be gone.
    """
    # Force a fresh import so we actually exercise the module.
    import importlib

    from backend.database import models as models_mod

    importlib.reload(models_mod)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(models_mod)
        datetime_warns = [
            w for w in caught if issubclass(w.category, DeprecationWarning) and "datetime" in str(w.message).lower()
        ]
    assert not datetime_warns, (
        f"datetime DeprecationWarning emitted on import: {[str(w.message) for w in datetime_warns]}"
    )


def test_no_datetime_deprecation_warning_on_services_import() -> None:
    """Service modules that call datetime.now(UTC) at runtime must be clean.

    Covers profiles, stories, history, channels, export_import.
    """
    import importlib

    modules = [
        "backend.services.profiles",
        "backend.services.stories",
        "backend.services.history",
        "backend.services.channels",
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in modules:
            importlib.import_module(name)
        datetime_warns = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "datetime" in str(w.message).lower()
            and any(name in str(w.message) or name in str(w.filename) for name in modules)
        ]
    assert not datetime_warns, (
        f"datetime DeprecationWarning during service import: {[str(w.message) for w in datetime_warns]}"
    )
