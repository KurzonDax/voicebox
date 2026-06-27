"""Regression tests for the PyInstaller torch._torch_docs docstring patch hook."""

import types

import pytest

from backend.pyi_rth_torch_docs_patch import (
    _TARGET_IMPORT_LINE,
    _TARGET_MODULE,
    _patch_source,
    _TorchDocsPatchingFinder,
    _TorchDocsPatchingLoader,
)

# ---------------------------------------------------------------------------
# _patch_source
# ---------------------------------------------------------------------------


SIMPLE_SOURCE = (
    "# mypy: allow-untyped-defs\n"
    '"""Adds docstrings to functions defined in the torch._C module."""\n'
    "\n"
    "import re\n"
    "\n"
    "import torch._C\n"
    "from torch._C import _add_docstr as add_docstr\n"
    "\n"
    "\n"
    "_obj1 = object()\n"
    "_obj2 = object()\n"
    "add_docstr(_obj1, 'doc for obj1')\n"
    "add_docstr(_obj2, 'doc for obj2')\n"
    "RESULT = 'done'\n"
)


def test_patch_source_injects_wrapper_after_import_line():
    """The patched source must contain a wrapper def named add_docstr."""
    patched = _patch_source(SIMPLE_SOURCE)
    assert patched is not SIMPLE_SOURCE
    assert "def add_docstr(obj, docstring" in patched
    # The original import line must still be present (it's prepended, not replaced)
    assert _TARGET_IMPORT_LINE in patched
    # The wrapper must reference the original via _vb_orig_add_docstr
    assert "_vb_orig_add_docstr" in patched


def test_patch_source_preserves_remaining_body():
    """Calls to add_docstr after the wrapper must remain in the source."""
    patched = _patch_source(SIMPLE_SOURCE)
    assert "add_docstr(_obj1, 'doc for obj1')" in patched
    assert "add_docstr(_obj2, 'doc for obj2')" in patched


def test_patch_source_returns_unchanged_when_target_missing():
    """If the target import line isn't found, the source is returned unchanged."""
    source_without_target = "import torch\nprint('hello')\n"
    patched = _patch_source(source_without_target)
    assert patched is source_without_target


def test_patch_source_wrapper_swallows_docstring_typeerror():
    """The injected wrapper must catch TypeError containing 'docstring'."""
    import sys

    patched = _patch_source(SIMPLE_SOURCE)
    # Build fake torch + torch._C modules with _add_docstr that raises TypeError
    fake_torch_c = types.ModuleType("torch._C")

    def _raise_add_docstr(obj, docstring, *args, **kwargs):
        raise TypeError("don't know how to add docstring to type 'function'")

    fake_torch_c._add_docstr = _raise_add_docstr
    fake_torch = types.ModuleType("torch")
    fake_torch._C = fake_torch_c

    # Inject into sys.modules so `import torch._C` resolves to our fake
    saved = {k: sys.modules.get(k) for k in ("torch", "torch._C")}
    sys.modules["torch"] = fake_torch
    sys.modules["torch._C"] = fake_torch_c
    try:
        ns: dict = {}
        exec(compile(patched, "<test>", "exec"), ns)
        # If we got here, the wrapper swallowed the TypeError. Verify add_docstr
        # is the wrapper (not the original _raise_add_docstr).
        assert callable(ns["add_docstr"])
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


def test_patch_source_wrapper_reraises_non_docstring_typeerror():
    """TypeErrors not containing 'docstring' must propagate."""
    import sys

    patched = _patch_source(SIMPLE_SOURCE)
    fake_torch_c = types.ModuleType("torch._C")

    def _raise_other_error(obj, docstring, *args, **kwargs):
        raise TypeError("some unrelated type error")

    fake_torch_c._add_docstr = _raise_other_error
    fake_torch = types.ModuleType("torch")
    fake_torch._C = fake_torch_c

    saved = {k: sys.modules.get(k) for k in ("torch", "torch._C")}
    sys.modules["torch"] = fake_torch
    sys.modules["torch._C"] = fake_torch_c
    try:
        ns: dict = {}
        with pytest.raises(TypeError, match="some unrelated type error"):
            exec(compile(patched, "<test>", "exec"), ns)
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


# ---------------------------------------------------------------------------
# _TorchDocsPatchingFinder
# ---------------------------------------------------------------------------


class _FakeInnerLoader:
    """Minimal loader that yields fixed source and tracks exec calls."""

    def __init__(self, source: str):
        self._source = source
        self.exec_called = False

    def get_source(self, name):
        return self._source

    def create_module(self, spec):
        return None  # use default module creation

    def exec_module(self, module):
        self.exec_called = True


class _FakeInnerFinder:
    """Finder that returns a spec with the given loader for the target module."""

    def __init__(self, loader):
        self._loader = loader

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODULE:
            return None
        from importlib.machinery import ModuleSpec

        return ModuleSpec(fullname, self._loader)


def test_finder_returns_none_for_unrelated_module():
    finder = _TorchDocsPatchingFinder()
    assert finder.find_spec("os") is None
    assert finder.find_spec("torch._C") is None
    assert finder.find_spec("transformers") is None


def test_finder_wraps_loader_for_target_module():
    """The finder must replace the inner loader with _TorchDocsPatchingLoader."""
    inner_loader = _FakeInnerLoader(SIMPLE_SOURCE)
    inner_finder = _FakeInnerFinder(inner_loader)
    finder = _TorchDocsPatchingFinder()

    # Temporarily insert both finders into sys.meta_path
    import sys

    original_meta_path = sys.meta_path[:]
    sys.meta_path.insert(0, finder)
    sys.meta_path.insert(1, inner_finder)
    try:
        spec = finder.find_spec(_TARGET_MODULE)
    finally:
        sys.meta_path[:] = original_meta_path

    assert spec is not None
    assert isinstance(spec.loader, _TorchDocsPatchingLoader)


def test_finder_skips_self_in_meta_path():
    """The finder must skip itself when scanning sys.meta_path."""
    finder = _TorchDocsPatchingFinder()
    # Insert only the finder — no inner finder — so it should find no spec
    import sys

    original_meta_path = sys.meta_path[:]
    sys.meta_path = [finder]
    try:
        spec = finder.find_spec(_TARGET_MODULE)
    finally:
        sys.meta_path[:] = original_meta_path
    assert spec is None


def test_finder_skips_finders_that_raise():
    """If an inner finder raises, the finder must skip it and continue."""

    class _RaisingFinder:
        def find_spec(self, fullname, path=None, target=None):
            raise RuntimeError("inner finder error")

    inner_loader = _FakeInnerLoader(SIMPLE_SOURCE)
    good_finder = _FakeInnerFinder(inner_loader)
    raising_finder = _RaisingFinder()
    finder = _TorchDocsPatchingFinder()

    import sys

    original_meta_path = sys.meta_path[:]
    sys.meta_path.insert(0, finder)
    sys.meta_path.insert(1, raising_finder)
    sys.meta_path.insert(2, good_finder)
    try:
        spec = finder.find_spec(_TARGET_MODULE)
    finally:
        sys.meta_path[:] = original_meta_path

    # The raising finder was skipped, the good finder provided the spec
    assert spec is not None
    assert isinstance(spec.loader, _TorchDocsPatchingLoader)


# ---------------------------------------------------------------------------
# Integration: patched source actually runs without TypeError
# ---------------------------------------------------------------------------


def test_patched_source_execs_without_typeerror_on_unsupported_type():
    """Full integration: exec patched source where add_docstr would crash."""
    import sys

    # Build source that calls add_docstr on a plain Python function
    source = (
        "import torch._C\n"
        "from torch._C import _add_docstr as add_docstr\n"
        "\n"
        "def my_func():\n"
        "    pass\n"
        "\n"
        "add_docstr(my_func, 'This is a docstring')\n"
        "RESULT = 'ok'\n"
    )
    patched = _patch_source(source)

    fake_torch_c = types.ModuleType("torch._C")

    def _crash_add_docstr(obj, docstring, *args, **kwargs):
        raise TypeError("don't know how to add docstring to type 'function'")

    fake_torch_c._add_docstr = _crash_add_docstr
    fake_torch = types.ModuleType("torch")
    fake_torch._C = fake_torch_c

    saved = {k: sys.modules.get(k) for k in ("torch", "torch._C")}
    sys.modules["torch"] = fake_torch
    sys.modules["torch._C"] = fake_torch_c
    try:
        ns: dict = {}
        # Should not raise — the wrapper swallows the TypeError
        exec(compile(patched, "<test>", "exec"), ns)
        assert ns["RESULT"] == "ok"
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


# ---------------------------------------------------------------------------
# _TorchDocsPatchingLoader
# ---------------------------------------------------------------------------


class _SpecReturningInnerLoader(_FakeInnerLoader):
    """Inner loader whose create_module returns a real ModuleSpec so the
    loader's own create_module delegation can be exercised."""

    def __init__(self, source: str):
        super().__init__(source)
        self.create_module_called_with = None
        # .path is an attribute (matches PyInstaller FrozenLoader semantics),
        # not a callable — the loader reads it via getattr(self._inner, "path").
        self.path = "<inner-path>"

    def create_module(self, spec):  # type: ignore[override]
        self.create_module_called_with = spec
        from importlib.machinery import ModuleSpec

        return ModuleSpec(spec.name, self)


def _make_module_with_spec(name: str, submodule_search_locations=None):
    """Build a fresh ModuleType with a real ModuleSpec attached."""
    from importlib.machinery import ModuleSpec

    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None)
    if submodule_search_locations is not None:
        module.__spec__.submodule_search_locations = submodule_search_locations
    return module


def test_loader_delegates_getattr_to_inner():
    """Unknown attribute access on the loader forwards to the inner loader."""
    sentinel_attr = "the_sentinel_attribute"
    inner = _SpecReturningInnerLoader(SIMPLE_SOURCE)
    inner.the_sentinel_attribute = "present"  # type: ignore[attr-defined]
    wrapper = _TorchDocsPatchingLoader(inner)
    assert wrapper.the_sentinel_attribute == "present"  # type: ignore[attr-defined]
    # Also confirm getattr on an arbitrary attribute we just set on inner
    inner.some_other_thing = 42  # type: ignore[attr-defined]
    assert wrapper.some_other_thing == 42  # type: ignore[attr-defined]
    # Use the sentinel so the assertion references the dynamic name
    assert getattr(wrapper, sentinel_attr) == "present"


def test_loader_create_module_delegates_to_inner():
    """create_module must delegate to the inner loader and return its result."""
    inner = _SpecReturningInnerLoader(SIMPLE_SOURCE)
    wrapper = _TorchDocsPatchingLoader(inner)
    from importlib.machinery import ModuleSpec

    spec = ModuleSpec("torch._torch_docs", loader=None)
    result = wrapper.create_module(spec)
    assert isinstance(result, ModuleSpec)
    assert result.name == "torch._torch_docs"
    assert inner.create_module_called_with is spec


def test_loader_exec_module_falls_back_when_no_source():
    """When get_source returns empty/None, exec_module delegates to inner."""
    inner = _FakeInnerLoader("")
    inner.exec_called = False
    wrapper = _TorchDocsPatchingLoader(inner)
    module = _make_module_with_spec("torch._torch_docs")
    wrapper.exec_module(module)
    assert inner.exec_called is True


def test_loader_exec_module_handles_get_source_exception():
    """When get_source raises, exec_module falls back to inner.exec_module."""

    class _BoomLoader(_FakeInnerLoader):
        def get_source(self, name):
            raise OSError("simulated get_source failure")

    inner = _BoomLoader(SIMPLE_SOURCE)
    inner.exec_called = False
    wrapper = _TorchDocsPatchingLoader(inner)
    module = _make_module_with_spec("torch._torch_docs")
    wrapper.exec_module(module)
    assert inner.exec_called is True


def test_loader_exec_module_runs_patched_source():
    """When source is available, exec_module compiles and execs patched source."""
    inner = _SpecReturningInnerLoader(SIMPLE_SOURCE)
    inner.exec_called = False  # should remain False — we exec ourselves, not the inner
    wrapper = _TorchDocsPatchingLoader(inner)
    module = _make_module_with_spec("torch._torch_docs", submodule_search_locations=["/some/pkg/path"])
    wrapper.exec_module(module)
    # The patched module sets add_docstr (our wrapper) and calls it on _obj1.
    assert callable(module.add_docstr)
    # Inner.exec_module should NOT have been called — we did the exec ourselves.
    assert inner.exec_called is False
    # submodule_search_locations must be installed on the module's __path__
    assert module.__path__ == ["/some/pkg/path"]


def test_loader_exec_module_uses_module_name_when_inner_has_no_path():
    """If inner loader has no .path attribute, exec_module falls back to module name."""
    inner = _FakeInnerLoader(SIMPLE_SOURCE)
    # Explicitly delete .path so getattr(..., 'path') raises AttributeError
    if hasattr(inner, "path"):
        delattr(inner, "path")
    wrapper = _TorchDocsPatchingLoader(inner)
    module = _make_module_with_spec("torch._torch_docs")
    # Should not raise — fallback filename = module.__name__
    wrapper.exec_module(module)
    assert callable(module.add_docstr)


# ---------------------------------------------------------------------------
# Finder branches: skip finders without find_spec; skip finders returning None
# ---------------------------------------------------------------------------


class _NoFindSpecFinder:
    """A finder object that doesn't have a find_spec attribute."""

    def __repr__(self):  # for the diagnostic log message
        return "<NoFindSpecFinder>"


class _NoLoaderInnerFinder(_FakeInnerFinder):
    """Inner finder whose spec has loader=None — should be skipped."""

    def __init__(self):
        super().__init__(loader=None)


class _ReturnsNoneInnerFinder:
    """Inner finder that returns None from find_spec — should be skipped."""

    def find_spec(self, fullname, path=None, target=None):
        return None

    def __repr__(self):
        return "<ReturnsNoneInnerFinder>"


def test_finder_skips_finders_without_find_spec():
    """Finders lacking find_spec must be skipped, not crash."""
    good_loader = _FakeInnerLoader(SIMPLE_SOURCE)
    good_finder = _FakeInnerFinder(good_loader)
    no_find_spec = _NoFindSpecFinder()

    finder = _TorchDocsPatchingFinder()
    import sys

    original = sys.meta_path[:]
    sys.meta_path.insert(0, finder)
    sys.meta_path.insert(1, no_find_spec)
    sys.meta_path.insert(2, good_finder)
    try:
        spec = finder.find_spec(_TARGET_MODULE)
    finally:
        sys.meta_path[:] = original
    assert spec is not None
    assert isinstance(spec.loader, _TorchDocsPatchingLoader)


def test_finder_skips_finders_returning_spec_with_no_loader():
    """If inner finder returns a spec whose loader is None, skip and try next."""
    good_loader = _FakeInnerLoader(SIMPLE_SOURCE)
    good_finder = _FakeInnerFinder(good_loader)
    no_loader_finder = _NoLoaderInnerFinder()

    finder = _TorchDocsPatchingFinder()
    import sys

    original = sys.meta_path[:]
    sys.meta_path.insert(0, finder)
    sys.meta_path.insert(1, no_loader_finder)
    sys.meta_path.insert(2, good_finder)
    try:
        spec = finder.find_spec(_TARGET_MODULE)
    finally:
        sys.meta_path[:] = original
    assert spec is not None
    assert isinstance(spec.loader, _TorchDocsPatchingLoader)


def test_finder_skips_finders_returning_none_spec():
    """If inner finder returns None for the spec, skip and try next."""
    good_loader = _FakeInnerLoader(SIMPLE_SOURCE)
    good_finder = _FakeInnerFinder(good_loader)
    none_finder = _ReturnsNoneInnerFinder()

    finder = _TorchDocsPatchingFinder()
    import sys

    original = sys.meta_path[:]
    sys.meta_path.insert(0, finder)
    sys.meta_path.insert(1, none_finder)
    sys.meta_path.insert(2, good_finder)
    try:
        spec = finder.find_spec(_TARGET_MODULE)
    finally:
        sys.meta_path[:] = original
    assert spec is not None
    assert isinstance(spec.loader, _TorchDocsPatchingLoader)


# ---------------------------------------------------------------------------
# _diag robustness — silently swallows file-write failures
# ---------------------------------------------------------------------------


def test_diag_swallows_file_write_errors(tmp_path, monkeypatch):
    """_diag must never raise even if the log path is unwritable."""
    from backend import pyi_rth_torch_docs_patch as _mod

    # Point _DIAG_PATH at a directory so open(... 'a') raises IsADirectoryError
    monkeypatch.setattr(_mod, "_DIAG_PATH", str(tmp_path))  # tmp_path is a dir
    # Must not raise
    _mod._diag("this should be silently dropped")


def test_finder_installation_failure_is_logged_and_swallowed(monkeypatch):
    """If sys.meta_path.insert raises during hook install, the except branch
    must log the failure rather than crash the import.

    Reproduces the defensive try/except around the install line by replacing
    sys.meta_path with an object whose .insert() raises — exec'ing the module
    body against the broken meta_path must not crash.
    """

    class _BoomMetaPathList(list):
        def insert(self, index, obj):
            raise RuntimeError("simulated meta_path.insert failure")

    import sys

    saved = sys.meta_path
    sys.meta_path = _BoomMetaPathList()
    try:
        # Re-exec the hook module body directly (it isn't installed as a
        # package, so importlib.reload can't find its spec). The module is
        # already loaded by the test session, so its global names are in
        # sys.modules; this re-runs the bottom-of-file try/except install
        # block against the broken meta_path.
        from pathlib import Path

        hook_path = Path(__file__).resolve().parent.parent / "pyi_rth_torch_docs_patch.py"
        with open(hook_path, encoding="utf-8") as f:
            source = f.read()
        # Must NOT raise — the except branch swallows the RuntimeError.
        exec(compile(source, str(hook_path), "exec"), sys.modules["backend.pyi_rth_torch_docs_patch"].__dict__)
    finally:
        sys.meta_path = saved

    # If we got here, the except branch successfully swallowed the error.
    assert True
