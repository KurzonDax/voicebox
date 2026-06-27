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
