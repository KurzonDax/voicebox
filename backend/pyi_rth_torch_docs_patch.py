"""
PyInstaller runtime hook: patch torch._torch_docs add_docstr to tolerate TypeError.

Problem
-------
When the bundled voicebox-server binary runs as a Tauri sidecar on macOS with
Python 3.12, torch._torch_docs.py calls add_docstr (which is torch._C._add_docstr)
hundreds of times to attach docstrings to C-level functions. On Python 3.12 some
C-level function types can't accept __doc__ assignment, causing:

    TypeError: don't know how to add docstring to type 'function'

This crashes the server at startup, preventing the desktop app from connecting
to its backend.

Fix
---
This runtime hook installs a meta-path finder that intercepts the import of
torch._torch_docs, reads its source, injects a TypeError-tolerant wrapper for
add_docstr immediately after the `from torch._C import _add_docstr as add_docstr`
line, and then compiles/execs the patched source into the module namespace.

The wrapper silently swallows TypeError when the error message contains
'docstring' (matching the crash signature), so unrelated TypeErrors are still
surfaced. Docstrings are cosmetic for inference — they don't affect torch's
functional behavior.

Requires the .py source to be bundled alongside the .pyc bytecode — see
backend/pyi_hooks/hook-torch._torch_docs.py (module_collection_mode = 'pyz+py').
"""

import os
import sys
import tempfile

# Diagnostics — log hook activity to a file alongside the bundle so we can
# see what's happening when the server is run as a sidecar (no stdout for
# runtime hook prints). Safe no-op if the file can't be written.
_DIAG_PATH = os.path.join(tempfile.gettempdir(), "voicebox_rt_hook.log")


def _diag(msg: str) -> None:
    try:
        with open(_DIAG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


_HOOK_VERSION = "v1-torch-docs-patch"
_diag(f"=== runtime hook load @ pid={os.getpid()} version={_HOOK_VERSION} ===")


_TARGET_MODULE = "torch._torch_docs"
_TARGET_IMPORT_LINE = "from torch._C import _add_docstr as add_docstr"


def _patch_source(source: str) -> str:
    """Inject a TypeError-tolerant wrapper for add_docstr into torch._torch_docs source.

    Replaces the import line with the import plus a wrapper that catches
    TypeError containing 'docstring' and silently swallows it. All other
    TypeErrors propagate normally.

    Returns the source unchanged if the target line isn't found (e.g. torch
    version has changed the import structure).
    """
    if _TARGET_IMPORT_LINE not in source:
        _diag(f"[torch-docs-patch] target line not found in source (len={len(source)})")
        return source

    wrapper_code = (
        "\n"
        "\n"
        "import torch._C as _vb_torch_C  # re-import for the wrapper closure\n"
        "_vb_orig_add_docstr = _vb_torch_C._add_docstr\n"
        "\n"
        "def add_docstr(obj, docstring, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003\n"
        "    \"\"\"Wrapper around torch._C._add_docstr that tolerates TypeError on C-level functions.\n"
        "\n"
        "    On Python 3.12 some C-level function types reject __doc__ assignment,\n"
        "    raising TypeError: don't know how to add docstring to type 'function'.\n"
        "    Docstrings are cosmetic for inference, so we silently skip them.\n"
        "    \"\"\"\n"
        "    try:\n"
        "        return _vb_orig_add_docstr(obj, docstring, *args, **kwargs)\n"
        "    except TypeError as _e:\n"
        "        if 'docstring' in str(_e):\n"
        "            return  # cosmetic failure — docstrings not needed for inference\n"
        "        raise\n"
        "\n"
    )

    patched = source.replace(_TARGET_IMPORT_LINE, _TARGET_IMPORT_LINE + wrapper_code, 1)
    _diag(f"[torch-docs-patch] source patched: len {len(source)} -> {len(patched)}")
    return patched


class _TorchDocsPatchingLoader:
    """Delegate loader that reads source, patches add_docstr, and execs the result.

    Forwards all attribute access (get_source, get_filename, is_package, etc.)
    to the inner PyInstaller loader via __getattr__, only overriding
    create_module and exec_module.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        source = None
        try:
            source = self._inner.get_source(module.__name__)
        except Exception as e:
            _diag(f"[torch-docs-patch] get_source({module.__name__}) failed: {e!r}")

        if not source:
            _diag(
                f"[torch-docs-patch] no source for {module.__name__}; "
                "falling back to inner exec_module (patch NOT applied)"
            )
            self._inner.exec_module(module)
            return

        patched = _patch_source(source)
        _diag(
            f"[torch-docs-patch] {module.__name__}: "
            f"patched={patched is not source}, len={len(patched)}"
        )
        spec = module.__spec__
        if spec is not None and spec.submodule_search_locations is not None:
            module.__path__ = spec.submodule_search_locations
        filename = getattr(self._inner, "path", module.__name__)
        exec(compile(patched, filename, "exec"), module.__dict__)
        _diag(f"[torch-docs-patch] {module.__name__} OK")


class _TorchDocsPatchingFinder:
    """Meta-path finder that intercepts torch._torch_docs and wraps its loader.

    Inserts at sys.meta_path position 0 so we run before PyInstaller's
    FrozenImporter. We delegate to other finders to locate the real spec,
    then swap the loader for our patching wrapper.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODULE:
            return None
        _diag(f"[torch-docs-patch] match: {fullname}, path={path!r}")
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                real_spec = find(fullname, path, target)
            except Exception as e:
                _diag(f"[torch-docs-patch] inner finder {type(finder).__name__} raised: {e}")
                continue
            if real_spec is None or real_spec.loader is None:
                continue
            _diag(
                f"[torch-docs-patch] wrapped loader from "
                f"{type(finder).__name__} -> {type(real_spec.loader).__name__}"
            )
            real_spec.loader = _TorchDocsPatchingLoader(real_spec.loader)
            return real_spec
        _diag("[torch-docs-patch] NO inner finder returned a spec")
        return None


try:
    sys.meta_path.insert(0, _TorchDocsPatchingFinder())
    _diag("installed finder: _TorchDocsPatchingFinder")
except Exception as _e:
    _diag(f"FAILED to install _TorchDocsPatchingFinder: {_e!r}")

_diag(
    "final sys.meta_path head: "
    + ", ".join(type(f).__name__ for f in sys.meta_path[:6])
)
