"""
Force torch._torch_docs to be bundled with its .py source file alongside
the .pyc bytecode.

The runtime hook in backend/pyi_rth_torch_docs_patch.py patches this module's
source at load time to wrap add_docstr with a TypeError-tolerant version
(on Python 3.12 some C-level function types can't accept __doc__ assignment,
causing `TypeError: don't know how to add docstring to type 'function'`).
That patch reads the source via loader.get_source(), which only works if the
.py file was actually collected into the bundle.
"""

module_collection_mode = "pyz+py"
