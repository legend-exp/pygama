"""
This sphinx extension aims to improve the documentation of numba-decorated
functions. It it inspired by the design of Celery's sphinx extension
'celery.contrib.sphinx'.

Adapted from https://github.com/numba/numba/issues/5755#issuecomment-646587651
"""

from copy import copy
from typing import Iterator, List

from docutils import nodes
from numba.core.dispatcher import Dispatcher
from numba.np.ufunc.dufunc import DUFunc
from numba.np.ufunc.gufunc import GUFunc
from sphinx.domains.python import PyFunction
from sphinx.ext.autodoc import FunctionDocumenter


class NumbaFunctionDocumenter(FunctionDocumenter):
    """Document numba decorated functions."""

    def import_object(self) -> bool:
        """Import the object given by *self.modname* and *self.objpath* and set
        it as *self.object*.

        Returns True if successful, False if an error occurred.
        """
        success = super().import_object()
        if success:
            # Store away numba wrapper
            self.jitobj = self.object
            # And bend references to underlying python function
            if hasattr(self.object, "py_func"):
                self.object = self.object.py_func
            elif hasattr(self.object, "_dispatcher") and hasattr(
                self.object._dispatcher, "py_func"
            ):
                self.object = self.object._dispatcher.py_func
            elif hasattr(self.object, "gufunc_builder") and hasattr(
                self.object.gufunc_builder, "py_func"
            ):
                self.object = self.object.gufunc_builder.py_func
            else:
                success = False
        return success

    def process_doc(self, docstrings: List[List[str]]) -> Iterator[str]:
        """Let the user process the docstrings before adding them."""
        # Essentially copied from FunctionDocumenter
        for docstringlines in docstrings:
            if self.env.app:
                # let extensions preprocess docstrings
                # need to manually set 'what' to FunctionDocumenter.objtype
                # to not confuse preprocessors like napoleon with an objtype
                # that they don't know
                self.env.app.emit(
                    "autodoc-process-docstring",
                    FunctionDocumenter.objtype,
                    self.fullname,
                    self.object,
                    self.options,
                    docstringlines,
                )

            extra_lines = []

            # This block inserts information about precompiled signatures
            if getattr(self.jitobj, "types", []) and getattr(
                self.jitobj, "_frozen", False
            ):
                line = "``" + "``, ``".join(self.jitobj.types) + "``"
                extra_lines.insert(0, "- *Precompiled signatures:* " + line)

            if hasattr(self.jitobj, "gufunc_builder"):
                if hasattr(self.jitobj.gufunc_builder, "targetoptions"):
                    opts = copy(self.jitobj.gufunc_builder.targetoptions)
                else:
                    opts = {}

                if hasattr(self.jitobj.gufunc_builder, "cache"):
                    opts["cache"] = self.jitobj.gufunc_builder.cache

                line = []
                for k, v in opts.items():
                    line.append(f"{k}={v}")

                if line:
                    line = "``" + "``, ``".join(line) + "``"
                    extra_lines.insert(0, "- *Options:* " + line)

            if extra_lines:
                docstringlines = extra_lines + [""] + docstringlines

            yield from docstringlines


class JitDocumenter(NumbaFunctionDocumenter):
    """Document jit/njit decorated functions."""

    objtype = "jitfun"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, Dispatcher) and hasattr(member, "py_func")


class VectorizeDocumenter(NumbaFunctionDocumenter):
    """Document vectorize decorated functions."""

    objtype = "vecfun"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return (
            isinstance(member, DUFunc)
            and hasattr(member, "_dispatcher")
            and hasattr(member._dispatcher, "py_func")
        )


class GUVectorizeDocumenter(NumbaFunctionDocumenter):
    """Document guvectorize decorated functions."""

    objtype = "guvecfun"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return (
            isinstance(member, GUFunc)
            and hasattr(member, "gufunc_builder")
            and hasattr(member.gufunc_builder, "py_func")
        )


class JitDirective(PyFunction):
    """Sphinx jitfun directive."""

    def get_signature_prefix(self, sig):
        return [nodes.Text("@numba.jit ")]


class VectorizeDirective(PyFunction):
    """Sphinx vecfun directive."""

    def get_signature_prefix(self, sig):
        return [nodes.Text("@numba.vectorize ")]


class GUVectorizeDirective(PyFunction):
    """Sphinx guvecfun directive."""

    def get_signature_prefix(self, sig):
        return [nodes.Text("@numba.guvectorize ")]


def setup(app):
    """Setup Sphinx extension."""
    # Register the new documenters and directives (autojitfun, autovecfun)
    # Set the default prefix which is printed in front of the function signature
    app.setup_extension("sphinx.ext.autodoc")
    app.add_autodocumenter(JitDocumenter)
    app.add_directive_to_domain("py", "jitfun", JitDirective)
    app.add_autodocumenter(VectorizeDocumenter)
    app.add_directive_to_domain("py", "vecfun", VectorizeDirective)
    app.add_autodocumenter(GUVectorizeDocumenter)
    app.add_directive_to_domain("py", "guvecfun", GUVectorizeDirective)

    return {"parallel_read_safe": True}
