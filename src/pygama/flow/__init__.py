# TODO: remove these imports, query functions from __all__, and __getattr__ when
# we no longer need to deprecate...
import importlib
import warnings

from .data_loader import DataLoader
from .file_db import FileDB

__all__ = [
    "DataLoader",
    "FileDB",
    "query_runs",
    "query_meta",
    "query_data",
    "query_hist",
    "query_evt",
    "build_iterator",
    "list_run_fields",
]

def __getattr__(name):
    if name in ["query_runs", "list_run_fields", "query_meta", "build_iterator", "query_data", "query_evt", "query_hist"]:
        warnings.warn(
            f"Importing '{name}' from 'flow' is deprecated; use 'datatools' instead!",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(f"pygama.datatools.{name}"), name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
