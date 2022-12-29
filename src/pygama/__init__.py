"""
Pygama: decoding and processing digitizer data.
Check out the `online documentation <https://pygama.readthedocs.io>`_
"""

from ._version import version as __version__
from .lgdo import lh5

__all__ = ["__version__", "lh5"]
