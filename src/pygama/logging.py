"""This module implements some helpers for setting up logging."""

import logging

import colorlog

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
FATAL = logging.FATAL
CRITICAL = logging.CRITICAL


def setup(level: int = logging.INFO, logger: logging.Logger = None) -> None:
    """Setup a colorful logging output.

    If `logger` is None, sets up only the ``pygama`` logger.

    Parameters
    ----------
    level
        logging level (see :mod:`logging` module).
    logger
        if not `None`, setup this logger.

    Examples
    --------
    >>> from pygama import logging
    >>> logging.setup(level=logging.DEBUG)
    """
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(name)s [%(levelname)s] %(message)s")
    )

    if logger is None:
        logger = colorlog.getLogger("pygama")

    logger.setLevel(level)
    logger.addHandler(handler)
