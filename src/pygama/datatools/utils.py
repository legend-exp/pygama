from __future__ import annotations

import keyword
import os
import re
import string
from collections.abc import Mapping
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from dbetto import Props, TextDB
from rich.console import Console
from rich.status import Status


def get_recursive(db: TextDB, path: str):
    """Helper to recursively access values from nested dict-likes"""
    fields = re.split("[,/]", path)
    ret = db
    for f in fields:
        ret = ret[f]
    return ret


def format_vars(fstring: str):
    """Helper to get list of variables referenced in format string"""
    return [v[1] for v in string.Formatter().parse(fstring) if v[1]]


def parse_query_paths(
    expr: str, fullmatch: bool = False
) -> list[tuple[str, str | None, str]] | tuple[str, str | None, str]:
    """
    Parse input string for variable names of the form::

        [alias][@ or :][par.path]

    and return a list of each matching 3-tuple of the form::

        (full_match, alias, path)

    Aliases and names in paths must be legal python names (i.e. alphanumeric, doesn't
    start with a digit). If ``@`` is used to separate the alias and path, it is left in
    the path (to denote a metadata location); if ``:`` is used, it is omitted.
    Note that function names (i.e. a valid name followed by ``(``) are excluded.
    Values inside of ``[...]``, ``{...}``, ``"..."``, and ``'...'`` are also excluded.

    If fullmatch is ``True``, expect full string to match pattern and return single tuple.
    Otherwise return a list of tuples, for each match found.
    """
    # Note: ast does not like @'s and :'s used in this way, so instead we parse with regex
    if not fullmatch:
        # remove substrings inside of brackets or quotes
        var_list = " ".join(
            re.split(r"(?:\{.*?\})|(?:\[.*?\])|(?:\".*?\")|(?:'.*?')", expr)
        )
        var_list = re.findall(r"[\w:@\.]+(?![\w:@\.(])", var_list)
    else:
        var_list = [expr]

    ret = []
    for var in var_list:
        # skip numerals
        try:
            float(var)
            if fullmatch:
                msg = f"'{var}' is not a valid variable"
                raise NameError(msg)
            continue
        except ValueError:
            pass

        # skip reserved keywords in python
        if keyword.iskeyword(var):
            if fullmatch:
                msg = f"'{var}' is an illegal name"
                raise NameError(msg)
            continue

        match = re.fullmatch(
            r"([a-zA-Z_]\w*)??:?(@?[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)", var
        )
        if match is None:
            msg = f"'{var}' could not be parsed"
            raise NameError(msg)

        if keyword.iskeyword(match.group(1)):
            msg = f"{match.group(1)} is an illegal name"
            raise NameError(msg)
        ret.append((match.group(0), match.group(1), match.group(2)))

    return ret if not fullmatch else ret[0]


def _setup_spinner(
    stack: ExitStack,
    progress: Status | Console | bool,
):
    # Helper to set up spinner bar and enter context
    if isinstance(progress, Status):
        progress.update("Querying runs...")
        # start spinner in context if not already started
        return progress if progress._live.is_started else stack.enter_context(progress)
    if isinstance(progress, Console):
        return stack.enter_context(
            progress.status("Querying runs...", spinner="betaWave")
        )
    if progress:
        return stack.enter_context(Status("Querying runs...", spinner="betaWave"))
    return None


def _setup_executor(
    stack: ExitStack,
    processes: int | None,
    executor: Executor | None,
):
    # Helper to set up concurrent.futures.Executor and enter context
    if processes is None and isinstance(executor, Executor):
        processes = executor._max_workers

    if executor is None and isinstance(processes, int):
        executor = stack.enter_context(ProcessPoolExecutor(processes))

    return processes, executor


def _read_dataflow_config(dataflow_config="$REFPROD/dataflow-config.yaml"):
    # helper to get configs
    if isinstance(dataflow_config, (Path, str)):
        df_config = Props.read_from(
            os.path.expandvars(dataflow_config), subst_pathvar=True
        )
    elif isinstance(dataflow_config, Mapping):
        df_config = dataflow_config
    else:
        msg = "dataflow_config must be a str, Path, or Mapping"
        raise ValueError(msg)
    return df_config, df_config["paths"], df_config.get("query", {})
