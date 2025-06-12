"""
pygama's command line interface utilities.
"""

import argparse
import logging
import os
import sys

import pygama
import pygama.logging
from pygama.hit import build_hit


def pygama_cli():
    """pygama's command line interface.

    Defines the command line interface (CLI) of the package, which exposes some
    of the most used functions to the console.  This function is added to the
    ``entry_points.console_scripts`` list and defines the ``pygama`` executable
    (see ``setuptools``' documentation). To learn more about the CLI, have a
    look at the help section:

    .. code-block:: console

      $ pygama --help
      $ pygama build-hit --help  # help section for a specific sub-command
    """

    parser = argparse.ArgumentParser(
        prog="pygama", description="pygama's command-line interface"
    )

    # global options
    parser.add_argument(
        "--version", action="store_true", help="""Print pygama version and exit"""
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""Increase the program verbosity""",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="""Increase the program verbosity to maximum""",
    )

    subparsers = parser.add_subparsers()

    add_build_hit_parser(subparsers)

    if len(sys.argv) < 2:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.verbose:
        pygama.logging.setup(logging.DEBUG)
    elif args.debug:
        pygama.logging.setup(logging.DEBUG, logging.root)
    else:
        pygama.logging.setup()

    if args.version:
        print(pygama.__version__)  # noqa: T201
        sys.exit()

    args.func(args)


def add_build_hit_parser(subparsers):
    """Configure :func:`.hit.build_hit.build_hit` command line interface"""

    parser_r2d = subparsers.add_parser(
        "build-hit",
        description="""Process LH5 dsp files and produce a hit file using a
                       JSON configuration""",
    )
    parser_r2d.add_argument(
        "dsp_lh5_file",
        nargs="+",
        help="""Input dsp LH5 file. Can be a single file or a list of them""",
    )
    parser_r2d.add_argument(
        "--config",
        "-c",
        required=True,
        help=""""JSON file holding configuration of column operations on DSP output""",
    )
    parser_r2d.add_argument(
        "--hdf5-groups",
        "-g",
        nargs="*",
        default=None,
        help="""Name of group in the LH5 file. By default process all base
                groups. Supports wildcards""",
    )
    parser_r2d.add_argument(
        "--output",
        "-o",
        default=None,
        help="""Name of output file, if only one is supplied. By default,
                output to <input-filename>_hit.lh5""",
    )
    parser_r2d.add_argument(
        "--max-rows",
        "-n",
        default=None,
        type=int,
        help="""Number of rows to process. By default do the whole file""",
    )
    parser_r2d.add_argument(
        "--chunk",
        "-k",
        default=3200,
        type=int,
        help="""Number of waveforms to read from disk at a time. Default is
                3200""",
    )

    group = parser_r2d.add_mutually_exclusive_group()
    group.add_argument(
        "--overwrite",
        "-w",
        action="store_const",
        const="of",
        dest="writemode",
        default="w",
        help="""Overwrite file if it already exists. Default option""",
    )
    group.add_argument(
        "--update",
        "-u",
        action="store_const",
        const="u",
        dest="writemode",
        help="""Update existing file with new values""",
    )
    group.add_argument(
        "--append",
        "-a",
        action="store_const",
        const="a",
        dest="writemode",
        help="""Append values to existing file""",
    )

    parser_r2d.set_defaults(func=build_hit_cli)


def build_hit_cli(args):
    """Passes command line arguments to :func:`.hit.build_hit.build_hit`."""

    if len(args.dsp_lh5_file) > 1 and args.output is not None:
        raise NotImplementedError("not possible to set multiple output file names yet")

    out_files = []
    if len(args.dsp_lh5_file) == 1:
        if args.output is None:
            basename = os.path.splitext(os.path.basename(args.dsp_lh5_file[0]))[0]
            basename = basename.removesuffix("_dsp")
            out_files.append(f"{basename}_hit.lh5")
        else:
            out_files.append(args.output)
    else:
        for file in args.dsp_lh5_file:
            basename = os.path.splitext(os.path.basename(file))[0]
            basename = basename.removesuffix("_dsp")
            out_files.append(f"{basename}_hit.lh5")

    for i in range(len(args.dsp_lh5_file)):
        build_hit(
            args.dsp_lh5_file[i],
            args.config,
            outfile=out_files[i],
            lh5_tables=args.hdf5_groups,
            n_max=args.max_rows,
            wo_mode=args.writemode,
            buffer_len=args.chunk,
        )
