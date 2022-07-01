import argparse
import logging
import os
import sys

import numpy as np

import pygama
from pygama.dsp import build_dsp
from pygama.raw import build_raw


def pygama_cli():
    parser = argparse.ArgumentParser(
        prog='pygama',
        description="pygama's command-line interface")

    # global options
    parser.add_argument('--version', action='store_true',
                        help="""Print pygama version and exit""")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="""Increase the program verbosity""")

    subparsers = parser.add_subparsers()

    # build_raw interface
    parser_d2r = subparsers.add_parser(
        'build-raw', description="""Convert data into LEGEND HDF5 (LH5) raw format""")
    parser_d2r.add_argument('in_stream', nargs='+',
                            help="""Input stream. Can be a single file, a list
                            of files or any other input type supported by the
                            selected streamer""")
    parser_d2r.add_argument('--stream-type', '-t',
                            help="""Input stream type name. Use this if the
                            stream type cannot be automatically deduced by
                            pygama""")
    parser_d2r.add_argument('--out-spec', '-o',
                            help="""Specification for the output stream. HDF5
                            or JSON file name""")
    parser_d2r.add_argument('--buffer_size', '-b', type=int, default=8192,
                            help="""Set buffer size""")
    parser_d2r.add_argument('--max-rows', '-n', type=int, default=np.inf,
                            help="""Maximum number of rows of data to process
                            from the input file""")
    parser_d2r.add_argument('--overwrite', '-w', action='store_true',
                            help="""Overwrite output files""")

    parser_d2r.set_defaults(func=build_raw_cli)

    # TODO: build_dsp interface
    parser_r2d = subparsers.add_parser(
        'build-dsp', description="""Process LH5 raw files and produce a
        dsp file using a JSON configuration""")

    parser_r2d.set_defaults(func=build_dsp_cli)

    if len(sys.argv) < 2:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(name)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(name)s [%(levelname)s] %(message)s")

    if args.version:
        print(pygama.__version__)
        sys.exit()

    args.func(args)


def build_raw_cli(args):
    for stream in args.in_stream:
        basename = os.path.splitext(os.path.basename(stream))[0]
        build_raw(stream, in_stream_type=args.stream_type,
                  out_spec=args.out_spec, buffer_size=args.buffer_size,
                  n_max=args.max_rows, overwrite=args.overwrite,
                  orig_basename=basename)


def build_dsp_cli(args):
    for file in args.files:
        build_dsp(file, args.output, args.jsonconfig, lh5_tables=args.group,
                  database=args.dbfile, verbose=args.verbose,
                  outputs=args.outpar, n_max=args.nevents,
                  write_mode=args.writemode, buffer_len=args.chunk,
                  block_width=args.block)
