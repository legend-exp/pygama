#!/usr/bin/env python3

import sys, os, re, argparse, json, numpy
from collections import OrderedDict
from pygama.io.raw_to_dsp import raw_to_dsp
from pygama.io.fcdaq import *


def main():
    doc="""
    Pygma Data Processing Utility
    """
    # options
    parser = argparse.ArgumentParser(description='Pygama Data Processing Utility')
    parser.add_argument('-i', '--input-files', help='name (regex) of the input file(s)', required=True)
    parser.add_argument('-o', '--output-dir',  help='output directory', required=True)
    parser.add_argument('-c', '--config-file', help='path/name of the config file', required=False, default="")
    parser.add_argument('-s', '--step',        help='data production step (e.g. raw_to_dsp)', required=True)

    parser.add_argument('-O', '--overwrite',   help='overwrite output files', action="store_true")
    parser.add_argument('-v', '--verbose',     help='increase output verbosity', action="store_true")
    parser.add_argument('-m', '--max-ev-num',  help='maximum number of events to process', type=int, default=np.inf)

    args = parser.parse_args()

    # dump info
    if args.verbose:
        print('Pygama Data Processing Utility')
        print('  Running step      ', args.step)
        print('  Input  files:     ', args.input_files)
        print('  Output directory: ', args.output_dir)
        print('  Config file:      ', args.config_file)
        print('  Max ev number:    ', args.max_ev_num)

    # check dsp config file
    f_config = args.config_file
    if not os.path.exists(f_config):
        print('  Error: config file ' + f_config + ' does not exist')
        exit()

    # input file
    input_str = args.input_files
    re_input = re.compile(os.path.basename(input_str))
    dir_input = os.path.dirname(input_str)

    f_input = []
    f_datetimes = []
    with os.scandir(dir_input) as dirs:
        for entry in dirs:
            match = re.search(re_input, entry.name)
            if match is not None:
                f_input.append(os.path.join(dir_input, match.string))
                match = re.search(r'.*(\d{8})-(\d{6}).*\.fcio', entry.name)
                if match is not None:
                    f_datetimes.append(match.groups()[0] + 'T' + match.groups()[1] + 'Z')
                    continue

                match = re.search(r'.*(\d{8})T(\d{6})Z.*\.lh5', entry.name)
                if match is not None:
                    f_datetimes.append(match.groups()[0] + 'T' + match.groups()[1] + 'Z')
                    continue

                print('  Error: cannot extract datetime from file ' + entry.name)
                exit()

    if not f_input:
        print('  Error: empty input file list')
        exit()

    os.makedirs(args.output_dir, exist_ok=True)

    # do the processing
    if args.step == 'daq_to_raw':

        with open(f_config) as f:
            f_config = json.load(f)

        ch_groups = f_config['ch_groups']
        template = f_config['output_template']

        f_output = [os.path.join(args.output_dir, template.replace('{timestamp}', s)) for s in f_datetimes]
        for i in range(len(f_output)):
            if not args.overwrite and os.path.isfile(f_output[i]):
                print('  Warning: file ' + f_output[i] + ' already exists, skipping')
                continue
            elif args.overwrite and os.path.isfile(f_output[i]):
                os.remove(f_output[i])

            process_flashcam(f_input[i], f_output[i], ch_groups_dict=ch_groups,
                    n_max=args.max_ev_num, verbose=args.verbose)

    elif args.step == 'raw_to_dsp':

        with open(f_config) as f:
            f_config = json.load(f, object_pairs_hook=OrderedDict)

        config_dict = f_config['dsp_config']
        template = f_config['output_template']

        f_output = [os.path.join(args.output_dir, template.replace('{timestamp}', s)) for s in f_datetimes]
        for i in range(len(f_output)):
            raw_to_dsp(f_input[i], f_output[i], config_dict, lh5_tables=['spms/raw'], n_max=args.max_ev_num,
                    verbose=args.verbose, overwrite=args.overwrite)

    else: print('  Error: data produciton step not known');

if __name__=="__main__":
    main()
