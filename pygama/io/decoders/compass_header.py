#!/usr/bin/env python3
import glob
import json
import xml.etree.ElementTree as ET
from pprint import pprint

def main():
    """
    parse CAEN's Compass XML header files.
    python xml package docs: https://docs.python.org/3/library/xml.html
    """
    global data_dir
    data_dir = "/Users/wisecg/Data/pt100/DAQ"

    # print_file()
    compare_settings()


def print_file():
    """
    print one file
    """
    f = "{}/run_2/settings.xml".format(data_dir)
    print_settings(f)


def print_settings(f, verbose=True):
    """
    creates an ElementTree, and calls the parser function
    """
    tree = ET.ElementTree(file=f)

    # run the parser
    settings_list = []
    parse_header(tree.getroot(), "", settings_list)

    # fill a dict w/ settings
    settings = {}
    ctr = -1
    while True:
        ctr += 1
        if ctr >= len(settings_list):
            break
        key = settings_list[ctr][0]
        val = settings_list[ctr][1]

        # format keys
        if "parameters" in key:
            settings[val] = settings_list[ctr+1][1]
            ctr += 1
        elif "Memento" in key:
            settings[key.replace("Memento","")] = val
        else:
            settings[key] = val

    # convert vals to python types
    for key, val in settings.items():
        try:
            settings[key] = json.loads(val)
        except:
            pass

    # pretty print the settings
    if verbose:
        pprint(settings)

    return settings


def parse_header(root, key, settings_list, verbose=False):
    """
    recursively parses a row of the XML ElementTree, also saves stuff to a dict
    https://stackoverflow.com/questions/28194703/recursive-xml-parsing-python-using-elementtree
    """
    tag, entry = root.tag, root.text
    if entry is None:
        return

    key = key.replace("/configuration","")
    key = "{}/{}".format(key, tag)
    entry = entry.replace(" ","")

    # print everything
    # print("KEY: {:<60}   TAG: {:<20}   ENTRY: {}".format(key, tag, entry))

    # only print a selection
    if "descriptor" not in key.lower() and "\n" not in entry:
        if verbose:
            print("{:<40}   {}".format(key, entry))

        # save stuff
        settings_list.append([key, entry])

    for elem in list(root):
        parse_header(elem, key, settings_list, verbose)


def compare_settings():
    """
    compare settings in multiple runs
    """
    f_list = sorted(glob.iglob("{}/**/settings.xml".format(data_dir)))
    s_dict = {}
    for f in f_list:
        # pull out the run number to use as the key
        for tmp in f.split("/"):
            if "run" in tmp:
                key = tmp
        s_dict[key] = print_settings(f, False)

    # print(s_dict.keys())
    # ['run_1', 'run_2', 'run_3', 'run_4', 'run_5']

    # comparing two dicts
    # dict_1.keys() & dict_2.keys() # set intersection (keys in common)
    # dict_1.keys() - dict_2.keys() # set difference (keys not in common)
    # dict_1.items() & dict_2.items() # key/value pairs in common
    # dict_1.items() - dict_2.items() # different key/value pairs

    diffs = s_dict['run_1'].items() - s_dict['run_4'].items() # returns a set

    for k, v in diffs:
        print(k, v)

    # print(diffs)


if __name__=="__main__":
    main()