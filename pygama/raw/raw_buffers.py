"""
raw_buffers.py: manages data buffering for raw data conversion

Manages LGDO buffers and their corresponding output streams. Allows for
one-to-many mapping of input sreams to output streams.
"""

import os
from pygama import lgdo

class RawBuffer:
    '''
    A "raw_buffer" is in essence a an lgdo object (typically a Table) to which
    detecoded data will be written, along with some meta-data defining what data
    goes into the lgdo, and where the lgdo gets written out

    Attributes
    ----------
    key_list : list
        a list of keys (e.g. channel numbers) identifying data to be written
        into this buffer. The key scheme is specific to the decoder with which
        the raw_group is associated. This is called "key_list" instead of "keys"
        to avoid confusion with the dict function "keys()", e.g.
        raw_group.keys()
    file_path : str (optional)
        the name of the output file path to which the raw_buffer's lgdo should
        be written
    group_path : str (optional)
        the name of the hdf5 group to with the raw_buffer's lgdo should be written
    lgdo : lgdo 
        the lgdo used as the actual buffer. Typically a table. Set to None upon
        creation so that the user can initialize it later.
    '''

    def __init__(self):
        key_list = []
        file_path = ''
        group_path = ''
        lgdo = None


"""
raw_groups 
"raw_groups" is a dictionary of named raw_group's. Typically, the names are the
same as group_path in each raw_group

raw_group_library
Different data decoders can use different key schemes, so the raw_groups must be
specified in a raw_group_library, which is a dictionary whose keys are the names
of data decoders, and whose corresponding values are the raw_groups for the
specified decoder.

Example json yielding a valid raw_group_library (see expand_raw_groups() for
documentation on allowed shorthand notations):

{
"FlashCamEventDecoder" : {
  "g{key:0>3d}" : {
    "key_list" : [ [24,64] ],
    "out_path" : "$DATADIR/{file_key}_geds:geds"
  },
  "spms" : {
    "key_list" : [ [6,23] ],
    "out_path" : "$DATADIR/{file_key}_spms:spms"
  },
  "puls" : {
    "key_list" : [ 0 ],
    "out_path" : "$DATADIR/{file_key}_auxs:auxs"
  },
  "muvt" : {
    "key_list" : [ 1, 5 ],
    "out_path" : "$DATADIR/{file_key}_auxs:auxs"
  }
}
"""



def expand_raw_groups_library(raw_groups_library, out_path_kwargs):
    for raw_groups in raw_groups_library.values():
        expand_raw_groups(raw_groups, out_path_kwargs)


def expand_raw_groups(raw_groups, out_path_kwargs):
    """
    Expand shorthands in raw_groups

    Allowed shorthands, in order of exapansion:
    * key_list may have entries that are 2-integer lists corresponding to the
      first and last integer keys in a continguous range (e.g. of channels) that
      belong in the group. These simply get replaced with the explicit list of
      integers in the range. We use lists not tuples for json compliance.
    * The raw_group name can include {key:xxx} format specifiers, indicating
      that each key in key_list should be given its own group with the
      corresponding name.  The same specifier can appear in out_path to write
      the key's data to its own output path.
    * You may also include variables in your out_path specification that get
      sent in as kwargs to raw_groups.expand_raw_groups(raw_groups, kwargs).
      These get evaluated simultaneously with the {key:xxx} specifiers.
    * Environment variables can also be used in out_path. They get expanded
      after kwargs are handled and thus can be used inside the kwargs
      themselves.

    Parameters
    ----------
    raw_groups : dict
        A raw_groups dictionary (NOT a raw_groups_library). See module docstring
        for format info. Gets modified in-place
    out_path_kwargs : dict { str : value }
        Variable names and values used to substitue into f-string-format
        specifiers in out_path strings
    """
    # get the original list of groups because we are going to change the
    # dict.keys() of raw_groups inside the next list. Note: we have to convert
    # from dict_keys to list here otherwise the loop complains about changing
    # the dictionary during iteration
    groups = list(raw_groups.keys())
    for group in groups:
        if group == '':
            if len(raw_groups) != 1:
                print("Error: got dummy group (no name) in non-dummy raw_groups")
                return None
            return
        info = raw_groups[group] # changes to info will change raw_groups[group]
        # make sure we have a channel list
        if 'key_list' not in info: 
            print('raw_group', group, 'missing channel specification')
            continue

        # find and expand any ranges in the key_list
        # do in a while loop with a controlled index since we are modifying
        # the length of the list inside the loop (be careful)
        i = 0
        while i < len(info['key_list']):
            key_range = info['key_list'][i]
            # expand any 2-int lists
            if isinstance(key_range, list) and len(key_range) == 2:
                info['key_list'][i:i+1] = range(key_range[0], key_range[1]+1)
                i += key_range[1]-key_range[0]
            i += 1
        
        # Expand groups if group name contains a key-based formatter
        if '{key:' in group: 
            for key in info['key_list']:
                expanded_group = group.format(key=key)
                raw_groups[expanded_group] = info.copy()
                raw_groups[expanded_group]['key_list'] = [ key ];
            raw_groups.pop(group)

    # now re-iterate and exand out_paths
    for group, info in raw_groups.items():
        if 'out_path' not in group: continue
        if len(group['key_list']) == 1: 
            out_path_kwargs['key'] = group['key_list'][0]
        group['out_path'].format(**out_path_kwargs)
        group['out_path'] = os.path.expandvars(group['out_path'])



def get_list_of(dict_key, raw_groups):
    """
    Return a list of values of raw_groups[*][dict_key]

    Parameters
    ----------
    dict_key : str
        a key of each raw_groups[group] dict
    raw_groups: dict
        a raw_groups dict

    Returns
    -------
    values : list
        A list of values of raw_groups[*][dict_key]

    Example
    -------
    output_file_list = get_list_of('out_path', raw_groups)
    """
    values = []
    for ch_info in raw_groups.values():
        if key in ch_info and ch_info[key] not in values: 
            values.append(ch_info[key])
    return values



def build_tables(raw_groups, table_size, init_obj=None):
    """ build tables and associated I/O info for the channel groups.

    Parameters
    ----------
    raw_groups : dict
        raw_groups dict
    table_size : int
        the size to use for each table
    init_obj : obj
        An object with an initialize_lgdo_table(table, key) function

    Returns
    -------
    tables : dict or Table
        A group-key-indexed dictionary of tables for quick look-up, or if passed
        a dummy group (no group name), return the one table made.
    """

    tables = {} 

    # set up a table for each group
    for group_name, group_info in raw_groups.items():

        table = lgdo.Table(table_size)
        if init_obj is not None:
            key = None # for dummy raw_group
            # Note: all ch in key_list will be written to the same table. So it
            # should suffice to initials for first key in the list
            if 'key_list' in group_info: key = group_info['key_list'][0]
            init_obj.initialize_lgdo_table(table, key)

        group_info['table'] = table

        if group_name == '':
            if len(raw_groups) != 1:
                print("Error: got dummy group (no name) in non-dummy raw_groups")
                return None
            return table

        # cache the table to a ch-indexed dict for quick look-up
        for key in group_info['key_list']: tables[ch] = table

    return tables


# can be removed?
'''
def set_outputs(raw_groups, out_file_template=None, grp_path_template='{group_name}'):
    """ Set up output filenames and/or group paths for the channel group

    Parameters
    ----------
    out_file_template : str or dict
        Value(s) for attribute "out_file" when not provided explicitly by the user.
        If a string, use it for the out_file attribute for all ch groups
        If a dict, the out_file must be keyed by the raw_group's "system"
        You can use {system} in the template and the system name will be added.

    group_path_template : str or dict (optional)
        If attribute "group_path" is not provided explicitly by the user,
        raw_groups are stored by default in path {group_name} inside their output
        files. You can use {group_name} as well as {system} and the
        corresponding group info will be filled in for you.  If a dict, the
        grp_path_template is keyed by the raw_group's name Example:
        grp_path_template='/data/{system}/{group_name}/raw'
    """
    for group_name, group_info in raw_groups.items():

        # set the output file name and group path
        format_dict = { 'group_name' : group_name }
        if 'system' in group_info: format_dict['system'] = group_info['system']
        else: format_dict['system'] = ''

        oft = out_file_template
        if isinstance(oft, dict):
            if 'system' not in group_info:
                print('Error, group', group_name, 'needs a "system" to key output filenames')
                continue
            if group_info['system'] not in oft.keys():
                print('Error, no output filename found for system:', group_info['system'])
            else: oft = out_file_template[group_info['system']]
        group_info['out_file'] = oft.format(**format_dict)

        gpt = grp_path_template
        if isinstance(gpt, dict): gpt = group_path_template[group_name]
        group_info['group_path'] = gpt.format(**format_dict)
'''


def create_dummy_raw_groups():
    return { '' : {} }
