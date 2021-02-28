"""
ch_group.py: manages channel grouping for raw data conversion

Often input streams contain multiple types of data that need to get written out
into separate files, or into separate groups within a file. In pygama such data
flow control is provided via ch_groups. 

ch_groups is a dictionary of named channel groupings, each of which is itself a
dictionary with the following fields:
* ch_list (list): a list of channels and channel ranges associated with the
  group. The channel numbering scheme is specific to the decoder with which the
  ch_group is associated.
* out_path (str, optional): the name of the output path to which the ch_group's
  data should be written. Format: "/path/to/file:/in/file/path"

The key behavior is: all channels in the same group get written to the
same table, which then gets written to the specified out_path.
Since each data decoder has its own set of channels, the ch_groups must be
specified in a ch_group_library, which is a dictionary whose keys are the names
of data decoders, and whose corresponding values are the ch_groups for the
specified decoder.

Example json yielding a valid ch_group_library (see expand_ch_groups() for
documentation on allowed shorthand notations):

{
"FlashCamEventDecoder" : {
  "g{ch:0>3d}" : {
    "ch_list" : [ [24,64] ],
    "out_path" : "$DATADIR/{file_key}_geds:geds"
  },
  "spms" : {
    "ch_list" : [ [6,23] ],
    "out_path" : "$DATADIR/{file_key}_spms:spms"
  },
  "puls" : {
    "ch_list" : [ 0 ],
    "out_path" : "$DATADIR/{file_key}_auxs:auxs"
  },
  "muvt" : {
    "ch_list" : [ 1, 5 ],
    "out_path" : "$DATADIR/{file_key}_auxs:auxs"
  }
}
"""

import os
from pygama import lgdo

def expand_ch_groups_library(ch_groups_library, out_path_kwargs):
    for ch_groups in ch_groups_library.values():
        expand_ch_groups(ch_groups, out_path_kwargs)

def expand_ch_groups(ch_groups, out_path_kwargs):
    """
    Expand a ch_group from its shorthands

    Allowed shorthands, in order of exapansion:
    * ch_list may have entries that are 2-integer lists corresponding to the
      first and last channels in a continguous range of channels that belong in
      the group. These simply get replaced with the explicit list of integers in
      the range.
    * The ch_group name can include {ch:xxx} format specifiers, indicating that
      each channel in ch_list should be given its own group with the
      corresponding name.  The same specifier can appear in out_path to write
      the channel's data to its own output path.
    * You may also include variables in your out_path specification that get
      sent in as kwargs to ch_groups.expand_ch_groups(ch_groups, kwargs). These
      get evaluated simultaneously with the {ch:xxx} specifiers.
    * Environment variables can also be used in out_path, they get expanded
      after kwargs are handled and thus can be used inside the kwargs themselves.

    Parameters
    ----------
    ch_groups : dict
        A ch_groups dictionary (NOT a ch_groups_library). See module docstring
        for format info. Gets modified in-place
    out_path_kwargs : dict { str : value }
        Variable names and values used to substitue into f-string-format
        specifiers in out_path strings
    """
    # get the original list of keys because we are going to change the keys
    # of ch_groups inside the next list. Note: we have to convert from
    # dict_keys to list here otherwise the loop complains about changing
    # the dictionary during iteration
    groups = list(ch_groups.keys())
    for group in groups:
        if group == '':
            if len(ch_groups) != 1:
                print("Error: got dummy group (no name) in non-dummy ch_groups")
                return None
            return
        info = ch_groups[group] # changes to info will change ch_groups[group]
        # make sure we have a channel list
        if 'ch_list' not in info: 
            print('ch_group', group, 'missing channel specification')
            continue

        # find and expand any ranges in the ch_list
        # do in a while loop with a controlled index since we are modifying
        # the length of the list inside the loop (be careful)
        i = 0
        while i < len(info['ch_list']):
            ch_range = info['ch_list'][i]
            # expand any 2-int lists
            if isinstance(ch_range, list) and len(ch_range) == 2:
                info['ch_list'][i:i+1] = range(ch_range[0], ch_range[1]+1)
                i += ch_range[1]-ch_range[0]
            # any other entry should be an int!
            elif not isinstance(ch_range, int): 
                print('ch_group', group, 'has malformed channel list:', ch_range)
                print(type(ch_range), len(ch_range))
            i += 1
        
        # Expand groups if group name contains a ch-based formatter
        if '{ch:' in group: 
            for ch in info['ch_list']:
                expanded_group = group.format(ch=ch)
                ch_groups[expanded_group] = info.copy()
                ch_groups[expanded_group]['ch_list'] = [ ch ];
                if '{ch:' in info['system']:
                    expanded_system = info['system'].format(ch=ch)
                    ch_groups[expanded_group]['system'] = expanded_system
            ch_groups.pop(group)

    # now re-iterate and exand out_paths
    for group, info in ch_groups.items():
        if 'out_path' not in group: continue
        if len(group['ch_list']) == 1: 
            out_path_kwargs['ch'] = group['ch_list'][0]
        group['out_path'].format(**out_path_kwargs)
        group['out_path'] = os.path.expandvars(group['out_path'])



def get_list_of(key, ch_groups):
    """
    Return a list of values for key in ch_groups[groups]

    key (str): a key of the ch_groups[groups] dict
    ch_groups (dict): a group-name-indexed dict whose values are
        dictionaries of group information (see expand_ch_groups)

    Example: get_list_of('out_path', ch_groups)
        If ch_groups is the one specified in the json in the expand_ch_groups
        example, this will return the list output files
    """
    values = []
    for ch_info in ch_groups.values():
        if key in ch_info and ch_info[key] not in values: 
            values.append(ch_info[key])
    return values



def build_tables(ch_groups, buffer_size, init_obj=None):
    """ build tables and associated I/O info for the channel groups.

    Parameters
    ----------
    ch_groups : dict
    buffer_size : int
    init_obj : object with initialize_lgdo_table() function

    Returns
    -------
    ch_to_tbls : dict or Table
        A channel-indexed dictionary of tables for quick look-up, or if passed a
        dummy group (no group name), return the one table made.
    """

    ch_to_tbls = {} 

    # set up a table for each group
    for group_name, group_info in ch_groups.items():

        tbl = lgdo.Table(buffer_size)
        if init_obj is not None:
            channel = None # for dummy ch_group
            # Note: all ch in ch_list will be written to the same table. So it
            # should suffice to initials for first channel in the list
            if 'ch_list' in group_info: channel = group_info['ch_list'][0]
            init_obj.initialize_lgdo_table(tbl, channel)

        group_info['table'] = tbl

        if group_name == '':
            if len(ch_groups) != 1:
                print("Error: got dummy group (no name) in non-dummy ch_groups")
                return None
            return tbl

        # cache the table to a ch-indexed dict for quick look-up
        for ch in group_info['ch_list']: ch_to_tbls[ch] = tbl

    return ch_to_tbls


# can be removed?
'''
def set_outputs(ch_groups, out_file_template=None, grp_path_template='{group_name}'):
    """ Set up output filenames and/or group paths for the channel group

    Parameters
    ----------
    out_file_template : str or dict
        Value(s) for attribute "out_file" when not provided explicitly by the user.
        If a string, use it for the out_file attribute for all ch groups
        If a dict, the out_file must be keyed by the ch_group's "system"
        You can use {system} in the template and the system name will be added.

    group_path_template : str or dict (optional)
        If attribute "group_path" is not provided explicitly by the user,
        ch_groups are stored by default in path {group_name} inside their output
        files. You can use {group_name} as well as {system} and the
        corresponding group info will be filled in for you.  If a dict, the
        grp_path_template is keyed by the ch_group's name Example:
        grp_path_template='/data/{system}/{group_name}/raw'
    """
    for group_name, group_info in ch_groups.items():

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


def create_dummy_ch_group():
    return { '' : {} }
