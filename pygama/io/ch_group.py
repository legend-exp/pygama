from pygama import lh5

def expand_ch_groups(ch_groups):
    """
    Expand a ch_group from its json shorthand

    ch_groups: a dictionary whose keys are the names of channel groupings,
        and whose values have the following fields:
        ch_list (list): a list of channels and channel ranges associated
            with this group
        system (str, optional): name of the (sub)system to which this group
            belongs. Used to set e.g. the name of the output file or group path,
            etc.

    In json we allow a shorthand where the group name and system name can
    include a {ch:xxx} format specifier, indicating that each channel in
    ch_list should be given its own group with the corresponding name. In
    this function, we expand this formatting to give a list of
    properly-named groups with the right channel specified in each.

    In json we also allow ch_list to have entries that are 2-integer lists
    corresponding to the first and last channels in a continguous range of
    channels that belong in the group. In this function we expand those
    ranges to give a properly formated list of ints.

    Note: the intent is that channels in the same group get written to the
    same table, and that groups in the same system get written out to the
    same file. 

    Note 2: during run time, a ch group's info can be updated with things like
    the lh5 table to which this channel's data gets written, the name of the
    output file, etc.

    Example valid json:
    "FlashCamEventDecoder" : {
      "g{ch:0>3d}" : {
        "ch_list" : [ [24,64] ],
        "system" : "geds"
      },
      "spms" : {
        "ch_list" : [ [6,23] ],
        "system" : "spms"
      },
      "puls" : {
        "ch_list" : [ 0 ],
        "system" : "auxs"
      },
      "muvt" : {
        "ch_list" : [ 1, 5 ],
        "system" : "auxs"
      }
    }
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
            if isinstance(ch_range, list) and len(ch_range) != 2:
                info['ch_list'][i:i+1] = range(ch_range[0], ch_range[1]+1)
                i += ch_range[1]-ch_range[0]
            # any other entry should be an int!
            if not isinstance(ch_range, int): 
                print('ch_group', group, 'has malformed channel list:', ch_range)
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



def get_list_of(key, ch_groups):
    """
    Return a list of values for key in ch_groups[groups]

    key (str): a key of the ch_groups[groups] dict
    ch_groups (dict): a group-name-indexed dict whose values are
        dictionaries of group information (see expand_ch_groups)

    Example: get_list_of('system', ch_groups)
        If ch_groups is the one specified in the json in the expand_ch_groups
        example, this will return the list [ 'geds', 'spms', 'auxs' ]
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
    init_obj : object with initialize_lh5_table() function

    Returns
    -------
    ch_to_tbls : dict or Table
        A channel-indexed dictionary of tables for quick look-up, or if passed a
        dummy group (no group name), return the one table made.
    """

    ch_to_tbls = {} 

    # set up a table for each group
    for group_name, group_info in ch_groups.items():

        tbl = lh5.Table(buffer_size)
        if init_obj is not None:
            channel = None # for dummy ch_group
            # Note: all ch in ch_list will be written to the same table. So it
            # should suffice to initials for first channel in the list
            if 'ch_list' in group_info: channel = group_info['ch_list'][0]
            init_obj.initialize_lh5_table(tbl, channel)

        group_info['table'] = tbl

        if group_name == '':
            if len(ch_groups) != 1:
                print("Error: got dummy group (no name) in non-dummy ch_groups")
                return None
            return tbl

        # cache the table to a ch-indexed dict for quick look-up
        for ch in group_info['ch_list']: ch_to_tbls[ch] = tbl

    return ch_to_tbls


def set_outputs(ch_groups, out_file_template=None, grp_path_template='{group_name}'):
    ''' Set up output filenames and/or group paths for the channel group

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
    '''
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



def create_dummy_ch_group():
    return { '' : {} }
