def expand_ch_groups(ch_groups):
    """
    Expand a ch_group from its json shorthand

    ch_groups: a dictionary whose keys are the names of channel groupings,
        and whose values have the following fields:
        ch_list (list): a list of channels and channel ranges associated
            with this group
        system (str, optional): name of the (sub)system to which this group
            belongs. Used to set e.g. the name of the output file

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
    same file. A ch group's info can be updated with things like the LH5
    table to which this channel's data gets written, the name of the output
    file, etc.

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

