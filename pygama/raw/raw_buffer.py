"""
raw_buffer.py: manages data buffering for raw data conversion

Manages LGDO buffers and their corresponding output streams. Allows for
one-to-many mapping of input sreams to output streams.

Primary Classes
---------------
RawBuffer: a single lgdo buffer

RawBufferList: a collection of RawBuffers with lgdo's that all have the same
structure

RawBufferLibrary: a collection (dict) of RawBufferLists, e.g. one for each
data-generating DAQ object

RawBuffers support a json short-hand notation, see
RawBufferLibrary.set_from_json_dict() for full specification.
Example json yielding a valid RawBufferLibrary: 

{
  "FlashCamEventDecoder" : {
    "g{key:0>3d}" : {
      "key_list" : [ [24,64] ],
      "out_stream" : "$DATADIR/{file_key}_geds",
      "out_name" : "geds/{name}"
    },
    "spms" : {
      "key_list" : [ [6,23] ],
      "out_stream" : "$DATADIR/{file_key}_spms",
      "out_name" : "spms/{name}"
    },
    "puls" : {
      "key_list" : [ 0 ],
      "out_stream" : "$DATADIR/{file_key}_auxs",
      "out_name" : "auxs/{name}"
    },
    "muvt" : {
      "key_list" : [ 1, 5 ],
      "out_stream" : "$DATADIR/{file_key}_auxs",
      "out_name" : "auxs/{name}"
    }
  }
}
"""

import os
from pygama import lgdo

class RawBuffer:
    '''
    A RawBuffer is in essence a an lgdo object (typically a Table) to which
    decoded data will be written, along with some meta-data distinguishing
    what data goes into it, and where the lgdo gets written out

    Attributes
    ----------
    lgdo : lgdo 
        the lgdo used as the actual buffer. Typically a table. Set to None upon
        creation so that the user can initialize it later.
    key_list : list
        a list of keys (e.g. channel numbers) identifying data to be written
        into this buffer. The key scheme is specific to the decoder with which
        the raw_group is associated. This is called "key_list" instead of "keys"
        to avoid confusion with the dict function "keys()", i.e. raw_group.keys()
    out_stream : str (optional)
        the output stream to which the raw_buffer's lgdo should be sent or
        written (e.g. '/path/filename')
    out_name : str (optional)
        the name / identifier of the object in the ouput stream 
        (e.g. '/group/obj_name')
    '''

    def __init__(self, lgdo=None, key_list=[], out_stream='', out_name=''):
        self.lgdo = lgdo
        self.key_list = key_list
        self.out_stream = out_stream
        self.out_name = out_name

    def init_lgdo(self, lgdo_class=lgdo.Table, init_obj=None, key=None, **lgdo_args):
        ''' initialize this buffer's lgdo

        Parameters
        ----------
        lgdo_class : class name or None
            the class of the lgdo for this buffer. Only used if self.lgdo is None.
            If lgdo_class is None, init_obj will allocate the lgdo
        init_obj : obj
            an object with an init_lgdo(lgdo, key, **lgdo_args) function that can be
            used to initialize the lgdo (e.g. set up the columns of the Table)
        key : int, str, etc
            used by init_obj to initialize the lgdo for a particular key (e.g. to
            have different trace lengths for different channels of a piece of
            hardware). Leave as None if such specialization is not necessary
        lgdo_args : dict
            arguments used for the lgdo's __init__ function. Only used if self.lgdo
            is not None
        '''
        if self.lgdo is None and lgdo_class is not None: 
            self.lgdo = lgdo_class(**lgdo_args)
        if init_obj is not None: init_obj.init_lgdo(self.lgdo, key, **lgdo_args)



class RawBufferList(list):
    '''
    A RawBufferDict holds a collection of RawBuffers of identical structure
    (same format lgdo's). 
    '''


    def init_lgdos(self, lgdo_class=lgdo.Table, init_obj=None, key=None, **lgdo_args):
        ''' Initialize the lgdos in this list 

        See RawBuffer.init_lgdo for parameter info
        '''
        for rb in self: rb.init_lgdo(lgdo_class, init_obj, key, **lgdo_args)


    def get_keyed_dict(self):
        ''' returns a dict of RawBuffers built from the buffers' key_lists

        Different keys may point to the same buffer.
        Requires the buffers in the list to have non-overlapping key lists
        '''
        keyed_dict = {}
        for rb in self:
            for key in rb.key_list: keyed_dict[key] = rb
        return keyed_dict


    def set_from_json_dict(self, json_dict, kw_dict={}):
        ''' set up a RawBufferList from a dict written in json shorthand

        See RawBufferLibrary.set_from_json_dict() for details

        Note: json_dict is changed by this function
        '''
        expand_rblist_json_dict(json_dict, kw_dict)
        for name in json_dict:
            rb = RawBuffer()
            if 'key_list' in json_dict[name]: 
                rb.key_list = json_dict[name]['key_list']
            if 'out_stream' in json_dict[name]: 
                rb.out_stream = json_dict[name]['out_stream']
            if 'out_name' in json_dict[name]: 
                rb.out_name = json_dict[name]['out_name']
            self.append(rb);


    def get_list_of(self, key):
        """
        Return a list of values of RawBuffer.key

        Parameters
        ----------
        key : str
            The RawBuffer attribute queried to make the list

        Returns
        -------
        values : list
            The list of values of RawBuffer.key

        Example
        -------
        output_file_list = rbl.get_list_of('out_stream')
        """
        values = []
        for rb in self:
            if not hasattr(rb, key): continue
            val = getattr(rb, key)
            if val not in values: values.append(val)
        return values



class RawBufferLibrary(dict):
    '''
    A RawBufferLibrary is a collection of RawBufferLists associated with the
    names of decoders that can write to them
    '''
    def __init__(self, json_dict=None, kw_dict={}):
        if json_dict is not None: 
            self.set_from_json_dict(json_dict, kw_dict)

    def set_from_json_dict(self, json_dict, kw_dict={}):
        ''' set up a RawBufferLibrary from a dict written in json shorthand

        Basic structure:
        {
        "list_name" : {
          "name" : {
              "key_list" : [ key1, key2, ... ],
              "out_stream" : "out_stream_str",
              "out_name" : "out_name_str"
          }
        }

        Allowed shorthands, in order of exapansion:
        * key_list may have entries that are 2-integer lists corresponding to
          the first and last integer keys in a continguous range (e.g. of
          channels) that get stored to the same buffer. These simply get
          replaced with the explicit list of integers in the range. We use lists
          not tuples for json compliance.
        * The "name" can include {key:xxx} format specifiers, indicating that
          each key in key_list should be given its own buffer with the
          corresponding name.  The same specifier can appear in out_path to
          write the key's data to its own output path.
        * You may also include keywords in your out_stream and out_name
          specification whose values get sent in via kwdict These get evaluated
          simultaneously with the {key:xxx} specifiers.
        * Environment variables can also be used in out_stream. They get
          expanded after kw_dict is handled and thus can be used inside kw_dict
        * out_stream and out_name can also include {name}, to be replaced with
          the buffer's "name"

        Parameters
        ----------
        json_dict : dict
            dict loaded from a json file written in the allowed shorthand.
            Note: json_dict is changed by this function
        kw_dict : dict
            dict of keyword-value pairs for substitutions into the out_stream
            and out_name fields
        '''                
        for list_name in json_dict:
            if list_name not in self: self[list_name] = RawBufferList()
            self[list_name].set_from_json_dict(json_dict[list_name], kw_dict)



def expand_rblist_json_dict(json_dict, kw_dict):
    """ Expand shorthands in json_dict representing a RawBufferList

    See RawBufferLibrary.set_from_json_dict() for details

    Note: json_dict is changed by this function
    """
    # get the original list of groups because we are going to change the
    # dict.keys() of json_dict inside the next list. Note: we have to convert
    # from dict_keys to list here otherwise the loop complains about changing
    # the dictionary during iteration
    buffer_names = list(json_dict.keys())
    for name in buffer_names:
        if name == '':
            if len(json_dict) != 1:
                print("Error: got dummy name ('') in non-dummy json_dict")
                return None
            return
        info = json_dict[name] # changes to info will change json_dict[name]
        # make sure we have a key list
        if 'key_list' not in info: 
            print(f'expand_json_dict: {name} is missing key_list')
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
        
        # Expand list_names if name contains a key-based formatter
        if '{key' in name: 
            for key in info['key_list']:
                expanded_name = name.format(key=key)
                json_dict[expanded_name] = info.copy()
                json_dict[expanded_name]['key_list'] = [key];
            json_dict.pop(name)

    # now re-iterate and exand out_paths
    for name, info in json_dict.items():
        if len(info['key_list']) == 1: 
            kw_dict['key'] = info['key_list'][0]
        if 'out_stream' in info:
            if '{name' in info['out_stream']: kw_dict['name'] = name
            info['out_stream'] = info['out_stream'].format(**kw_dict)
            info['out_stream'] = os.path.expandvars(info['out_stream'])
        if 'out_name' in info:
            if '{name' in info['out_name']: kw_dict['name'] = name
            info['out_name'] = info['out_name'].format(**kw_dict)
            info['out_name'] = os.path.expandvars(info['out_name'])



