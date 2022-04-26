"""
raw_buffer.py: manages data buffering for raw data conversion

Manages LGDO buffers and their corresponding output streams. Allows for
one-to-many mapping of input sreams to output streams.

Primary Classes
---------------
RawBuffer: an lgdo (e.g. a table) along with buffer metadata, such as the
current write location, the list of keys (e.g. channels) that write to it, the
output stream it is associated with (if any), etc. Each data_decoder is
associated with a RawBuffer of a particular format.

RawBufferList: a collection of RawBuffers with lgdo's that all have the same
structure (same type, same fields, etc). A data_decoder will write its output to
a RawBufferList.

RawBufferLibrary: a collection (dict) of RawBufferLists, e.g. one for each
data_decoder. Keyed by the decoder name.

RawBuffers support a json short-hand notation, see
RawBufferLibrary.set_from_json_dict() for full specification.

Example json yielding a valid RawBufferLibrary is below. In the example, the
user would call RawBufferLibrary.set_from_json_dict(json_dict, kw_dict) with
kw_dict containing an entry for 'file_key'. The other keywords {key} and {name}
are understood by and filled in during set_from_json_dict() unless overloaded in
kw_dict. Note the use of the wildcard "*": this will match all other decoder
names / keys.

{
  "FCEventDecoder" : {
    "g{key:0>3d}" : {
      "key_list" : [ [24,64] ],
      "out_stream" : "$DATADIR/{file_key}_geds.lh5:/geds"
    },
    "spms" : {
      "key_list" : [ [6,23] ],
      "out_stream" : "$DATADIR/{file_key}_spms.lh5:/spms"
    },
    "puls" : {
      "key_list" : [ 0 ],
      "out_stream" : "$DATADIR/{file_key}_auxs.lh5:/auxs"
    },
    "muvt" : {
      "key_list" : [ 1, 5 ],
      "out_stream" : "$DATADIR/{file_key}_auxs.lh5:/auxs"
    }
  },
  "*" : {
    "{name}" : {
      "key_list" : [ "*" ],
      "out_stream" : "$DATADIR/{file_key}_{name}.lh5"
    }
  }
}

later: could initially make field "lgdo" a dict of args for lgdo.__init__(),
e.g. to have object-specific buffer sizes
"""

import os
from pygama import lgdo

class RawBuffer:
    '''
    A RawBuffer is in essence a an lgdo object (typically a Table) to which
    decoded data will be written, along with some meta-data distinguishing
    what data goes into it, and where the lgdo gets written out. Also holds on
    to the current location in the buffer for writing.

    Attributes
    ----------
    lgdo : lgdo
        the lgdo used as the actual buffer. Typically a table. Set to None upon
        creation so that the user or a decoder can initialize it later.
    key_list : list
        a list of keys (e.g. channel numbers) identifying data to be written
        into this buffer. The key scheme is specific to the decoder with which
        the RawBuffer is associated. This is called "key_list" instead of "keys"
        to avoid confusion with the dict function "keys()", i.e. raw_buffer.lgdo.keys()
    out_stream : str (optional)
        the output stream to which the raw_buffer's lgdo should be sent or
        written. A colon can be used to separate the stream name/address from an
        in-stream path / port:
        File example: '/path/filename.lh5:/group'
        Socket example: '198.0.0.100:8000'
    out_name : str (optional)
        the name / identifier of the object in the output stream
    '''


    def __init__(self, lgdo=None, key_list=[], out_stream='', out_name=''):
        self.lgdo = lgdo
        self.key_list = key_list
        self.out_stream = out_stream
        self.out_name = out_name
        self.loc = 0
        self.fill_safety = 1


    def __len__(self):
        if self.lgdo is None: return 0
        if not hasattr(self.lgdo, '__len__'): return 1
        return len(self.lgdo)


    def is_full(self):
        return (len(self) - self.loc) < self.fill_safety


    def __str__(self):
        return f'RawBuffer {"{"} lgdo={self.lgdo}, key_list={self.key_list}, out_stream={self.out_stream}, out_name={self.out_name}, loc={self.loc}, fill_safety={self.fill_safety} {"}"}'


    def __repr__(self): return str(self)



class RawBufferList(list):
    '''
    A RawBufferList holds a collection of RawBuffers of identical structure
    (same format lgdo's with the same fields).
    '''


    def get_keyed_dict(self, default=None):
        ''' returns a dict of RawBuffers built from the buffers' key_lists

        Different keys may point to the same buffer. Requires the buffers in the
        RawBufferList to have non-overlapping key lists.
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
            else: rb.out_name = name
            self.append(rb);


    def get_list_of(self, attribute):
        """
        Return a list of values of RawBuffer.attribute

        Parameters
        ----------
        attribute : str
            The RawBuffer attribute queried to make the list

        Returns
        -------
        values : list
            The list of values of RawBuffer.attribute

        Example
        -------
        output_file_list = rbl.get_list_of('out_stream')
        """
        values = []
        for rb in self:
            if not hasattr(rb, attribute): continue
            val = getattr(rb, attribute)
            if val not in values: values.append(val)
        return values

    def clear_full(self):
        for rb in self:
            if rb.is_full(): rb.loc = 0


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
              "out_name" : "out_name_str" (optional)
          }
        }

        By default "name" is used for the RawBuffer's "out_name" attribute, but
        this can be overridden if desired by providing an explicit "out_name"

        Allowed shorthands, in order of expansion:
        * key_list may have entries that are 2-integer lists corresponding to
          the first and last integer keys in a contiguous range (e.g. of
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
        * list_name can use the wildcard "*" to match any other list_name known
          to a streamer
        * out_stream and out_name can also include {name}, to be replaced with
          the buffer's "name". In the case of list_name="*", {name} evaluates to
          list_name

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


    def get_list_of(self, attribute, unique=True):
        """
        Return a list of values of RawBuffer.attribute

        Parameters
        ----------
        attribute : str
            The RawBuffer attribute queried to make the list

        Returns
        -------
        values : list
            The list of values of RawBuffer.attribute

        Example
        -------
        output_file_list = rbl.get_list_of('out_stream')
        """
        values = []
        for rb_list in self.values():
            values += rb_list.get_list_of(attribute)
        if unique: values = list(set(values))
        return values

    def clear_full(self):
        for rb_list in self.values(): rb_list.clear_full()



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
            print("Error: name can't be ''")
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

    # now re-iterate and expand out_paths
    for name, info in json_dict.items():
        if len(info['key_list']) == 1 and info['key_list'][0] != "*":
            kw_dict['key'] = info['key_list'][0]
        if 'out_stream' in info:
            if name != '*' and '{name' in info['out_stream']: kw_dict['name'] = name
            info['out_stream'] = info['out_stream'].format(**kw_dict)
            info['out_stream'] = os.path.expandvars(info['out_stream'])


def write_to_lh5_and_clear(raw_buffers, lh5_store=None, wo_mode='append', verbosity=0):
    ''' Write a list of RawBuffers to lh5 files and then clears them

    Parameters
    ----------
    raw_buffers : list(RawBuffer)
        The list of RawBuffers to be written to file. Note this is not a
        RawBufferList because the RawBuffers may not have the same structure.
    lh5_store : LH5Store or None
        Allows user to send in a store holding a collection of already open
        files (saves some time opening / closing files)
    '''
    if lh5_store is None: lh5_store = lgdo.LH5Store()
    for rb in raw_buffers:
        if rb.lgdo is None or rb.loc == 0: continue # no data to write
        ii = rb.out_stream.find(':')
        if ii == -1:
            filename = rb.out_stream
            group = '/'
        else:
            filename = rb.out_stream[:ii]
            group = rb.out_stream[ii+1:]
            if len(group) == 0: group = '/' # in case out_stream ends with :
        # write...
        lh5_store.write_object(rb.lgdo, rb.out_name, filename, group=group,
                               n_rows=rb.loc, wo_mode=wo_mode, verbosity=verbosity)
        # and clear
        rb.loc = 0
