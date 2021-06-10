"""
io_base.py

base classes for decoding daq data into raw lh5 files
"""

from abc import ABC
import numpy as np

from pygama import lh5


class DataDecoder(ABC):
    """Decodes DAQ stream data packets.

    The values that get decoded need to be described by a dict called
    'decoded_values' that helps determine how to set up the buffers and write
    them to file. See ORCAStruck3302 for an example.

    Subclasses should define a method for decoding data to a buffer like
    decode_packet(packet, data_buffer, packet_id, verbose=False)

    Garbage collection writes binary data as an array of uint32s to a
    variable-length array in the output file. If a problematic packet is found,
    call put_in_garbage(packet). User should set up an enum or bitbank of garbage
    codes to be stored along with the garbage packets.
    """
    def __init__(self, garbage_length=256, packet_size_guess=1024):
        self.garbage_table = lh5.Table(garbage_length)
        shape_guess=(garbage_length, packet_size_guess)
        self.garbage_table.add_field('packets',
                                     lh5.VectorOfVectors(shape_guess=shape_guess, dtype='uint8'))
        self.garbage_table.add_field('packet_id',
                                     lh5.Array(shape=garbage_length, dtype='uint32'))
        # TODO: add garbage codes enum attribute: user supplies in constructor
        # before calling super()
        self.garbage_table.add_field('garbage_code',
                                     lh5.Array(shape=garbage_length, dtype='uint32'))


    def get_decoded_values(self, channel=None):
        """ Get decoded values (optionally for a given channel)

        Must overload for your decoder if it has channel-specific decoded
        values. Note: implement channel = None returns a "default" decoded_values

        Otherwise, just returns self.decoded_values, which should be defined in
        the constructor
        """
        if channel is None: return self.decoded_values
        name = type(self).__name__
        print("You need to implement channel-specific get_decoded_values for", name)
        return None


    def initialize_lh5_table(self, lh5_table, channel=None):
        """ initialize and lh5 Table based on decoded_values 
        channel is the channel according to ch_group
        """
        if not hasattr(self, 'decoded_values'):
            name = type(self).__name__
            print(name, 'Error: no decoded_values available for setting up buffer')
            return
        dec_vals = self.get_decoded_values(channel)
        size = lh5_table.size
        for field, fld_attrs in dec_vals.items():
            attrs = fld_attrs.copy()
            if 'dtype' not in attrs:
                name = type(self).__name__
                print(name, 'Error: must specify dtype for', field)
                continue

            dtype = attrs.pop('dtype')
            if 'datatype' not in attrs:
                # no datatype: just a "normal" array
                # allow to override "kind" for the dtype for lh5
                if 'kind' in attrs:
                    attrs['datatype'] = 'array<1>{' + attrs.pop('kind') + '}'
                lh5_table.add_field(field, lh5.Array(shape=size, dtype=dtype, attrs=attrs))
                continue

            datatype = attrs.pop('datatype')

            # handle waveforms from digitizers in a uniform way
            if datatype == 'waveform':
                wf_table = lh5.Table(size)

                # Build t0 array. No attributes for now
                # TODO: add more control over t0: another field to fill it?
                # Optional units attribute?
                t0_attrs = { 'units': 'ns' }
                wf_table.add_field('t0', lh5.Array(nda=np.zeros(size, dtype='float'), attrs = t0_attrs))

                # Build sampling period array with units attribute
                wf_per = attrs.pop('sample_period')
                dt_nda = np.full(size, wf_per, dtype='float')
                wf_per_units = attrs.pop('sample_period_units')
                dt_attrs = { 'units': wf_per_units }
                wf_table.add_field('dt', lh5.Array(nda=dt_nda, attrs = dt_attrs))

                # Build waveform array. All non-popped attributes get sent
                # TODO: add vector of vectors and compression capabilities
                wf_len = attrs.pop('length')
                dims = [1,1]
                aoesa = lh5.ArrayOfEqualSizedArrays(shape=(size,wf_len), dtype=dtype, dims=dims, attrs=attrs)
                wf_table.add_field('values', aoesa)

                lh5_table.add_field(field, wf_table)
                continue

            # If we get here, must be a LH5 datatype
            datatype, shape, elements = lh5.parse_datatype(datatype)

            if datatype == 'array_of_equalsized_arrays':
                length = attrs.pop('length')
                dims = [1,1]
                aoesa = lh5.ArrayOfEqualSizedArrays(shape=(size,length), dtype=dtype, dims=dims, attrs=attrs)
                lh5_table.add_field(field, aoesa)
                continue

            if elements.startswith('array'): # vector-of-vectors
                length_guess = size
                if 'length_guess' in attrs: length_guess = attrs.pop('length_guess')
                vov = lh5.VectorOfVectors(shape_guess=(size,length_guess), dtype=dtype, attrs=attrs)
                lh5_table.add_field(field, vov)
                continue

            else:
                name = type(self).__name__
                print(name, 'Error: do not know how to make a', datatype, 'for', field)


    def put_in_garbage(self, packet, packet_id, code):
        i_row = self.garbage_table.loc
        p8 = np.frombuffer(packet, dtype='uint8')
        self.garbage_table['packets'].set_vector(i_row, p8)
        self.garbage_table['packet_id'].nda[i_row] = packet_id
        self.garbage_table['garbage_codes'].nda[i_row] = code
        self.garbage_table.push_row()


    def write_out_garbage(self, filename, group='/', lh5_store=None):
        if lh5_store is None: lh5_store = lh5.Store()
        n_rows = self.garbage_table.loc
        if n_rows == 0: return
        lh5_store.write_object(self.garbage_table, 'garbage', filename, group, n_rows=n_rows, append=True)
        self.garbage_table.clear()



# TODO: remove this after all DataTaker references have been removed from old
# code
class DataTaker(DataDecoder):
    pass
