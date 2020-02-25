import pygama.io.io_base as pgio
import numpy as np

'''
Should print:
[1 2 3 ... 0 0 0]
[6 7 8]
['energy', 'wf']
{'dtype': 'uint16', 'max_length': 10}
None
'''

tb = pgio.TableBuffer()
tb.add_field('energy', attributes = { 'dtype': 'uint32' })
tb.add_field('wf', attributes = { 'dtype': 'uint16', 'max_length': 10 })
data = np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9 ])

for i in range(3):
    tb.energy = data[i]
    tb.wf = data[i+3:i+6]
    tb.next_row()

print(tb.get_data_buffer('energy'))
print(tb.get_flat_buffer('wf', 2))
print(tb.get_column_names())
print(tb.get_attributes('wf'))
print(tb.get_attributes('wf2'))

