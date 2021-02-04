import sys
import pygama.lh5 as lh5

if len(sys.argv) != 5:
    print('Usage: python', sys.argv[0], '[filename] [table_path] [buffer_size] [arr_col]')
    print('  where arr_col is the name of an Array-like object in one of the table columns.')
    sys.exit()

filename = sys.argv[1]
name = sys.argv[2]
buffer_size = int(sys.argv[3])
arr_col = sys.argv[4]
n_iter = 4

test_rows = n_iter*buffer_size
store = lh5.Store()

comp_table, n_rows_read = store.read_object(name, filename, n_rows=test_rows)

table_buf = store.get_buffer(name, filename, size=buffer_size)

success_its = 0
for i_it in range(n_iter):
    print('iteration', i_it)
    start_row = i_it*buffer_size
    table_buf, n_rows_read = store.read_object(name, filename, start_row=start_row, obj_buf=table_buf)
    if n_rows_read == 0: 
        print('n_rows_read = 0')
        break
    if not (table_buf[arr_col].nda[:n_rows_read] == comp_table[arr_col].nda[start_row:start_row+n_rows_read]).all():
        print('read error. buffer:')
        print(table_buf[arr_col].nda[:n_rows_read])
        print('comp_table:')
        print(comp_table[start_row:start_row+n_rows_read])
    else: 
        print('success! (e.g.', table_buf[arr_col].nda[0], '=', comp_table[arr_col].nda[start_row])
        success_its += 1

print(success_its, 'successes')

    
