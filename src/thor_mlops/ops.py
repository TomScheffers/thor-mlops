import pyarrow as pa
import orjson as json
import numpy as np

def loads_json_column(table: pa.Table, column:str, drop:bool = False) -> pa.Table:
    arr = np.vectorize(json.loads)(table.column(column).to_numpy())
    arr[arr == None] = dict()
    keys = set.union(*np.vectorize(lambda x: set(x.keys()))(arr[:min(arr.shape[0], 1000)])) # Gather keys from first 1000 samples
    jt = pa.Table.from_pylist(arr, schema=pa.schema([(k, pa.string()) for k in keys])) 
    for pc in jt.column_names:
        table = table.append_column(column + '/' + pc, jt.column(pc))
    return (table.drop([column]) if drop else table)

# Show for easier printing
def head(table, n=5, max_width=100):
    if table.num_rows == 0:
        print("No data in table")
        return
    
    # Extract head data
    t = table.slice(length=n)
    head = {k: list(map(str, v)) for k, v in t.to_pydict().items()}

    # Calculate width
    col_width = list(map(len, head.keys()))
    data_width = [max(map(len, h)) for h in head.values()]

    # Print data
    data = [list(head.keys())] + [[head[c][i] for c in head.keys()] for i in range(t.num_rows)]
    for i in range(len(data)):
        adjust = [w.ljust(max(cw, dw) + 2) for w, cw, dw in zip(data[i], col_width, data_width)]
        print(('Row  ' if i == 0 else str(i-1).ljust(5)) + "".join(adjust)[:max_width])
    print('\n')