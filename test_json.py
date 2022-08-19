import pyarrow.dataset as ds
from thor_mlops.ops import loads_json_column

t = ds.dataset("data/stores/", format="parquet").to_table()
print(t)

t = loads_json_column(t, column='store_properties', drop=False)
