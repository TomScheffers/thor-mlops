import time
import pyarrow.dataset as ds

from thor_mlops.starschema import ThorStarSchema
from thor_mlops.clean import ThorTableCleaner
from thor_mlops.ops import head, loads_json_column

ss = ThorStarSchema()

# Load data
t1 = time.time()
t_sk = ds.dataset("data/skus/", format="parquet").to_table()
t_sc = ds.dataset("data/stock_current/", format="parquet").to_table()

ss.register_table(name='skus', table=t_sk, keys=['sku_key'], core=True)

print(t_sk.num_rows, t_sk.column_names)
head(t_sk)

# Join stock_current
t2 = time.time()

t_en = ss.enrich(base=t_sc, verbose=True)
print(t_en.num_rows, t_en.column_names)
head(t_en)

# Properties to Struct
t3 = time.time()
t_sk = loads_json_column(t_sk, 'properties')
print(t_sk.num_rows, t_sk.column_names)
head(t_sk)

# Cleaning
t4 = time.time()
cln = ThorTableCleaner()
X, y = cln.fit_transform(t_sk, numericals=['original_price', 'skus', 'properties/colors'], labels=['group_key', 'collection_key', 'sku_name', 'properties/actie', 'properties/brand', 'properties/season', 'properties/color_code', 'properties/life_cycle'])
head(X)

# Mutate table
t5 = time.time()
X = cln.mutate(X)
head(X)

t6 = time.time()
X_train, X_test = cln.split(table=X, perc=0.2)
head(X_train)

t7 = time.time()
cln.write_to_csv(table=X_train, path='data/train.csv')

print("Timing", t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, time.time() - t7)
