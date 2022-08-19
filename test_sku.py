import time
import pyarrow.dataset as ds
import pyarrow.compute as c
from thor_mlops.starschema import ThorStarSchema
from thor_mlops.clean import ThorTableCleaner
from thor_mlops.ops import head, loads_json_column

ss = ThorStarSchema(
    numericals=['original_price', 'skus', 'properties/colors', 'discount_value'], 
    categoricals=['group_key', 'collection_key', 'sku_name', 'properties/actie', 'properties/brand', 'properties/season', 'properties/color_code', 'properties/life_cycle'],
    one_hots=[],
    label='technical'
)

# Load data
t_sk = ds.dataset("data/skus/", format="parquet").to_table()
t_sc = ds.dataset("data/stock_current/", format="parquet").to_table()

t1 = time.time()
ss.register_table(name='skus', table=t_sk, keys=['sku_key'], core=True, json_columns=['properties'])
ss.register_calculation(name='discount_value', func=lambda t: c.subtract(t.column('original_price_c'), t.column('original_price_c')))

# Join stock_current
t2 = time.time()

context, X, y = ss.enrich(base=t_sc, verbose=True)
print("Context")
head(context)
print("X")
head(X) 

# Mutate table
t3 = time.time()
X = ss.cln.mutate(X)
head(X)

t4 = time.time()
X_train, y_train, X_test, y_test = ss.cln.split(X=X, y=y, perc=0.2)
head(X_train)

t5 = time.time()
ss.cln.write_to_csv(table=ss.cln.align(X=X_train, y=y_train), path='data/train.csv')

print("Timing", t2 - t1, t3 - t2, t4 - t3, t5 - t4, time.time() - t5)
