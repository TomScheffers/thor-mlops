import time
import pyarrow.dataset as ds
import pyarrow.compute as c
from thor_mlops.starschema import ThorStarSchema
from thor_mlops.ops import head

sts = ThorStarSchema(
    numericals=['original_price', 'skus', 'properties/colors', 'discount_value'], 
    categoricals=['group_key', 'collection_key', 'sku_name', 'properties/actie', 'properties/brand', 'properties/season', 'properties/color_code', 'properties/life_cycle', 'unknown_column'],
    one_hots=[],
    label='technical',
    weight='technical',
    config={'some key': 'some value'}
)

# Load data
t_sk = ds.dataset("data/skus/", format="parquet").to_table()
t_sc = ds.dataset("data/stock_current/", format="parquet").to_table()

t1 = time.time()
sts.register_table(name='skus', table=t_sk, keys=['sku_key'], core=False, json_columns=['properties'])
sts.register_table(name='skus_2', table=t_sk, keys=['sku_key'], core=False, json_columns=['properties'])
sts.register_calculation(name='discount_value', func=lambda t: c.subtract(t.column('original_price_c'), t.column('original_price_c')))

# Join stock_current
t2 = time.time()

context, X, y, w = sts.enrich(base=t_sc, verbose=True)
print("Context")
head(context)
print("X")
head(X) 

# Mutate table
t3 = time.time()
X = sts.cln.mutate(X)
head(X)

t4 = time.time()
X_train, y_train, X_test, y_test = sts.cln.split(X=X, y=y, perc=0.2)
head(X_train)

t5 = time.time()
sts.cln.write_to_csv(table=sts.cln.align(X=X_train, y=y_train), path='data/train.csv')

print("Timing", t2 - t1, t3 - t2, t4 - t3, t5 - t4, time.time() - t5)

# Serialization
sts.to_json('starschema.json')
sts = ThorStarSchema.from_json('starschema.json')
print(sts.config)

sts.register_table(name='skus', table=t_sk, keys=['sku_key'], core=False, json_columns=['properties'])
sts.register_calculation(name='discount_value', func=lambda t: c.subtract(t.column('original_price_c'), t.column('original_price_c')))

context, X, y, w = sts.enrich(base=t_sc, verbose=True)
print("Context")
head(context)
print("X")
head(X) 




