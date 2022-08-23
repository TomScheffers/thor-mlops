import json
import numpy as np
import pyarrow as pa
import pyarrow.compute as c
import pyarrow.csv as csv
from typing import List, Tuple, Union

# Cleaning functions
def clean_numerical(arr: pa.array, impute: float = 0.0, clip_min: float = None, clip_max: float = None) -> pa.array:
    arr = arr.cast(pa.float32()).fill_null(impute)
    if clip_min: arr = c.if_else(c.greater(arr, pa.scalar(clip_min)), arr, pa.scalar(clip_min))
    if clip_max: arr = c.if_else(c.less(arr, pa.scalar(clip_max)), arr, pa.scalar(clip_max))
    return c.round(arr, ndigits=5)

def clean_categorical(arr: pa.array, categories: List[str] = []) -> Tuple[pa.array, List[str]]:
    arr = arr.cast(pa.string()).dictionary_encode()
    dic = arr.dictionary.to_pylist()
    if categories:
        dmap = [(categories.index(v) + 1 if v in categories else 0) for v in dic]
        return (c.take(pa.array(dmap), arr.indices), categories)
    else:
        return (c.add(arr.indices, pa.scalar(1)).fill_null(0), dic)

def clean_onehot(arr: pa.array, categories: List[str] = [], drop_first: bool = False) -> Tuple[pa.array, List[str]]:
    arr = arr.cast(pa.string())
    if categories:
        clns =[c.equal(arr, v).fill_null(False) for v in categories]
    else:
        categories = [u for u in arr.unique().to_pylist() if u]
        clns = [c.equal(arr, v).fill_null(False) for v in categories]
    return clns[(1 if drop_first else 0):], categories[(1 if drop_first else 0):]

# Cleaning Classes
class NumericalColumn():
    def __init__(self, name: str, impute: str = 'mean', clip: bool = True, v_min: float = None, v_mean: float = None, v_max: float = None,  mutate_perc: float = 0.0, fill_value: int = -1):
        self.name, self.impute, self.clip = name, impute, clip
        self.measured = any((v_min, v_mean, v_max))
        self.mean, self.min, self.max = (v_mean or 0), (v_min or 0), (v_max or 0)
        self.mutate_perc, self.fill_value = mutate_perc, fill_value

    def to_dict(self) -> dict:
        return {"name": self.name, "type": "numerical", "impute": self.impute, "clip": self.clip, "v_min": self.min, "v_mean": self.mean, "v_max": self.max, "mutate_perc": self.mutate_perc, "fill_value": self.fill_value}

    def update(self, arr: pa.array):
        self.mean = float(c.mean(arr.cast(pa.float32())).as_py())
        minmax = c.min_max(arr)
        self.min, self.max = float(minmax['min'].as_py()), float(minmax['max'].as_py())

    def features(self):
        return [self.name]
    
    def value(self) -> float:
        if hasattr(self, self.impute):
            return getattr(self, self.impute)
        else:
            raise Exception("{} is not a valid impute method".format(self.impute))
    
    def clean(self, arr: pa.array) -> pa.array:
        if not self.measured:
            self.update(arr)
            self.measured = True
        cln = clean_numerical(arr, impute=self.value(), clip_min=(self.min if self.clip else None), clip_max=(self.max if self.clip else None))
        return cln

class CategoricalColumn():
    def __init__(self, name: str, categories: List[str] = [], mutate_perc: float = 0.0, fill_value: int = 0):
        self.name, self.categories = name, categories
        self.measured = (True if categories else False)
        self.mutate_perc, self.fill_value = mutate_perc, fill_value

    def to_dict(self) -> dict:
        return {"name": self.name, "type": "categorical", "categories": self.categories, "mutate_perc": self.mutate_perc, "fill_value": self.fill_value}

    def features(self):
        return [self.name]

    def clean(self, arr: pa.array) -> pa.array:
        cln, cats = clean_categorical(arr, categories=self.categories)
        if not self.measured:
            self.categories = cats
            self.measured = True
        return cln

class OneHotColumn():
    def __init__(self, name: str, categories: List[str] = [], mutate_perc: float = 0.0, fill_value: int = 0):
        self.name, self.categories = name, categories
        self.measured = (True if categories else False)
        self.mutate_perc, self.fill_value = mutate_perc, fill_value

    def to_dict(self) -> dict:
        return {"name": self.name, "type": "one_hot", "categories": self.categories, "mutate_perc": self.mutate_perc, "fill_value": self.fill_value}

    def features(self):
        return [self.name + '_' + cat for cat in self.categories]

    def clean(self, arr: pa.array) -> pa.array:
        cln, cats = clean_onehot(arr, categories=self.categories)
        if not self.measured:
            self.categories = cats
            self.measured = True
        return cln

class ThorTableCleaner():
    def __init__(self):
        self.columns = []
    
    # HANDY FUNCTIONS
    def names(self):
        return list(map(lambda x: x.name, self.columns))

    def features(self):
        return [feat for col in self.columns for feat in col.features()]

    def uninitialized(self):
        return [col.name for col in self.columns if not col.measured]
    
    # REGISTERING COLUMNS
    def register_numerical(self, name: str, impute: str = 'mean', clip: bool = True, mutate_perc: float = 0.1, fill_value: int = -1):
        self.columns.append(NumericalColumn(name=name, impute=impute, clip=clip, mutate_perc=mutate_perc, fill_value=fill_value))

    def register_categorical(self, name: str, categories: List[str] = [], mutate_perc: float = 0.1, fill_value: int = 0):
        self.columns.append(CategoricalColumn(name=name, categories=categories, mutate_perc=mutate_perc, fill_value=fill_value))
    
    def register_one_hot(self, name: str, categories: List[str] = [], mutate_perc: float = 0.1, fill_value: int = 0):
        self.columns.append(OneHotColumn(name=name, categories=categories, mutate_perc=mutate_perc, fill_value=fill_value)) 

    def register(self, numericals: List[str] = [], categoricals: List[str] = [], one_hots: List[str] = []):
        [self.register_numerical(c) for c in numericals], [self.register_categorical(c) for c in categoricals], [self.register_one_hot(c) for c in one_hots]

    # CLEANING
    def clean_column(self, table: pa.Table, column: Union[NumericalColumn, CategoricalColumn, OneHotColumn]) -> Tuple[List[str], List[pa.array]]:
        cln = column.clean(table.column(column.name).combine_chunks())
        if isinstance(column, OneHotColumn):
            return [column.name + '_' + cat for cat in column.categories], cln
        else:
            return [column.name], [cln]

    def transform(self, table: pa.Table, label: str = None, warn_missing: bool = True) -> Tuple[pa.Table, pa.array]:
        keys, arrays = [], []
        for column in self.columns:
            if column.name not in table.column_names:
                if warn_missing:
                    print(f"{column.name} is missing in table.")
                continue
            k, a = self.clean_column(table, column)
            keys.extend(k)
            arrays.extend(a)
        return pa.Table.from_arrays(arrays, names=keys), (table.column(label) if label else None)

    def fit_transform(self, table: pa.Table, numericals: list = [], categoricals: list = [], one_hots: list = [], label: str = None):
        self.register(numericals=numericals, categoricals=categoricals, one_hots=one_hots)
        return self.transform(table=table, label=label)

    # ML OPS
    def random_mask(self, n, perc):
        return c.greater(pa.array(np.random.uniform(size=n)), pa.scalar(perc))

    def mutate(self, table: pa.Table) -> pa.Table:
        for col in self.columns:
            if isinstance(col, OneHotColumn):
                continue
            arr = c.if_else(self.random_mask(n=table.num_rows, perc=col.mutate_perc), table.column(col.name), pa.scalar(col.fill_value))
            table = table.drop([col.name]).append_column(col.name, arr)
        return table

    def split(self, X: pa.Table, y: pa.array, perc=0.2) -> Tuple[pa.Table, pa.Table]:
        msk = self.random_mask(n=X.num_rows, perc=perc)
        return X.filter(msk), y.filter(msk), X.filter(c.invert(msk)), y.filter(c.invert(msk))

    def fill_nans(self, table: pa.Table) -> pa.Table:
        for col in self.columns:
            if isinstance(col, OneHotColumn):
                continue
            elif isinstance(col, CategoricalColumn):
                    arr = table.column(col.name).fill_null(col.fill_value)                    
            elif isinstance(col, NumericalColumn):
                arr = table.column(col.name).fill_null(col.fill_value)
            
            table = table.drop([col.name]).append_column(col.name, arr)        
        return table

    def align(self, X: pa.Table, y: pa.array = None) -> pa.Table:
        if y: 
            X = X.append_column('label', y)
            feat = ['label'] + self.features()
        else:
            feat = self.features()
        return X.select(feat)

    # TABLE TO CSV USING CORRECT ORDERING OF COLUMNS
    def write_to_csv(self, table: pa.Table, path:str):
        options = csv.WriteOptions(include_header=False)
        csv.write_csv(table, path, options)

    # SERIALIZATION
    def to_dict(self):
        return {
            'columns': [column.to_dict() for column in self.columns]
        }

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, state):
        cln = ThorTableCleaner()
        ctypes = {
            "numerical": NumericalColumn,
            "categorical": CategoricalColumn,
            "one_hot": OneHotColumn,
        }
        for column in state['columns']:
            t = column.pop('type')
            cln.columns.append(ctypes[t](**column))
        return cln

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            state = json.load(f)
        return cls.from_dict(state)


    
    

    