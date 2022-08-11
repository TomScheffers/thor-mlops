import pyarrow as pa
import pyarrow.compute as c
from typing import List, Tuple, Union

# Cleaning functions
def clean_numerical(arr: pa.array, impute: float = 0.0, clip_min: float = None, clip_max: float = None) -> pa.array:
    arr = arr.cast(pa.float32()).fill_null(impute)
    if clip_min: arr = c.if_else(c.greater(arr, pa.scalar(clip_min)), arr, pa.scalar(clip_min))
    if clip_max: arr = c.if_else(c.less(arr, pa.scalar(clip_max)), arr, pa.scalar(clip_max))
    return arr

def clean_label(arr: pa.array, categories: List[str] = []) -> Tuple[pa.array, List[str]]:
    arr = arr.cast(pa.string()).dictionary_encode()
    dic = arr.dictionary.to_pylist()
    if categories:
        dmap = [(categories.index(v) if v in categories else -1) for v in dic]
        return (c.take(pa.array(dmap), arr.indices), categories)
    else:
        return (arr.indices.fill_null(-1), dic)

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
    def __init__(self, name: str, impute: str = 'mean', clip: bool = True, v_min: float = None, v_mean: float = None, v_max: float = None):
        self.name, self.impute, self.clip = name, impute, clip
        self.measured = any((v_min, v_mean, v_max))
        self.mean, self.min, self.max = (v_mean or 0), (v_min or 0), (v_max or 0)

    def to_dict(self) -> dict:
        return {"name": self.name, "type": "numerical", "impute": self.impute, "clip": self.clip, "v_min": self.min, "v_mean": self.mean, "v_max": self.max}

    def update(self, arr: pa.array):
        self.mean = float(c.mean(arr.cast(pa.float32())).as_py())
        minmax = c.min_max(arr)
        self.min, self.max = float(minmax['min'].as_py()), float(minmax['max'].as_py())

    def value(self) -> float:
        if hasattr(self, self.impute):
            return getattr(self, self.impute)
        else:
            raise Exception("{} is not a valid impute method".format(self.impute))
    
    def clean(self, arr: pa.array) -> pa.array:
        if not self.measured:
            self.update(arr)
        cln = clean_numerical(arr, impute=self.value(), clip_min=(self.min if self.clip else None), clip_max=(self.max if self.clip else None))
        return cln

class CategoricalColumn():
    def __init__(self, name: str, method: str, categories: List[str] = []):
        self.name, self.method, self.categories = name, method, categories
        self.measured = (True if categories else False)

    def to_dict(self) -> dict:
        return {"name": self.name, "type": "categorical", "method": self.method, "categories": self.categories}

    def update(self, categories: List[str]):
        self.categories = self.categories + [c for c in categories if c not in self.categories]

    def clean(self, arr: pa.array) -> pa.array:
        if self.method == 'one_hot':
            cln, cats = clean_onehot(arr, categories=self.categories)
        else:
            cln, cats = clean_label(arr, categories=self.categories)
        if not self.measured:
            self.categories = cats
        return cln

class ThorTableCleaner():
    def __init__(self):
        self.columns = []

    # REGISTERING COLUMNS
    def register_numerical(self, name: str, impute: str = 'mean', clip: bool = True):
        self.columns.append(NumericalColumn(name, impute, clip))

    def register_label(self, name: str, categories: List[str] = []):
        self.columns.append(CategoricalColumn(name, method='label', categories=categories))
    
    def register_one_hot(self, name: str, categories: List[str] = []):
        self.columns.append(CategoricalColumn(name, method='one_hot', categories=categories)) 

    # CLEANING
    def clean_column(self, table: pa.Table, column: Union[NumericalColumn, CategoricalColumn]) -> Tuple[List[str], List[pa.array]]:
        arr = table.column(column.name).combine_chunks()
        cln = column.clean(arr)
        if isinstance(column, CategoricalColumn) and column.__dict__.get('method', '') == 'one_hot':
            return [column.name + '_' + cat for cat in column.categories], cln
        else:
            return [column.name], [cln]

    def fit(self, table: pa.Table, numericals: list = [], labels: list = [], one_hots: list = []):
        [self.register_numerical(c) for c in numericals], [self.register_label(c) for c in labels], [self.register_one_hot(c) for c in one_hots]

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

    def fit_transform(self, table: pa.Table, numericals: list = [], labels: list = [], one_hots: list = [], label: str = None):
        self.fit(table=table, numericals=numericals, labels=labels, one_hots=one_hots)
        return self.transform(table=table, label=label)

    # SERIALIZATION
    def to_dict(self):
        return [column.to_dict() for column in self.columns]

    def from_dict(self, columns):
        for column in columns:
            t = column.pop('type')
            if t == 'numerical':
                self.columns.append(NumericalColumn(**column))
            else:
                self.columns.append(CategoricalColumn(**column))  
        return self

    