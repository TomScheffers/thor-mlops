import json
import pyarrow as pa
import pyarrow.compute as c
from typing import List, Tuple, Union

from thor_mlops.ops import loads_json_column
from thor_mlops.clean import ThorTableCleaner

class ThorStarSchema():
    def __init__(self, numericals: List[str], categoricals: List[str], one_hots: List[str], label: str, weight: str = None, config: dict = {}):
        self.tables, self.calculations = {}, {}
        self.numericals, self.categoricals, self.one_hots, self.label, self.weight, self.config = numericals, categoricals, one_hots, label, weight, config
        
        # Register TableCleaner
        self.cln = ThorTableCleaner()
        self.cln.register(numericals=numericals, categoricals=categoricals, one_hots=one_hots)

    # TRACKING TABLES
    def clean_table(self, table: pa.Table, keys: List[str] = [], contexts: List[str] = [], json_columns: List[str] = []):
        # CLEAN JSON STRINGS TO COLUMNS
        for col in json_columns:
            table = loads_json_column(table=table, column=col, drop=True)

        # CLEAN TABLE AND APPEND TO DEFAULT TABLE WITH PREFIX
        clean,_ = self.cln.transform(table=table, warn_missing=False)
        for col in clean.column_names:
            table = table.append_column(col + "_c", clean.column(col))

        # REMOVE ALL COLUMNS WHICH ARE NOT IN KEYS OR CONTEXTS
        table = table.select([col for col in table.column_names if col in keys or col in contexts or col[-2:] == '_c'])
        return table

    def register_table(self, name: str, table: pa.Table, keys: List[str], contexts: List[str] = [], core: bool = False, json_columns: List[str] = []):
        assert all(k in table.column_names for k in keys)

        # CLEAN & SAVE TABLE
        self.tables[name] = {
            'table': self.clean_table(table=table, keys=keys, contexts=contexts, json_columns=json_columns),
            'keys': keys,
            'contexts': contexts,
            'core': core
        }

    def register_calculation(self, name: str, func):
        self.calculations[name] = func
    
    # ENRICHING
    def enrich(self, base: pa.Table, verbose: bool = False) -> pa.Table:
        for k, v in self.tables.items():
            start_size = base.num_rows
            keys_overlap = [k for k in v['keys'] if k in base.column_names]
            if not keys_overlap:
                if not v['core']: # AVOID CROSS JOINING NON CORE TABLES
                    if verbose: print(f"Avoiding cross join for table {k}, since it is not core and has no overlapping keys")
                    continue
                if '$join_key' not in base.column_names: base = base.append_column('$join_key', pa.array([0] * base.num_rows, pa.int8()))
                if '$join_key' not in v['table'].column_names: v['table'] = v['table'].append_column('$join_key', pa.array([0] * v['table'].num_rows, pa.int8()))
                keys_overlap = '$join_key'
            join_method = ('inner' if v['core'] else 'left outer')
            base = base.join(v['table'], keys=keys_overlap, join_type=join_method, right_suffix='_r')

            if verbose: print(f"Size after {join_method} joining {k} on {keys_overlap}: {base.num_rows} rows")
            if not v['core']: assert base.num_rows == start_size # WE DO NOT WANT TO GROW ON NON-CORE TABLE JOINS

            # COALESCE WHEN MULTIPLE VALUES ARE FOUND
            for col in base.column_names:
                if col[-2:] == '_r':
                    if verbose: print(f"Coalescing double columns {col}")
                    arr = c.coalesce(base.column(col[:-2]), base.column(col))
                    base = base.drop([col, col[:-2]]).append_column(col[:-2], arr)

        # PERFORM CALCULATIONS
        for k, func in self.calculations.items():
            # PERFORM CALCULATION & CLEAN & APPEND
            base = base.append_column(k, func(base))
            tc = self.clean_table(table=base.select([k]))
            if k + '_c' in tc.column_names:
                base = base.append_column(k + '_c', tc.column(k + '_c'))

        # ADD MISSING FEATURES
        for col in self.cln.columns:
            for feat in col.features():
                if feat + "_c" not in base.column_names: 
                    if verbose: print(f"Adding missing feature {feat} with default value {col.fill_value}")
                    base = base.append_column(feat + "_c", pa.array([col.fill_value] * base.num_rows))

        # RETURN DATA
        features = [col + '_c' for col in self.cln.features()]
        if verbose: print("Features:", features)
        if verbose: print("Unclean columns:", self.cln.uninitialized())
        if verbose: print("Base columns:", base.column_names)
        return base.select([col for col in base.column_names if col[-2:] != '_c']), base.select(features).rename_columns(map(lambda x: x[:-2], features)), (base.column(self.label) if self.label and self.label in base.column_names else None), (base.column(self.weight) if self.weight and self.weight in base.column_names else None)

    def growth_rate(self, base: pa.Table) -> int:
        rate = 1
        for _, v in self.tables.items():
            if v['core']: # WE CAN ONLY GROW FROM CORE FEATURES
                keys_overlap = [k for k in v['keys'] if k in base.column_names]
                if not keys_overlap: # WE ONLY GROW WHEN THERE IS A CROSS JOIN (NO KEYS OVERLAP)
                    rate *= v['table'].num_rows
        return rate

    # SERIALIZATION
    def to_dict(self):
        return {
            'numericals': self.numericals,
            'categoricals': self.categoricals,
            'one_hots': self.one_hots,
            'label': self.label,
            'weight': self.weight,
            'config': self.config,
            'cleaner': self.cln.to_dict() 
        }

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, state):
        sts = ThorStarSchema(numericals=state['numericals'], categoricals=state['categoricals'], one_hots=state['one_hots'], label=state['label'], weight=state['weight'], config=state['config'])
        sts.cln = ThorTableCleaner.from_dict(state=state['cleaner'])
        return sts

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            state = json.load(f)
        return cls.from_dict(state)



