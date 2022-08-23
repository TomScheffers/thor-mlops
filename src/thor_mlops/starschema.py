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
                base, v['table'] = base.append_column('$join_key', pa.scalar(0)), v['table'].append_column('$join_key', pa.scalar(0)) 
                keys_overlap = '$join_key'
            join_method = ('inner' if v['core'] else 'left outer')
            base = base.join(v['table'], keys=keys_overlap, join_type=join_method)
            if verbose: print(f"Size after {join_method} joining {k} on {keys_overlap}: {base.num_rows} rows")
            if not v['core']: assert base.num_rows == start_size # WE DO NOT WANT TO GROW ON NON-CORE TABLE JOINS

        # PERFORM CALCULATIONS
        for k, func in self.calculations.items():
            # PERFORM CALCULATION & CLEAN & APPEND
            base = base.append_column(k, func(base))
            tc = self.clean_table(table=base.select([k]))
            base = base.append_column(k + '_c', tc.column(k + '_c'))

        if verbose: print("Unclean columns:", self.cln.uninitialized())

        # SPLIT CONTEXT AND CLEANS
        features = [col + '_c' for col in self.cln.features()]
        if verbose: print("Features:", features)
        if verbose: print("Base columns:", base.column_names)
        return base.select([col for col in base.column_names if col[-2:] != '_c']), base.select(features).rename_columns(map(lambda x: x[:-2], features)), base.column(self.label), (base.column(self.weight) if self.weight else None)

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



