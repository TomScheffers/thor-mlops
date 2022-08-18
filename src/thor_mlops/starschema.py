import pyarrow as pa
import pyarrow.compute as c
from typing import List, Tuple, Union

from thor_mlops.ops import loads_json_column
from thor_mlops.clean import ThorTableCleaner

class ThorStarSchema():
    def __init__(self, numericals: List[str], categoricals: List[str], one_hots: List[str], label: str):
        self.tables, self.contexts = {}, []
        self.cln = ThorTableCleaner()
        self.cln.register(numericals=numericals, categoricals=categoricals, one_hots=one_hots)
        self.label = label

    # TRACKING TABLES
    def register_table(self, name: str, table: pa.Table, keys: List[str], contexts: List[str] = [], core: bool = False, json_columns: List[str] = []):
        assert all(k in table.column_names for k in keys)
        # CLEAN JSON STRINGS TO COLUMNS
        for col in json_columns:
            table = loads_json_column(table=table, column=col, drop=True)

        # CLEAN TABLE AND APPEND TO DEFAULT TABLE WITH PREFIX
        clean,_ = self.cln.transform(table=table, warn_missing=False)
        for col in clean.column_names:
            table = table.append_column(col + "_c", clean.column(col))
        print(table.column_names)

        # CLEAN & SAVE TABLE
        self.tables[name] = {
            'table': table,
            'keys': keys,
            'contexts': contexts,
            'core': core,
            'clean_columns': clean.column_names
        }
    
    # ENRICHING
    def enrich(self, base: pa.Table, verbose: bool = False) -> pa.Table:
        for k, v in self.tables.items():
            keys_overlap = [k for k in v['keys'] if k in base.column_names]
            if not keys_overlap:
                if not v['core']: # AVOID CROSS JOINING NON CORE TABLES
                    if verbose: print(f"Avoiding cross join for table {k}, since it is not core and has no overlapping keys")
                    continue
                base, v['table'] = base.append_column('$join_key', pa.scalar(0)), v['table'].append_column('$join_key', pa.scalar(0)) 
                keys_overlap = '$join_key'
            base = base.join(v['table'], keys=keys_overlap, join_type=('inner' if v['core'] else 'left semi')) # LEFT SEMI AVOIDS DUPLICATING LEFT VALUES IN CASE OF MULTIPLE MATCHES
            if verbose: print(f"Size after joining {k}: {base.num_rows} rows")

        # TODO: CALCULATIONS & UNCLEANED COLUMNS
        print("Unclean columns:", self.cln.uninitialized())

        # SPLIT CONTEXT AND CLEANS
        features = [col + '_c' for col in self.cln.features()]
        return base.select([col for col in base.column_names if col[-2:] != '_c']), base.select(features).rename_columns(map(lambda x: x[:-2], features)), base.column(self.label)

    def growth_rate(self, base: pa.Table) -> int:
        rate = 1
        for _, v in self.tables.items():
            if v['core']: # WE CAN ONLY GROW FROM CORE FEATURES
                keys_overlap = [k for k in v['keys'] if k in base.column_names]
                if not keys_overlap: # WE ONLY GROW WHEN THERE IS A CROSS JOIN (NO KEYS OVERLAP)
                    rate *= v['table'].num_rows
        return rate



