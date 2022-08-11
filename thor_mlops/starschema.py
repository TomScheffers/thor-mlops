import pyarrow as pa
import pyarrow.compute as c
from typing import List, Tuple, Union

class ThorStarSchema():
    def __init__(self):
        self.tables = {}

    # TRACKING TABLES
    def register_table(self, name: str, table: pa.Table, keys: List[str], core: bool = False):
        assert all(k in table.column_names for k in keys)
        self.tables[name] = {
            'table': table,
            'keys': keys,
            'core': core
        }
    
    # ENRICHING
    def enrich(self, base: pa.Table, verbose: bool = False):
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
        return base

    def growth_rate(self, base: pa.Table):
        rate = 1
        for k, v in self.tables.items():
            if v['core']: # WE CAN ONLY GROW FROM CORE FEATURES
                keys_overlap = [k for k in v['keys'] if k in base.column_names]
                if not keys_overlap: # WE ONLY GROW WHEN THERE IS A CROSS JOIN (NO KEYS OVERLAP)
                    rate *= v['table'].num_rows
        return rate



