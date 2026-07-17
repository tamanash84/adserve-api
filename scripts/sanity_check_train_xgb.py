import pyarrow.parquet as pq
import duckdb

big_file = r"../data/bandit/training/date=2026-01-10/hour=08/train.parquet"

pf = pq.ParquetFile(big_file)
schema = pf.schema_arrow
meta = pf.metadata

print(schema)
print(meta)

del pf

with duckdb.connect() as con:

    df = con.execute(f"""
        SELECT *
        FROM read_parquet('{big_file}')
        LIMIT 1000
    """).df()




