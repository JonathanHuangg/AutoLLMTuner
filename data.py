import polars as pl

MATH500 = pl.read_ndjson('hf://datasets/HuggingFaceH4/MATH-500/test.jsonl')
MATH500.head(50).write_parquet("math50.parquet")
