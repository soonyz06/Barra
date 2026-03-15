from yahooquery import Ticker
import polars as pl
from datetime import datetime
from pathlib import Path
import time

class Loader: #abit scuffed :)
    def __init__(self):
        self.basepath = Path.cwd() / "data" 
        self.basepath.mkdir(exist_ok=True, parents=True)
        self.identifiers = ["symbol", "date"]
         
    def fetch_history(self, symbols, period, interval="1d"): 
        assert isinstance(symbols, list), "Symbols should be a list"
        if len(symbols)==0: return None
        
        print(f"Fetching History ({period},{interval}) -> {len(symbols)}")
        ticker = Ticker(symbols, asynchronous=True)
        history = ticker.history(period=period, interval=interval)
        if history.empty:
            print("Dataframe is empty")
            return None
        history = pl.from_pandas(history, include_index=True)
        history = history.with_columns(pl.col("date").cast(pl.Date))
        return history

    def fetch_profile(self, symbols):
        assert isinstance(symbols, list), "Symbols should be a list"
        if len(symbols)==0: return None
        print(f"Fetching Profile -> {len(symbols)}")
        ticker = Ticker(symbols, asynchronous=True)
        pf = ticker.summary_profile
        profile = pl.DataFrame({"symbol": key, **val} for key, val in pf.items()).select(["symbol", "country", "industry", "sector"])
        return profile
    
    def fetch_generator(self, symbols, fetch_func, other_args, batch_size, schema):
        for i in range(0, len(symbols), batch_size):
            try:
                current_batch = symbols[i : i + batch_size]            
                batch_df = fetch_func(current_batch, **other_args) ##add retry/skip logic
                time.sleep(0.1) 
                
                if batch_df is not None and not batch_df.is_empty():
                    if schema is not None:
                        for col, dtype in schema.items():
                            if col not in batch_df.columns:
                                batch_df = batch_df.with_columns(pl.lit(None).cast(dtype).alias(col))
                        batch_df = batch_df.select(list(schema.keys())).cast(schema)
                    yield batch_df
            except Exception as e:
                print(f"[WARNING]Skipped symbols {i} to {i+batch_size} due to {e}")
            
    def load_data(self, symbols, dirname, fetch_func, other_args=None, schema=None): ##add refresh/updating
        if other_args is None: other_args = {}
        dirpath = self.basepath / dirname
        dirpath.mkdir(exist_ok=True, parents=True)

        batch_size = 5
        buffer_limit = 100
        
        existing_symbols = set()
        if list(dirpath.glob("*.parquet")):
            existing_symbols = set(
                pl.scan_parquet(dirpath / "*.parquet")
                .select("symbol").unique().collect().get_column("symbol")
            )

        missing_symbols = [s for s in symbols if s not in existing_symbols]
        if missing_symbols:
            buffer = []
            buffer_size = 0
            
            for batch_df in self.fetch_generator(missing_symbols, fetch_func, other_args, batch_size, schema): #seperate fetch and orchestration logic
                buffer.append(batch_df)
                buffer_size += len(batch_df)
                
                if buffer_size >= buffer_limit: #trade off: num I/O vs (RAM usage + crash risk) 
                    self.write_data(buffer, dirpath, "batch", schema) 
                    buffer = []
                    buffer_size = 0
                    
            if buffer:
                self.write_data(buffer, dirpath, "batch", schema)
        return pl.scan_parquet(dirpath / "*.parquet")
                
    def write_data(self, buffer, dirpath, tag="batch", schema=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = dirpath / f"{tag}_{ts}.parquet"
        temppath = filepath.with_suffix(".tmp")

        df = pl.concat(buffer)
        if df.is_empty():
            print("[ERROR]No data to write.")
            return self

        df = df.with_columns(pl.col("ts").fill_null(ts)) if ts in df.columns else df.with_columns(pl.lit(ts).alias("ts"))
        if schema is not None: df = df.cast(schema)
        df.write_parquet(temppath) #write-rename is safer
        temppath.rename(filepath)
        print(f"[SUCCESS]Wrote {len(df):,} rows to {filepath.name}")
        return self

    def compact_data(self, dirname, schema):
        dirpath = self.basepath / dirname
        batches = list(dirpath.glob("batch_*.parquet"))

        files_threshold = 20
        file_count = len(batches)
        if file_count < files_threshold:  
            return self

        lf = pl.scan_parquet(dirpath / "*.parquet")
        cols = lf.collect_schema().names()
        identifiers = [i for i in self.identifiers if i in cols]
        lf = lf.sort("ts", descending=False).unique(subset=identifiers, keep="last")
        df = lf.collect()

        self.write_data([df], dirpath, "master", schema=schema) #so don't compact the compacted or add checks do don't add 1mb to 1GB file
        for f in batches:
            f.unlink()

        for tmp_file in dirpath.glob("*.tmp"):
            try:
                tmp_file.unlink()
            except Exception as e:
                print(f"[WARNING]Could not delete {tmp_file}: {e}")
        return self        
