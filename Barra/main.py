from yahooquery import Ticker
import polars as pl
from datetime import datetime
from pathlib import Path
import re
from functools import reduce


class History:
    def __init__(self):
        self.basepath = Path.cwd() / "data" / "history"
        self.basepath.mkdir(exist_ok=True, parents=True)
        self.filepath = self.basepath / "history.parquet"

        self.identifiers = ["symbol", "date"]
        
    def fetch_history(self, symbols, period, interval="1d"):
        if len(symbols)==0: return None
        print(f"Fetching History ({period},{interval})-> {len(symbols)}")
        ticker = Ticker(symbols, asynchronous=True)
        history = ticker.history(period=period, interval=interval)
        if history.empty:
            print("Dataframe is empty")
            return None
        history = pl.from_pandas(history, include_index=True)
        history = history.drop("dividends")
        history = history.with_columns(pl.col("date").cast(pl.Date))
        return history

    def load_history(self, symbols):
        try:
            history = pl.read_parquet(self.filepath)
        except:
            history = self.fetch_history(symbols, period="max")
            history.write_parquet(self.filepath)
            return history

        missing = set(symbols) - set(history["symbol"].unique()) #unique is faster
        missing_df = self.fetch_history(missing, period="max")

        #date check + missing
        
        if missing_df is None or missing_df.is_empty():
            return history
        
        history = pl.concat([history, missing_df])
        history.write_parquet(self.filepath)
        return history
    
    def add_pct_change(self, factor, df, period, offset=None, k=1):
        #rolling raw pct change with validation within buffer_days 
        #start: t+offset, end: t+offset+period, default offset = -period
        buffer_days = 5

        period_num, period_unit = self.split_date(period)
        if offset is None: offset = f"0{period_unit}"
        offset_num, offset_unit = self.split_date(offset)
        assert period_unit == offset_unit, "Both periods should be the same"
        period = f"{int(period_num)-int(offset_num)}{period_unit}"
        offset = f"{-int(period_num)}{offset_unit}"
        
        df = df.rolling( 
            index_column="date",
            period=period,
            offset=offset,
            group_by="symbol"
        ).agg([
            ((pl.col("close").last()/pl.col("close").first() -1) * k).alias(factor),
            (pl.col("date").first().alias("actual_start")),
            (pl.col("date").last().alias("actual_end"))
        ])

        df = df.with_columns([
            (pl.col("date").dt.offset_by(offset).alias("target_start"))
        ]).with_columns([
            (pl.col("target_start").dt.offset_by(period).alias("target_end"))
        ])
        
        df = df.with_columns(
            pl.when(
                ((pl.col("actual_start")-pl.col("target_start")).dt.total_days().abs()<=buffer_days)
                &
                ((pl.col("actual_end")-pl.col("target_end")).dt.total_days().abs()<=buffer_days)
            )
            .then(pl.col(factor))
            .otherwise(None)
            .alias(factor)
        ).select(self.identifiers+[factor])
        return df

    def split_date(self, date):
        match = re.match(r"(-?\d+)(.*)", date)
        if not match:
            return 0, date
        return match.groups()   
                   
    def create_UMD(self, df, period, offset=None):
        factor = f"UMD"
        factor = f"{factor}_{period}_{offset}" if offset is not None else f"{factor}_{period}"
        
        df = self.add_pct_change(factor, df, period, offset, k=1)
        return df

    def create_HML(self, df, period, offset=None):
        factor = f"HML"
        factor = f"{factor}_{period}_{offset}" if offset is not None else f"{factor}_{period}"
        
        df = self.add_pct_change(factor, df, period, offset, k=-1)
        return df

    def winsor_factor(self, df, factor, p=0.01):
        df = df.with_columns(
            pl.col(factor).clip(
                pl.col(factor).quantile(p).over("date"),
                pl.col(factor).quantile(1-p).over("date")
            ).alias(factor)
        )
        return df

    def znorm_factor(self, df, factor):
        df = df.with_columns([
            ((pl.col(factor)-pl.col(factor).mean().over("date")) /
             pl.col(factor).std().over("date"))
            .alias(factor)
        ])
        return df

    def combine_factors(self, df, factors, name, weights=None):
        assert isinstance(factors, list), "Should be a list"
        n = len(factors)
        if weights is None: weights = [1/n for i in range(n)]

        combined_expr = sum(pl.col(f)*w for f, w in zip(factors, weights))
        df = df.with_columns([
            combined_expr.alias(name)
        ])
        return self.znorm_factor(df, name)
        

symbols = ["AAPL", "META", "MSFT", "WMT", "COST", "NVDA"]



his = History()
df = his.load_history(symbols)

umd_configs = [("12mo", "0mo"), ("12mo", "1mo"), ("6mo", "0mo"), ("6mo", "1mo")]
hml_configs = [("5y", None), ("3y", None)]

factor_dfs = []
for p, o in umd_configs:
    factor_dfs.append(his.create_UMD(df, p, o))
for p, o in hml_configs:
    factor_dfs.append(his.create_HML(df, p, o))
    
processed_dfs = []
for f_df in factor_dfs:
    f_name = f_df.columns[-1]
    proc = (
        f_df
        .pipe(his.winsor_factor, f_name)
        .pipe(his.znorm_factor, f_name)
    )
    processed_dfs.append(proc)

factor_df = reduce( #reduces processed_dfs to factor_df
    lambda left, right: left.join(right, on=["symbol", "date"], how="left"), 
    processed_dfs
)
df = df.join(factor_df, on=["symbol", "date"], how="left")


df = df.sort("date", "symbol")
print(df.tail())






#log returns are additive -> rolling sum of log returns

#sampling





