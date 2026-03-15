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

def neutralise_factors(self, df, factors, risk_factors, schema): #(sector, beta, size, vol, etc)
        #F= a + BX + e
        #,where F is the raw factor signal, X is exposures risk factors, B is the regression coefficients
        #,a to fit cross-sectional mean so that residual mu = 0 and and e is the 'pure' factor (X^T @ e = 0)
        total_dates = df.select(pl.col("date").n_unique()).collect().item()
        print(f"[INFO]Running cross-sectional regressions across {total_dates:,} dates: ")
        print(f"[INFO]{len(factors)} factors against {len(risk_factors)} risk factors")

        X_cols = ["intercept"]+risk_factors
        y_cols = factors
        
        df = df.with_columns(
            [pl.col(c).fill_null(pl.col(c).mean().over("industry")).fill_null(pl.col(c).mean())
             for c in risk_factors]
            +
            [pl.lit(1.0).alias("intercept")]
        )
        train_df = df.drop_nulls(subset=y_cols)
        
