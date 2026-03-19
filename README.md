not refined

# Data Pipeline
- Price-derived factors are temporarily used as a proxy for fundamental factors (Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and Momentum Everywhere. The Journal of Finance)
- Methods based on Fama-MacBeth with own changes (shrinkage, signals instead of time-series regression for step 1)
- Regression used to orthogonalise signals against various risk factors to generate "purified" signals

# Applications
- Hedging: Long BMBL -> unintended bet on VAL and MKT -> Long X BMBL and Short Y SRPT -> isolate idio alpha
- Risk decomposition, [portfolio construction](https://github.com/soonyz06/Factor_Model_Prototype), performance attribution, scenario anaysis, MVO  (kxk instead of nxn), etc
- OLS: F = (B'B)^-1 B'R ≈ Factor-mimicking portfolios: F = WR

# Factor Returns
![me](img/Figure_1.png) 
![value](img/Figure_2.png)
- Differ quite a bit from actual values (Orange~VAL) due to only small sample size, different 'definition' of factors, i haven't scaled by MC (WLS), i prob overlooked something in the logic, etc

# Performance Attribution
Portfolio 1
![1](img/SPY.png)
Portfolio 2
![2](img/mom.png)
Portfolio 3
![3](img/a.png)

