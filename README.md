# Sortino Ratio NIFTY 50 Portfolio Optimization

## Description
This Project uses SciPy's SLSQP to optimize for Sortino Ratio and generate weights to actively manage NIFTY 50 Portfolio. 


## Data 
The pool consists of equities which were in NIFTY 50 at the end of 2009. Equity data is fetched using the yfinance API.


## Steps 
1. Fetch equity returns data from t-1 to t-n
2. Optimize the data to get the Maximum Sortino Ratio (Rf is considered to be 0) subject to the following constraints
   - Weights should be equal to 1 
   - Portfolio turnover ratio should be less than the specified limit
   - Sector weights should be less than the specified limits
3. Run simulation on the current month t and save the returns and update the new portfolio weights 


## Results
The optimized portfolio gave superior returns compared NIFTY 50 over the time period. The portfolio on average had 1.54% higher monthly standard deviation but gave 28.69% higher mean monthly returns than the benchmark.

![](https://github.com/saidattsamonkar/POPT/blob/main/Assets/Sim.png)
