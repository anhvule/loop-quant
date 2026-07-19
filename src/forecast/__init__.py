"""Multi-day price-path forecasting for equities.

This package is a *research* add-on layered on top of Loop Quant's data plumbing.
It reuses the SQLite `candles` table, the `IndicatorEngine`, and the `SignalEngine`,
but is entirely separate from the live trading loop and the optimizer -- importing
`src.forecast` never touches config, bounds, or any order path.

It produces a forecast of a stock's *price path* over the next N trading days by two
independent methods (Monte Carlo GBM and ARIMA). These are statistical baselines
built from historical drift and volatility, not predictions of the real future.

NOT INVESTMENT ADVICE.
"""
