# Reusable Functions. With an indication of the section where they were defined

import pandas as pd
import numpy as np

######################################################
# From "2. Denoising and Detoning.ipynb"
######################################################

def calculate_returns(chosen_assets = None, min_obs_threshold = None, remove_na = True, print_high_corr = True):
    # chosen_assets is a list of symbols / ticker, for which we want to calculate returns
    # min_obs_threshold is the minimum amount of observations we expect a symbol to have in the time period
    # remove_na - do we want to remove rows, for which a "na" value is present (so returns for 
    #   each stock are calculated on the same period)
    # print_high_corr - print the highest correlations, as these could be overlapping data (e.g. GOOG and GOOGL),
    #   that could indicate duplicated data

    # Step 0 - loading the raw data
    # df_stocks_info = pd.read_csv('stocks_info.csv')
    df_candles = pd.read_csv('stocks_candles.csv')

    # Step 1 - choose subsets of assets you are interested
    if chosen_assets is not None:
        df_candles = df_candles[df_candles['Symbol'].isin(chosen_assets)]

    # Step 2 - redefine the candles DataFrame
    df_candles['Date'] = pd.to_datetime(df_candles['Date'], utc=True).dt.date
    df_candles = df_candles.groupby(['Date', 'Symbol'])['Close'].mean().reset_index()
    df_candles = df_candles.pivot(index='Date', columns='Symbol', values='Close')

    # Step 3 (Optional) - remove assets with multiple missing observations
    if min_obs_threshold is not None:
        df_candles = df_candles.loc[:, df_candles.isnull().mean() < 1 - min_obs_threshold]

    # Step 4 (Optional) - remove missing values 
    if remove_na:
        df_candles.dropna(inplace=True)

    # Step 5 (Optional) - check and fix negative values (before np.log is applied)
    # - if interest rates are analysed, we should give possibility to add e.g. 3% to the series
    df_candles = df_candles.apply(lambda x: x.where(x >= 0, np.nan))

    # Step 6 - calculate the log returns
    nan_mask = df_candles.isna() # Some steps in case missing values were not removed
    df_returns = df_candles.ffill()
    df_returns = np.log(df_returns).diff()
    df_returns[nan_mask] = np.nan
    df_returns = df_returns.iloc[1:, :]

    # Step 7 (Optional) - Printing highly correlated assets
    if print_high_corr:
        corr_matrix = df_returns.corr()
        corr_long = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_long = corr_long.stack()
        corr_long.name = 'Corr'
        corr_long.index.names = ['Asset 1', 'Asset 2']
        corr_long = corr_long.reset_index()
        corr_long = corr_long.reindex(corr_long['Corr'].abs().sort_values(ascending=False).index)
        print(corr_long.head(10))

    return df_returns

