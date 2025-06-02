# Reusable Functions. With an indication of the section where they were defined

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

######################################################
# From "3. Distance Metrics.ipynb"
######################################################



######################################################
# From "4. Optimal Clustering.ipynb"
######################################################

def clustering_predictions_strength(data, max_n_clusters = 10, tries_per_n_clusters = 100, test_size=0.5, plot=True):

    '''
    We divide the dataset used for clustering into a training set and a test set. We then compare the cluster 
    assignments of the test set obtained in two ways:
    - By predicting with the model trained on the training set.
    - By fitting a new clustering model directly on the test set.
    The more appropriate the chosen number of clusters is, the higher the percentage of observations that 
    should be assigned to the same clusters by both methods.

    This method is inspired by The Hundred-Page Machine Learning Book by Andriy Burkov, Section 9.2.3.
    '''

    clusters_stats = []

    for n_clusters in range(2, max_n_clusters + 1):

        cluster_scores = []

        for i in range(tries_per_n_clusters):

            train_df, test_df = train_test_split(data, test_size=test_size)

            model_train = KMeans(n_clusters = n_clusters)
            model_train = model_train.fit(train_df)

            model_test = KMeans(n_clusters = n_clusters)
            model_test = model_test.fit(test_df)

            test_lables_from_model_train = model_train.predict(test_df)

            for n_clust in np.unique(model_test.labels_):
                labels_in_test_cluster = test_lables_from_model_train[model_test.labels_==n_clust]
                _, counts = np.unique(labels_in_test_cluster, return_counts=True)
                cluster_score = np.sum(counts**2)/len(labels_in_test_cluster)**2
                cluster_scores.append(cluster_score)

        clusters_stats.append({'n_clusters': n_clusters,
                               'min_score': np.min(cluster_scores),
                               'mean_score': np.mean(cluster_scores)})
        
    if plot:
        x = [c["n_clusters"] for c in clusters_stats]
        y = [c["min_score"] for c in clusters_stats]
        z = [c["mean_score"] for c in clusters_stats]

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker='o', linestyle='-', label='Min cluster score', color='blue')
        plt.plot(x, z, marker='s', linestyle='--', label='Mean cluster score', color='orange')
        plt.title('Number of Clusters - Prediction Strength Comparison')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()
        plt.show()

    return clusters_stats