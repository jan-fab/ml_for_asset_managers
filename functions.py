# Reusable Functions. With an indication of the section where they were defined

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, mean_squared_error

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

def plot_corr_matrix(corr_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        cmap="RdBu_r",       
        vmin=-1, 
        vmax=1,     
        center=0,
        annot=False,
        square=True
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

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

    if max_n_clusters > int(np.floor(data.shape[0] * min(test_size, 1-test_size))):
        max_n_clusters = int(np.floor(data.shape[0] * min(test_size, 1-test_size)))
        print(f'Info: max_n_clusers limited to {max_n_clusters}. Change test_size parameter to increase it.')

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
                               'mean_score': np.mean(cluster_scores),
                               'std_score': np.std(cluster_scores)})

    if plot:
        x = [c["n_clusters"] for c in clusters_stats]
        y_min = [c["min_score"] for c in clusters_stats]
        y_mean = [c["mean_score"] for c in clusters_stats]
        y_std = [c["std_score"] for c in clusters_stats]

        plt.figure(figsize=(8, 5))

        # Plot min line (optional, no error bars)
        plt.plot(x, y_min, marker='o', linestyle='-', label='Min cluster score', color='blue')

        # Plot mean ± std with error bars
        plt.errorbar(x, y_mean, yerr=y_std, fmt='--s', color='orange', ecolor='orange',
                    elinewidth=1, capsize=3, label='Mean cluster score ± std')

        plt.title('Number of Clusters - Prediction Strength Comparison')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score (higher = better prediction strength)')
        plt.grid(True)
        plt.legend()
        plt.show()

    return clusters_stats

def clustering_elbow_plot(data, max_n_clusters=10, tries_per_n_clusters=100, plot=True):
    """
    Computes and plots average inertia and silhouette score ± std dev across multiple 
    KMeans cluster counts.

    Returns:
        dict: {
            'inertia_mean': [...],
            'inertia_std': [...],
            'silhouette_mean': [...],
            'silhouette_std': [...]
        }
    """
    results = {'inertia_mean': [], 'inertia_std': [],
               'silhouette_mean': [], 'silhouette_std': []}

    for k in range(1, max_n_clusters + 1):
        inertias, silhouettes = [], []

        for _ in range(tries_per_n_clusters):
            kmeans = KMeans(n_clusters=k, n_init=1)
            labels = kmeans.fit_predict(data)
            inertias.append(kmeans.inertia_)

            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < data.shape[0]:
                try:
                    silhouettes.append(silhouette_score(data, labels))
                except:
                    silhouettes.append(np.nan)

        results['inertia_mean'].append(np.mean(inertias))
        results['inertia_std'].append(np.std(inertias))

        if silhouettes:
            sil_mean = np.nanmean(silhouettes)
            sil_std = np.nanstd(silhouettes)
        else:
            sil_mean, sil_std = np.nan, np.nan

        results['silhouette_mean'].append(sil_mean)
        results['silhouette_std'].append(sil_std)

    if plot:
        x = list(range(1, max_n_clusters + 1))
        fig, ax1 = plt.subplots(figsize=(10, 5))
        for i in range(1, max_n_clusters):
            ax1.axvline(x=i, color='lightgray', linestyle='--', linewidth=0.5)

        # Plot inertia
        ax1.errorbar(x, results['inertia_mean'], yerr=results['inertia_std'],
                     fmt='-o', color='blue', ecolor='blue', capsize=3,
                     label='Inertia ± std (lower is better)')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Avg Inertia (lower is better)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticks(x)

        # Plot silhouette
        ax2 = ax1.twinx()
        ax2.errorbar(x, results['silhouette_mean'], yerr=results['silhouette_std'],
                     fmt='--s', color='orange', ecolor='orange', capsize=3,
                     label='Silhouette ± std (higher is better)')
        ax2.set_ylabel('Avg Silhouette (higher is better)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Annotate silhouette mean / std above x-axis ticks
        y_min, y_max = ax2.get_ylim()
        y_text = y_min + 0.01 * (y_max - y_min)

        for xi, (mean, std) in enumerate(zip(results['silhouette_mean'], results['silhouette_std']), start=1):
            if not np.isnan(mean) and std > 0:
                ratio = mean / std
                ax2.text(xi, y_text, f"{ratio:.1f}", ha='center', va='bottom',
                         fontsize=8, color='green', rotation=45)

        # Add explanatory text
        ax2.text(0.5, y_min + 0.1 * (y_max - y_min),
                 "Silhouette mean / std (higher is better)",
                 transform=ax2.transData, ha='left', va='bottom',
                 fontsize=9, color='green', fontstyle='italic')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

        plt.title(f'KMeans Clustering: Inertia & Silhouette (±1 Std, {tries_per_n_clusters} runs)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results


def sort_clustered_corr_matrix(corr_matrix, labels, label_spacing = 1, plot=True):
    """
    Sort and plot a correlation matrix with subtle cluster boundaries and optional label spacing.

    Parameters:
        corr_matrix (pd.DataFrame): Correlation matrix.
        labels (np.array): Cluster labels corresponding to columns/rows.
        plot (bool): Whether to plot the matrix.
        label_spacing (int): Interval for showing axis tick labels.
        show_labels (bool): Whether to show tick labels.

    Returns:
        pd.DataFrame: Sorted correlation matrix.
    """
    corr_mean = []
    for cluster in np.unique(labels):
        cluster_indices = np.where(labels == cluster)[0]
        sub_corr = corr_matrix.iloc[cluster_indices, cluster_indices]
        means = sub_corr.mean(axis=1)
        for idx, mean_val in zip(cluster_indices, means):
            corr_mean.append((idx, cluster, mean_val))

    corr_df = pd.DataFrame(corr_mean, columns=["idx", "cluster", "mean_corr"])
    sorted_indices = corr_df.sort_values(["cluster", "mean_corr"], ascending=[True, False])["idx"].to_numpy()
    sorted_corr = corr_matrix.iloc[sorted_indices, sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    if plot:
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(sorted_corr, cmap='coolwarm', center=0,
                         xticklabels=False, yticklabels=False,
                         cbar_kws={"shrink": .6},
                         vmin=-1, vmax=1)

        # Add cluster lines
        prev = 0
        for cluster in np.unique(sorted_labels):
            count = sorted_labels.count(cluster)
            if prev > 0:
                ax.axhline(prev, color='gray', linewidth=0.5, linestyle='-')
                ax.axvline(prev, color='gray', linewidth=0.5, linestyle='-')
            prev += count

        tick_labels = sorted_corr.columns
        ticks = np.arange(len(tick_labels))
        spaced_ticks = ticks[::label_spacing]
        ax.set_xticks(spaced_ticks)
        ax.set_yticks(spaced_ticks)
        ax.set_xticklabels(tick_labels[::label_spacing], rotation=90, fontsize=8)
        ax.set_yticklabels(tick_labels[::label_spacing], rotation=0, fontsize=8)

        plt.title("Clustered Correlation Matrix")
        plt.tight_layout()
        plt.show()

    return sorted_corr


def find_clustres_with_best_silhouette(dist_matrix, n_clusters):

    # Returns the best clustering for a given amount of clusters.
    
    # Using the signal-to-noise ratio as described in Marcos López de Prado's book.
    # This ratio (mean / std of silhouette scores) rewards consistent clustering quality
    # by favoring uniform silhouette scores (tight, stable clusters) and penalizing instability.

    # Silhouette balances internal compactness and external separation — making it a better 
    # objective for selecting a good fit than the interia.

    try:
        names = dist_matrix.index
    except:
        names = None

    best_model = None
    best_score = -1

    for i in range(100):
        model = KMeans(n_clusters = n_clusters)
        labels = model.fit_predict(dist_matrix)
        silh_samples = silhouette_samples(dist_matrix, labels)
        score = np.mean(silh_samples) / np.std(silh_samples) 

        if score > best_score:
            best_score = score
            best_model = model

    best_model.fit(dist_matrix)
    labels = best_model.labels_

    return labels, names

######################################################
# From "6. Feature Importance Analysis.ipynb"
######################################################

def plot_feature_importances(importances, method = '', top_n=20):
    """
    Plot the top N feature importances with error bars.

    Args:
        importances (DataFrame): DataFrame with 'mean' and 'std' columns for each feature.
        top_n (int): Number of top features to display.
    """

    imp_sorted = importances.sort_values('mean', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(imp_sorted.index[::-1], imp_sorted['mean'][::-1],
             xerr=imp_sorted['std'][::-1], color='skyblue', edgecolor='black')
    plt.xlabel("Normalized Mean Importance")
    plt.title(f"{method} Top {top_n} Feature Importances (Mean ± Std Error)")
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def residualize_features_by_cluster(X: pd.DataFrame, cluster_labels: np.ndarray):
    """
    Residualizes features within each cluster against features from other clusters (in order),
    replacing regressors with already-residualized features to preserve intra-cluster signal.

    Returns:
        pd.DataFrame with residualized features.
    """

    # Define the clusters as a dictionary
    features = X.columns.to_list()
    clusters = {}
    for feat, clust in zip(features, cluster_labels):
        clusters.setdefault(clust, []).append(feat)
    clusters = dict(sorted((k, sorted(v)) for k, v in clusters.items()))

    X_resid = X.copy()

    orthogonal_clusters = []

    for _, cluster_feats in clusters.items():

        if len(orthogonal_clusters) == 0:
            orthogonal_clusters += cluster_feats
            continue

        for feature in cluster_feats:
            y = X[feature]
            X_orthogonal_clusters = X_resid[orthogonal_clusters]
            model = LinearRegression().fit(X_orthogonal_clusters, y)
            y_hat = model.predict(X_orthogonal_clusters)
            residual = y - y_hat
            X_resid[feature] = residual

        orthogonal_clusters += cluster_feats

    return X_resid


def feat_imp_mdi(X, y, feat_labels = None, classifier=True, plot_results = True):
    """
    Computes mean and standard error of MDI feature importances using a Random Forest model.
    Allows for defining clusters of features.

    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series or np.ndarray): Target variable.
        classifier (bool): If True, uses RandomForestClassifier; otherwise, RandomForestRegressor.

    Returns:
        pd.DataFrame: Feature importances with columns 'mean' and 'std'.
    """
    
    # Fit the model
    model_cls = RandomForestClassifier if classifier else RandomForestRegressor

    model = model_cls(
        n_estimators=1000,
        max_features='sqrt',
        bootstrap=True,
        oob_score=False,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Calculate feature importances based on the trees
    all_importances = np.array([
        tree.feature_importances_ for tree in model.estimators_
    ])

    all_importances[all_importances == 0] = np.nan # Ignore unused features

    index = X.columns

    if feat_labels is not None:
    # Get unique sorted labels
        unique_labels = sorted(np.unique(feat_labels))

        # Sum columns grouped by labels
        sums_by_label = []
        for label in unique_labels:
            cols = np.where(feat_labels == label)[0]
            col_sum = np.nansum(all_importances[:, cols], axis=1)  # sum across selected columns
            sums_by_label.append(col_sum)

        all_importances = np.transpose(sums_by_label)

        index = ['C_' + str(label) for label in unique_labels]


    mean_imp = np.nanmean(all_importances, axis=0)
    std_error = np.nanstd(all_importances, axis=0) / np.sqrt(all_importances.shape[0])

    mean_imp /= mean_imp.sum()

    imp_summary = pd.DataFrame({
        'mean': mean_imp,
        'std': std_error
    }, index=index)

    # Optional plot
    if plot_results:
        plot_feature_importances(imp_summary, method = 'MDI -', top_n = 21)

    return imp_summary


def feat_imp_mda(X, y, feat_labels = None, n_splits=10, classifier=True, plot_results = True):
    """
    Computes mean and standard error of MDA feature importances using a Random Forest model, based on
    log_loss for classification and mean_square_error for regression.

    Allows for defining clusters of features.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.array): Target variable.
        classifier (bool): Whether to use classification or regression model.
        n_splits (int): Number of KFold splits.

    Returns:
        pd.DataFrame: DataFrame with 'mean' and 'std' of MDA importances.
    """
    # Choose model
    model_cls = RandomForestClassifier if classifier else RandomForestRegressor
    model = model_cls(
        n_estimators=1000,
        max_features='sqrt',
        bootstrap=True,
        oob_score=False,
        n_jobs=-1
    )

    # Set up dummy labels if no clusters were defined
    if feat_labels is None:
        feat_labels = np.array(range(X.shape[1]))
        unique_labels = feat_labels
        index = X.columns
    else:
        unique_labels = sorted(np.unique(feat_labels), key=lambda x: int(x))
        index = ['C_' + str(label) for label in unique_labels]

    # Set up cross-validation
    cv = KFold(n_splits=n_splits, shuffle=True)
    base_score = pd.Series(index=range(n_splits), dtype=float)

    # Set up a DataFrame for collecting the results
    perm_scores = pd.DataFrame(index=range(n_splits), columns=index, dtype=float)

    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        if classifier:
            prob = model.predict_proba(X_test)
            base_score[i] = -log_loss(y_test, prob, labels=model.classes_)
        else:
            pred = model.predict(X_test)
            base_score[i] = mean_squared_error(y_test, pred)

        for label in unique_labels:

            cols_to_shuffle = (
                [X.columns[label]] if feat_labels is None
                else X.columns[np.where(feat_labels == label)[0]]
            )

            X_test_shuffled = X_test.copy()

            for col in cols_to_shuffle:
                X_test_shuffled[col] = np.random.permutation(X_test_shuffled[col].values)

            if classifier:
                prob_shuffled = model.predict_proba(X_test_shuffled)
                perm_scores.loc[i, index[label]] = -log_loss(y_test, prob_shuffled, labels=model.classes_)
            else:
                pred_shuffled = model.predict(X_test_shuffled)
                perm_scores.loc[i, index[label]] = mean_squared_error(y_test, pred_shuffled)

    # Calculate MDA importances
    mda = (perm_scores.subtract(base_score, axis=0)).div(perm_scores, axis = 0)

    imp_summary = pd.concat({
        'mean': mda.mean(),
        'std': mda.std(ddof=0) * mda.shape[0] ** -0.5  # standard error (CLT)
    }, axis=1)

    # Optional plot
    if plot_results:
        plot_feature_importances(imp_summary, method = 'MDA -', top_n = 21)

    return imp_summary