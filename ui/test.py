import sys
import os
import concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connector.Connector import Connector
from ui.ML_ui import Ui_MLWindow
from utils.FileUtil import FileUtil
import traceback
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


connector = Connector()

def connectdb():
    connector.server = "localhost"
    connector.port = 3306
    connector.database = "factorymanagement"
    connector.username = "root"
    connector.password = "@Obama123"
    connector.connect()

def find_optimal_clusters(data, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    optimal_clusters = np.argmin(np.diff(inertias)) + 3
    return optimal_clusters

def evaluate_feature_importance_for_clustering(X, y):
    model = KMeans(n_clusters=3, random_state=0)
    model.fit(X)
    result = permutation_importance(model, X, y, scoring='neg_mean_squared_error', n_jobs=-1)
    sorted_indices = result.importances_mean.argsort()[::-1]
    sorted_features = X.columns[sorted_indices]
    return sorted_features[:3]

def run():
    try:
        connectdb()
        sql = '''SELECT sub_ID, sub_age, sub_health_h, sub_commitment_h, sub_perceptiveness_h, sub_dexterity_h,
                        sub_sociality_h, sup_sub_age_diff, sub_goodness_h, sub_strength_h, sub_openmindedness_h, actual_efficacy_h
                        FROM factory WHERE record_comptype = "Efficacy"'''
        df = connector.queryDataset(sql)
        if df is not None:
            filtered_sampled_df = df[['sub_age', 'sub_health_h', 'sub_commitment_h', 'sub_perceptiveness_h', 'sub_dexterity_h',
                                              'sub_sociality_h', 'sub_goodness_h', 'sub_strength_h', 'sub_openmindedness_h',
                                              'sup_sub_age_diff', 'actual_efficacy_h']]
            X = filtered_sampled_df.drop('actual_efficacy_h', axis=1)
            y = filtered_sampled_df['actual_efficacy_h']

            print("PERMUTATION IMPORTANCE")
            best_features = evaluate_feature_importance_for_clustering(X, y)
            print(f"Best features: {best_features}")

            filtered_df = df[['sub_ID', 'sub_age', 'sub_health_h', 'sub_commitment_h', 'sub_perceptiveness_h', 'sub_dexterity_h',
                              'sub_sociality_h', 'sub_goodness_h', 'sub_strength_h', 'sub_openmindedness_h',
                              'sup_sub_age_diff', 'actual_efficacy_h']]
            
            numeric_features = [feature for feature in best_features if pd.api.types.is_numeric_dtype(filtered_df[feature])]

            worker_stats = filtered_df.groupby('sub_ID').agg(
                mean_efficacy=('actual_efficacy_h', 'mean'),
                **{f'mean_{feature}': (feature, 'mean') for feature in numeric_features}
            ).reset_index()

            data = worker_stats[['mean_efficacy'] + [f'mean_{feature}' for feature in numeric_features]]
            print("CLUSTER")
            optimal_clusters = find_optimal_clusters(data)
            print(f"Optimal number of clusters: {optimal_clusters}")

            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            worker_stats['efficacy_cluster'] = kmeans.fit_predict(data)

            print(worker_stats.head())
        else:
            print("No data loaded.")
    except Exception as e:
        traceback.print_exc()
        print(filtered_df.dtypes)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_clustering(worker_stats, numeric_features):
    """
    Create a 3D scatter plot of the clustering results using the top 3 numeric features.

    Args:
        worker_stats (pandas.DataFrame): DataFrame containing the worker statistics and clustering results.
        numeric_features (list): List of numeric feature names to use for the 3D plot.

    Returns:
        None (displays the 3D plot)
    """
    # Check if there are at least 3 numeric features
    if len(numeric_features) < 3:
        print("At least 3 numeric features are needed for 3D visualization.")
        return

    # Extract the top 3 numeric features and the cluster labels
    x, y, z = numeric_features[:3]
    labels = worker_stats['efficacy_cluster']

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster as a separate scatter plot
    for cluster in np.unique(labels):
        cluster_data = worker_stats[labels == cluster]
        ax.scatter(cluster_data[f'mean_{x}'], cluster_data[f'mean_{y}'], cluster_data[f'mean_{z}'], label=f'Cluster {cluster}')

    ax.set_xlabel(f'Mean {x}')
    ax.set_ylabel(f'Mean {y}')
    ax.set_zlabel(f'Mean {z}')
    ax.legend()

    plt.show()