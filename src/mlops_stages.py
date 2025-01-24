import time
import warnings
from typing import Any

import pandas as pd
from exectimeit import timeit
from joblib import Parallel, delayed
from pandas import DataFrame
from scipy.linalg import LinAlgWarning
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)

global_random_seed = 42
n_jobs = -1

def setNJobs(new_n_jobs):
    global n_jobs
    n_jobs = new_n_jobs

# stage 1: data cleaning
@timeit.exectime(5)
def clean_data(df: DataFrame, unuseful_columns: list) -> tuple[DataFrame, int]:
    """
    Clean the dataset by removing rows with missing values and duplicates.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - DataFrame: Cleaned dataset.
    - Quantity of removed rows
    """
    df_out = df.copy()

    # Remove rows with missing values
    df_out.dropna(inplace=True)

    # Remove duplicates
    df_out.drop_duplicates(inplace=True)

    # Remove unuseful columns
    if unuseful_columns:
        df_out.drop(columns=unuseful_columns, inplace=True)

    return df_out, df.shape[0] - df_out.shape[0]


# # stage 2: data preprocessing v0
# @timeit.exectime(5)
# def preprocess_data(df: DataFrame, class_name: str) -> tuple[DataFrame, list[Any]]:
#     """
#     Preprocess the dataset by encoding categorical variables and normalizing numerical variables.
#
#     Parameters:
#     - df: DataFrame containing the data.
#     - class_name: Name of the column representing the class labels.
#
#     Returns:
#     - DataFrame: Preprocessed dataset.
#     """
#     df_out = df.copy()
#
#     # Encode categorical variables
#     encoded_columns = []
#     le = LabelEncoder()
#     for column in df_out.columns:
#         if not df_out[column].dtype.kind in ['i', 'f']:
#             encoded_columns.append(column)
#             df_out[column] = le.fit_transform(df_out[column].astype(str))
#
#     # Separate features and target
#     X = df_out.drop(columns=[class_name])
#     y = df_out[class_name]
#
#     scaler = StandardScaler()
#     X_normalized = scaler.fit_transform(X)
#
#     # Apply PCA
#     pca = PCA(n_components=0.95)
#
#     X_reduced = pca.fit_transform(X_normalized)
#
#     # Combine reduced features with the label
#     X_reduced_df = pd.DataFrame(X_reduced, columns=[f'PC{i + 1}' for i in range(X_reduced.shape[1])])
#     final_df = pd.concat([X_reduced_df, y.reset_index(drop=True)], axis=1)
#
#     # print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} features")
#
#     return final_df, encoded_columns

# stage 2: data preprocessing v0
@timeit.exectime(5)
def preprocess_data(df: pd.DataFrame, class_name: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Preprocess the dataset by encoding categorical variables, removing low-variance features,
    removing outliers using Isolation Forest, and performing PCA for feature extraction.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - tuple:
        - DataFrame: Preprocessed dataset.
        - list[str]: List of encoded categorical columns.
    """
    df_out = df.copy()

    # Step 1: Encode categorical variables
    encoded_columns = []
    le = LabelEncoder()
    for column in df_out.columns:
        if not df_out[column].dtype.kind in ['i', 'f']:
            encoded_columns.append(column)
            df_out[column] = le.fit_transform(df_out[column].astype(str))

    # Step 2: Separate features and target
    X = df_out.drop(columns=[class_name])
    y = df_out[class_name]

    # Step 3: Remove low-variance features
    variance_filter = VarianceThreshold(threshold=0.1)  # Threshold can be adjusted
    X_high_variance = variance_filter.fit_transform(X)

    # Step 4: Remove outliers using Isolation Forest
    iso_forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=global_random_seed, n_jobs=n_jobs)
    outliers = iso_forest.fit_predict(X_high_variance)
    X_no_outliers = X_high_variance[outliers == 1]
    y_no_outliers = y[outliers == 1]

    # Step 5: Normalize the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_no_outliers)

    # Step 6: Apply PCA
    pca = PCA(n_components=0.90)  # Retain 95% of variance
    X_reduced = pca.fit_transform(X_normalized)

    # Combine reduced features with the label
    X_reduced_df = pd.DataFrame(X_reduced, columns=[f'PC{i + 1}' for i in range(X_reduced.shape[1])])
    final_df = pd.concat([X_reduced_df, y_no_outliers.reset_index(drop=True)], axis=1)

    return final_df, encoded_columns


# stage 3: model tuning
@timeit.exectime(3)
def tune_models(df: DataFrame, class_name: str) -> dict:
    """
    Tune the models by selecting the best hyperparameters using grid search.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - dict: Dictionary containing the best model name and the hyperparameters for the model.
    """

    models = {
        'LR': LogisticRegression(random_state=global_random_seed),
        'RandomForest': RandomForestClassifier(random_state=global_random_seed),
        'MLP': MLPClassifier(random_state=global_random_seed),
    }

    # Define hyperparameters for grid search
    param_grid = {
        'LR': {'solver': ['saga'], 'penalty': ['l1', 'l2']},
        'RandomForest': {'n_estimators': [50, 100, 200]},
        'MLP': {'hidden_layer_sizes': [(50,), (100,), (200,)]}
    }

    # Perform grid search for each model
    tuned_models = {}
    for model_name, model in models.items():
        print("Tuning model:", model_name, "in", n_jobs, "jobs")
        grid_search = GridSearchCV(model, param_grid[model_name], cv=5, n_jobs=n_jobs)
        grid_search.fit(df.drop(columns=[class_name]), df[class_name])
        tuned_models[model_name] = {
            'best_params': grid_search.best_params_,
            'best_model': grid_search.best_estimator_
        }

    return tuned_models


# stage 4: model evaluation and selection
@timeit.exectime(5)
def evaluate_models(df: DataFrame, class_name: str, tuned_models: dict) -> tuple:
    def evaluate_model(model_name, tuned_model):
        model = tuned_model['best_model']
        mcc = matthews_corrcoef(df[class_name], model.predict(df.drop(columns=[class_name])))
        return model_name, mcc

    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_model)(model_name, tuned_model) for model_name, tuned_model in tuned_models.items())
    results_dict = dict(results)

    best_model_name = max(results_dict, key=results_dict.get)
    best_model_params = tuned_models[best_model_name]['best_params']
    best_model_mcc = results_dict[best_model_name]
    return best_model_name, best_model_params, best_model_mcc