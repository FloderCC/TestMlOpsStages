import warnings
from typing import Any

from exectimeit import timeit
from pandas import DataFrame
from scipy.linalg import LinAlgWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from pymfe.mfe import MFE

# Suppress only ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)

global_random_seed = 42

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


# stage 2: data preprocessing
@timeit.exectime(5)
def preprocess_data(df: DataFrame) -> tuple[DataFrame, list[Any]]:
    """
    Preprocess the dataset by encoding categorical variables and normalizing numerical variables.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - DataFrame: Preprocessed dataset.
    """
    df_out = df.copy()

    # Encode categorical variables
    encoded_columns = []
    le = LabelEncoder()
    for column in df_out.columns:
        if not df_out[column].dtype.kind in ['i', 'f']:
            encoded_columns.append(column)
            df_out[column] = le.fit_transform(df_out[column].astype(str))

    # Normalize numerical variables
    for column in df_out.columns:
        if df_out[column].dtype == 'float64':
            df_out[column] = (df_out[column] - df_out[column].mean()) / df_out[column].std()

    return df_out, encoded_columns


# stage 3: data analysis
@timeit.exectime(5)
def analyze_data(df: DataFrame, class_name: str) -> dict:
    """
    Analyze the dataset using the pymfe library

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - dict: Dictionary containing the analysis results.
    """

    dataset_description_header = ["attr_to_inst", "class_ent", "eq_num_attr", "gravity", "inst_to_attr",
                                  "nr_attr", "nr_bin", "nr_class", "nr_cor_attr", "nr_inst", "nr_norm",
                                  "nr_outliers", "ns_ratio"]

    # Exclude the class column
    df_without_class = df.drop(columns=[class_name])

    # Initialize MFE and extract features
    mfe = MFE(groups=["concept", "general", "info-theory", "statistical"])
    mfe.fit(df_without_class.values, df[class_name].values)
    ft = mfe.extract()

    # Filter the extracted features to include only those in dataset_description_header
    results = {name: value for name, value in zip(ft[0], ft[1]) if name in dataset_description_header}
    return results


# stage 4: model tuning
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
        'RidgeClassifier': RidgeClassifier(random_state=global_random_seed),
        'RandomForest': RandomForestClassifier(random_state=global_random_seed),
        'MLP': MLPClassifier(random_state=global_random_seed),
    }

    # Define hyperparameters for grid search
    param_grid = {
        'RidgeClassifier': {'alpha': [0.1, 1.0, 10.0]},
        'RandomForest': {'n_estimators': [50, 100, 200]},
        'MLP': {'hidden_layer_sizes': [(50,), (100,), (200,)]}
    }

    # Perform grid search for each model
    tuned_models = {}
    for model_name, model in models.items():
        grid_search = GridSearchCV(model, param_grid[model_name], cv=5)
        grid_search.fit(df.drop(columns=[class_name]), df[class_name])
        tuned_models[model_name] = {
            'best_params': grid_search.best_params_,
            'best_model': grid_search.best_estimator_
        }

    return tuned_models


# stage 5: model evaluation and selection
@timeit.exectime(5)
def evaluate_models(df: DataFrame, class_name: str, tuned_models: dict) -> tuple:
    """
    Evaluate the tuned models by calculating the mcc of each model.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.
    - tuned_models: Dictionary containing the best model name and the hyperparameters for the model.
    """
    results = {}
    for model_name, tuned_model in tuned_models.items():
        model = tuned_model['best_model']
        mcc = matthews_corrcoef(df[class_name], model.predict(df.drop(columns=[class_name])))
        results[model_name] = mcc

    # returning th best model's name with its best_params and mcc
    best_model_name = max(results, key=results.get)
    best_model_params = tuned_models[best_model_name]['best_params']
    best_model_mcc = results[best_model_name]
    return best_model_name, best_model_params, best_model_mcc