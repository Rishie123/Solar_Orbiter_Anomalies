import pandas as pd # Import pandas library for data manipulation
from sklearn.ensemble import IsolationForest # Import Isolation Forest model for anomaly detection
import shap  # Import SHAP library for model interpretation
import numpy as np # Import numpy library for numerical operations
import matplotlib.pyplot as plt # Import matplotlib library for plotting
from memory_profiler import profile, memory_usage # Import memory profiler for memory profiling


def load_data(path):
    """Load data from CSV file to DataFrame."""
    return pd.read_csv(path)

def save_results(data, path):
    """Save the DataFrame to a CSV file at the specified path."""
    data.to_csv(path, index=False)

def fit_isolation_forest(data, n_estimators=100, contamination=0.01):
    """
    Fit an Isolation Forest model to the provided data. Useful for detecting outliers.
    
    Args:
        data (pandas.DataFrame): Data to fit the model on.
        n_estimators (int): Number of trees in the forest.
        contamination (float): The proportion of outliers in the data set.
    
    Returns:
        IsolationForest: A fitted Isolation Forest model.
    """
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    model.fit(data)
    return model

def calculate_shap_values(data, model):
    """
    Calculate SHAP values to provide interpretable explanations for the predictions made by Isolation Forest.
    
    Args:
        data (pandas.DataFrame): Data used for generating explanations.
        model (IsolationForest): The fitted Isolation Forest model.
    
    Returns:
        DataFrame: SHAP values for each feature for each instance in the data.
    """
    # Create an explainer for the model
    explainer = shap.TreeExplainer(model)
    # Calculate SHAP values for all data
    shap_values = explainer.shap_values(data)
    # Return as a DataFrame for usability
    return pd.DataFrame(shap_values, columns=data.columns)

