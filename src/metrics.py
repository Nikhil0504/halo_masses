import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Custom metrics functions
def calculate_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_nmad(y_true, y_pred):
    absolute_deviation = np.abs(y_pred - y_true)
    return 1.4826 * np.median(absolute_deviation)

def calculate_outlier_fraction(y_true, y_pred, threshold=0.5):
    absolute_deviation = np.abs(y_pred - y_true)
    return np.mean(absolute_deviation > threshold)

def evaluate_model(y_true, y_pred):
    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": calculate_mae(y_true, y_pred),
        "Bias": calculate_bias(y_true, y_pred),
        "NMAD": calculate_nmad(y_true, y_pred),
        "R^2 Score": r2_score(y_true, y_pred),
        "Outlier Fraction": calculate_outlier_fraction(y_true, y_pred, threshold=3*calculate_nmad(y_true, y_pred))
    }
    return metrics
