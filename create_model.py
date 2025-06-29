import os
import numpy as np
import json
import pathlib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from typing import Tuple
import pandas
import pandas as pd
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.metrics import (mean_absolute_error, 
                             mean_absolute_percentage_error, 
                             mean_squared_error, 
                             r2_score)

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv" # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, 
    demographics_path: str, 
    sales_column_selection: List[str], 
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv(demographics_path,
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y

def model_evaluation(model, 
                     X_test: pd.DataFrame, 
                     y_test: pd.Series) -> Tuple[dict, pd.Series]:

    """
    Generate the regression metrics for a given model.

    Args:


    Return:
    
    """

    # Predictions
    pred = model.predict(X_test)

    # Mean Absolute error
    mae = mean_absolute_error(y_true=y_test, y_pred=pred)

    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=pred)

    # Mean Squared Error
    mse = mean_squared_error(y_true=y_test, y_pred=pred)

    # Root Mean Squared Error
    rmse=np.sqrt(mse)

    # R-squared
    r_squared = r2_score(y_true=y_test, y_pred=pred)

    # metrics dict
    metrics = {
        "MAE": mae,
        "MAPE": mape,
        "MSE": mse,
        "RMSE": rmse,
        "R-Squared": r_squared
    }

    return metrics, pred

def plot_regression_results(
                            X:pd.DataFrame, 
                            y:pd.Series, 
                            pred:pd.Series,
                            output_dir=OUTPUT_DIR, 
                            prefix="regression"
                            ):
    """
    Generate a scatter plot with a regression line and residuals plot.

    Args:
    
    - X: array-like or input Dataframe with one or more features.
    - y: array-like or pandas series with the ground truth numbers.
    - pred: array-like or pandas series with the predicted values of the model.
    - output_dir: output directory.
    - prefix: prefix to the name of the files.

    Return:
    Image files on the specified output_dir.
    """

    # Residuals    
    y_pred = pred
    residuals = y - y_pred
    
    # Create JointGrid manually (bypassing broken jointplot(kind="reg"))
    g = sns.JointGrid(x=y, y=y_pred, height=6)
    g.plot(sns.scatterplot, sns.histplot)

    # Add regression line manually using regplot
    sns.regplot(x=y, y=y_pred, ax=g.ax_joint, scatter=False, color="red")

    reg_path = os.path.join(output_dir, f"{prefix}_scatterplot.png")
    g.figure.savefig(reg_path)


    # Residuals plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color='blue')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel("Predictions")
    plt.ylabel("Residuals")
    plt.title("Residuals plot")
    plt.tight_layout()
    resid_path = os.path.join(output_dir, f"{prefix}_residuals.png")
    plt.savefig(resid_path)
    plt.close()

    print(f"Plots delivered in:\n - {reg_path}\n - {resid_path}")


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)

    model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                   neighbors.KNeighborsRegressor()).fit(
                                       x_train, y_train)
    
    # Model Evaluation
    metrics, pred = model_evaluation(model=model, 
                               X_test=_x_test, 
                               y_test=_y_test)
    
    plot_regression_results(X=_x_test, y=_y_test, pred=pred)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Saving the metrics
    json.dump(metrics,
              open(output_dir / "metrics.json", 'w'))

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_features.json", 'w'))


if __name__ == "__main__":
    main()
