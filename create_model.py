import json
import os
import pathlib
import pickle
from typing import List  # noqa: UP035

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection, neighbors, pipeline, preprocessing
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str,
    demographics_path: str,
    sales_column_selection: List[str],  # noqa: UP006
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the target and feature data by merging sales and demographics.

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
    data = pd.read_csv(
        sales_path,
        usecols=sales_column_selection,
        dtype={"zipcode": str},
    )
    demographics = pd.read_csv(demographics_path, dtype={"zipcode": str})

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(
        columns="zipcode",
    )
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop("price")
    x = merged_data

    return x, y


def model_evaluation(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, pd.Series]:
    """
    Evaluates a regression model on the test dataset and computes key performance metrics.

    Args:
        model: A fitted regression model that implements a `.predict()` method.
        x_test (pd.DataFrame): Test feature set used for generating predictions.
        y_test (pd.Series): True target values corresponding to `x_test`.

    Returns:
        tuple:
            - dict: A dictionary containing regression metrics:
                - 'MAE' (float): Mean Absolute Error.
                - 'MAPE' (float): Mean Absolute Percentage Error.
                - 'MSE' (float): Mean Squared Error.
                - 'RMSE' (float): Root Mean Squared Error.
                - 'R-Squared' (float): Coefficient of determination (R² score).
            - pd.Series: The predicted values for `x_test`.

    """
    # Predictions
    pred = model.predict(x_test)

    # Mean Absolute error
    mae = mean_absolute_error(y_true=y_test, y_pred=pred)

    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=pred)

    # Mean Squared Error
    mse = mean_squared_error(y_true=y_test, y_pred=pred)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # R-squared
    r_squared = r2_score(y_true=y_test, y_pred=pred)

    # metrics dict
    metrics = {
        "MAE": mae,
        "MAPE": mape,
        "MSE": mse,
        "RMSE": rmse,
        "R-Squared": r_squared,
    }

    return metrics, pred


def plot_regression_results(
    y: pd.Series,
    pred: pd.Series,
    output_dir=OUTPUT_DIR,
    prefix="regression",
):
    """
    Generates and saves a scatter plot with regression line and a residuals plot.

    Args:
        y (pd.Series): Ground truth target values.
        pred (pd.Series): Predicted values from the regression model.
        output_dir (str, optional): Directory where the plots will be saved. Defaults to OUTPUT_DIR.
        prefix (str, optional): Prefix for the saved image filenames. Defaults to "regression".

    Returns:
        None. The function saves two image files:
            - A scatter plot with a regression line.
            - A residuals plot.
        Files are saved in the specified `output_dir` using the provided `prefix`.

    """
    # Residuals
    y_pred = pred
    residuals = y - y_pred

    # Create JointGrid manually (bypassing broken jointplot(kind="reg"))
    g = sns.JointGrid(x=y, y=y_pred, height=6)
    g.plot(sns.scatterplot, sns.histplot)

    # Add regression line manually using regplot
    sns.regplot(x=y, y=y_pred, ax=g.ax_joint, scatter=False, color="red")

    reg_path = os.path.join(output_dir, f"{prefix}_scatterplot.png")  # noqa: PTH118
    g.figure.savefig(reg_path)

    # Residuals plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color="blue")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.xlabel("Predictions")
    plt.ylabel("Residuals")
    plt.title("Residuals plot")
    plt.tight_layout()
    resid_path = os.path.join(output_dir, f"{prefix}_residuals.png")  # noqa: PTH118
    plt.savefig(resid_path)
    plt.close()

    print(f"Plots delivered in:\n - {reg_path}\n - {resid_path}")


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x,
        y,
        random_state=42,
    )

    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        neighbors.KNeighborsRegressor(),
    ).fit(x_train, y_train)

    # Model Evaluation
    metrics, pred = model_evaluation(model=model, X_test=_x_test, y_test=_y_test)

    plot_regression_results(X=_x_test, y=_y_test, pred=pred)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Saving the metrics
    with open(output_dir / "metrics.json", "w") as f:  # noqa: PTH123
        json.dump(metrics, f)

    # Output model artifacts: pickled model and JSON list of features
    with open(output_dir / "model.pkl", "wb") as f:  # noqa: PTH123
        pickle.dump(model, f)

    with open(output_dir / "model_features.json", "w") as f:  # noqa: PTH123
        json.dump(list(x_train.columns), f)


if __name__ == "__main__":
    main()
